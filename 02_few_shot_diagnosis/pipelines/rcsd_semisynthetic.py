"""Auditable RCSD background selection and controlled pattern construction.

Derived from Gao Haoyu's pinned RCSD background and controlled-v2 rules.
No learned generator or claim of measured mechanical failure is involved.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(tag):
    for token, name in [('ZI', 'axial'), ('TI', 'temperature'), ('XI', 'vibration'),
                        ('PDI', 'differential_pressure'), ('SI', 'speed')]:
        if token in tag:
            return name
    return 'pressure'


def background_limit(tag):
    kind = category(tag)
    if kind == 'axial': return -.5, .5
    if kind == 'temperature': return None, 120.
    if kind == 'vibration': return None, 89.
    return {'75PI808.pv': (9., None), '75PI823.pv': (1.8, None),
            '75PI870.pv': (4.2, None), '75PI845.pv': (.84, None),
            '75PDI853.pv': (None, 3.), '75SI865R.pv': (6050., 8500.)}[tag]


def background_plan(raw, times, config):
    """Match the colleague's causal warm-up and chronological 480-row blocks."""
    if not isinstance(raw, pd.DataFrame) or not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError('Finite numeric original observations required')
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    raw = raw.reset_index(drop=True)
    if len(times) != len(raw) or times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError('Unique increasing original clock required')
    first, second = config['split_boundaries']
    split = pd.Series(np.select([times < first, times < second], ['source', 'support_pool'], default='query'))
    within = pd.Series(True, index=raw.index)
    for tag in raw:
        low, high = background_limit(tag)
        if low is not None: within &= raw[tag].ge(low)
        if high is not None: within &= raw[tag].le(high)
    gap = times.diff().ne(pd.Timedelta(seconds=config['sample_seconds']))
    reset = gap | split.ne(split.shift()) | ~within
    streak = within.astype(int).groupby(reset.cumsum()).cumsum()
    candidate = within & streak.ge(config['stable_history_points'])
    segment = (gap | split.ne(split.shift()) | candidate.ne(candidate.shift())).cumsum()
    rows, tails = [], []
    size = config['source_block_points']
    for number, part in raw.loc[candidate].groupby(segment[candidate]):
        ids = part.index.to_numpy()
        for start in range(0, len(ids), size):
            take = ids[start:start + size]
            if len(take) != size:
                tails.append({'segment': int(number), 'points': len(take), 'reason': 'short_tail'})
                continue
            if not np.all(np.diff(take) == 1): raise ValueError('Background has nonconsecutive source rows')
            rows.append({'block_id': f'B{len(rows):04d}', 'split': split.iloc[take[0]],
                         'first_source_row': int(take[0]), 'last_source_row': int(take[-1]),
                         'start_time': str(times.iloc[take[0]]), 'end_time': str(times.iloc[take[-1]]),
                         'points': size})
    metadata = pd.DataFrame({'source_row_id': raw.index, 'timestamp': times, 'split': split,
                             'baseline_candidate': candidate, 'segment_id': segment})
    return pd.DataFrame(rows), metadata, tails


def training_amplitudes(raw, metadata, config):
    mask = metadata['split'].eq('source') & metadata['baseline_candidate']
    train = raw.loc[mask]
    if train.empty: raise ValueError('No eligible source training observations')
    result, rows = {}, []
    floors = {'temperature': .1, 'vibration': .1, 'axial': .001, 'pressure': .01,
              'differential_pressure': .01, 'speed': 1.}
    for tag in raw:
        kind = category(tag)
        centre = float(train[tag].median())
        scale = max(float((train[tag] - centre).abs().median() * 1.4826), floors[kind])
        amplitude = None
        if kind in config['amplitude_caps']:
            amplitude = float(np.clip(scale * config['amplitude_scale_multiplier'], *config['amplitude_caps'][kind]))
            result[tag] = amplitude
        rows.append({'tag': tag, 'category': kind, 'training_median': centre, 'robust_scale': scale,
                     'amplitude': amplitude, 'fit_observations': len(train)})
    return result, pd.DataFrame(rows)


def sample_parameters(config, block_number):
    """One common nuisance draw per background, shared across every class."""
    rng = np.random.default_rng(np.random.SeedSequence([config['generation_seed'], int(block_number)]))
    low, high = config['onset_range_inclusive']
    return {'onset': int(rng.integers(low, high + 1)), 'duration_hours': int(rng.choice(config['duration_hours'])),
            'exponent': float(rng.choice(config['damage_exponents']))}


def construct_scene(base, class_id, parameters, variant, amplitudes, config):
    """Return a copy and an exact change mask; no clipping or resampling."""
    if class_id not in range(len(config['groups'])): raise ValueError('Unknown simulation class')
    out = base.copy(deep=True)
    mask = np.zeros(out.shape, dtype=bool)
    elapsed = (np.arange(len(base)) - parameters['onset']) * config['sample_seconds'] / 3600.
    damage = np.clip(elapsed / parameters['duration_hours'], 0., 1.) ** parameters['exponent']
    lag = int(variant['lag_points'])
    response = np.r_[np.zeros(lag), damage][:len(base)] ** variant['response_power']
    for tag, sign in config['groups'][class_id].items():
        out[tag] += sign * amplitudes[tag] * variant['amplitude_multiplier'] * response
        mask[:, out.columns.get_loc(tag)] = response > 0
    for tag in out:
        if not out[tag].between(*config['bounds'][category(tag)]).all():
            raise ValueError('A planned scene violates declared simulation bounds: ' + tag)
    if not np.array_equal(out.to_numpy()[~mask], base.to_numpy()[~mask]):
        raise RuntimeError('A non-injected value changed')
    return out, mask, response


def input_contract(columns, config):
    units = {'axial': 'mm', 'temperature': 'degC', 'vibration': 'um', 'pressure': 'kg/cm2 (as reported)',
             'differential_pressure': 'kg/cm2 (as reported)', 'speed': 'rpm'}
    aggregation = {'axial': 'mean', 'speed': 'mean', 'temperature': 'maximum',
                   'vibration': 'maximum', 'pressure': 'minimum', 'differential_pressure': 'minimum'}
    return {'signal_type': 'multivariate_process', 'interval_seconds': float(config['sample_seconds']),
            'window_points': config['window_points'],
            'channels': [{'name': c, 'quantity': category(c), 'unit': units[category(c)],
                          'aggregation': aggregation[category(c)]} for c in columns]}


def prepare(source, config, destination):
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('Preserve existing prepared evidence; use another directory')
    if sha(source) != config['source_sha256']: raise ValueError('Original RCSD version differs')
    frame = pd.read_excel(source, sheet_name='Sheet1')
    times = pd.to_datetime(frame.pop('Timestamp'))
    raw = frame.astype(float)
    if len(raw) != 35036 or len(raw.columns) != 25: raise ValueError('Unexpected original RCSD dimensions')
    blocks, metadata, tails = background_plan(raw, times, config)
    if blocks['split'].value_counts().to_dict() != config['expected_background_blocks']:
        raise ValueError('Background budgets differ from the pinned plan')
    amplitudes, scales = training_amplitudes(raw, metadata, config)
    arrays, manifests, audit = [], [], []
    begin, points = config['window_start_point'], config['window_points']
    for number, block in blocks.iterrows():
        ids = np.arange(block.first_source_row, block.last_source_row + 1)
        base = raw.iloc[ids].reset_index(drop=True)
        parameters = sample_parameters(config, number)
        variants = config['variants'] if block['split'] == 'query' else {'main': config['variants']['main']}
        for variant_name, variant in variants.items():
            for class_id, class_name in enumerate(config['class_names']):
                scene, mask, response = construct_scene(base, class_id, parameters, variant, amplitudes, config)
                take = slice(begin, begin + points)
                if class_id and not np.all(response[take] > 0):
                    raise ValueError('Diagnostic window contains unmodified prefix/transition')
                values = scene.iloc[take].to_numpy(dtype=np.float32).T
                if values.shape != (25, points) or not np.isfinite(values).all(): raise ValueError('Invalid planned diagnostic window')
                arrays.append(values)
                sample_id = len(manifests)
                row = {'sample_id': sample_id, 'window_id': f'{block.block_id}_{variant_name}_c{class_id}',
                       'block_id': block.block_id, 'split': block['split'], 'variant': variant_name,
                       'class_id': class_id, 'class_name': class_name,
                       'first_source_row': int(ids[begin]), 'last_source_row': int(ids[begin + points - 1]),
                       'start_time': str(times.iloc[ids[begin]]), 'end_time': str(times.iloc[ids[begin + points - 1]]),
                       'onset': parameters['onset'], 'duration_hours': parameters['duration_hours'], 'exponent': parameters['exponent'],
                       'injected_scalar_values': int(mask[take].sum()), 'minimum_response': float(response[take].min()) if class_id else 0.,
                       'input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                       'label_provenance': config['label_provenance'] if class_id else 'unmodified_candidate_not_verified_healthy'}
                manifests.append(row)
                audit.append({'sample_id': sample_id, 'non_injected_cells_unchanged': True,
                              'all_anomaly_window_points_after_onset_and_lag': bool(not class_id or (response[take] > 0).all()),
                              'planned_bounds_pass': True, 'source_rows_contiguous': bool(np.all(np.diff(ids) == 1))})
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'windows.npz', x=np.stack(arrays))
    pd.DataFrame(manifests).to_csv(destination / 'sample_manifest.csv', index=False)
    blocks.to_csv(destination / 'background_blocks.csv', index=False)
    scales.to_csv(destination / 'training_amplitudes.csv', index=False)
    pd.DataFrame(audit).to_csv(destination / 'construction_checks.csv', index=False)
    (destination / 'unused_tails.json').write_text(json.dumps(tails, indent=2) + '\n', encoding='utf-8')
    protocol = {'config': config, 'input_contract': input_contract(list(raw), config),
                'raw_rows': len(raw), 'channels': len(raw.columns), 'background_blocks': len(blocks),
                'background_budget': blocks['split'].value_counts().to_dict(),
                'prepared_windows': len(arrays), 'natural_fault_incidents': 0, 'source_kind': 'semi_synthetic',
                'source_raw_sha256': sha(source), 'unchanged_source_after_read': sha(source) == config['source_sha256'],
                'rejected_planned_scenes': 0, 'preonset_windows_mislabeled_as_anomalies': 0,
                'sampling_interval_seconds': config['sample_seconds'], 'window_first_last_hours': (points-1)*config['sample_seconds']/3600.,
                'challenge_reuses_query_backgrounds': True, 'background_control_is_confirmed_healthy': False}
    (destination / 'preparation.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    checksums = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
    (destination / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n', encoding='utf-8')
    return protocol


def support_tasks(manifest, config):
    source = manifest.index[manifest['split'].eq('source') & manifest['variant'].eq('main')].to_numpy()
    pool = manifest[manifest['split'].eq('support_pool')]
    query = manifest[manifest['split'].eq('query')]
    if set(manifest.loc[source, 'block_id']) & (set(pool.block_id) | set(query.block_id)) or set(pool.block_id) & set(query.block_id):
        raise ValueError('An original background crosses training/support/query')
    tasks = []
    for seed in config['support_seeds']:
        chosen = np.random.default_rng(seed).permutation(sorted(pool.block_id.unique()))
        for k in config['k_values']:
            if len(chosen) < k: raise ValueError('Not enough original support backgrounds')
            support = pool.index[pool.block_id.isin(chosen[:k])].to_numpy()
            if len(support) != 4*k or not np.array_equal(np.bincount(pool.loc[support, 'class_id'], minlength=4), np.full(4, k)):
                raise ValueError('Each class must have one window per selected support background')
            tasks.append({'k': k, 'support_seed': seed, 'support_ids': support.tolist(),
                          'support_backgrounds': chosen[:k].tolist(),
                          'query_ids': {variant: query.index[query.variant.eq(variant)].tolist() for variant in config['variants']}})
    return source, tasks
