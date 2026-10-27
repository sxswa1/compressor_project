"""Build RCSD semi-synthetic v1. No training, no field data, no raw-file writes."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import gzip
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, frame):
    if str(path).endswith('.gz'):
        # No file name or wall-clock timestamp in gzip headers: reproducible bytes.
        with path.open('wb') as f:
            with gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as z:
                z.write(frame.to_csv(index=False, lineterminator='\n').encode('utf-8'))
    else:
        frame.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n')


def kind(c):
    if 'ZI' in c: return 'axial'
    if 'TI' in c: return 'temperature'
    if 'XI' in c: return 'vibration'
    if 'PDI' in c: return 'differential_pressure'
    if 'SI' in c: return 'speed'
    return 'pressure'


def limits(c):
    k = kind(c)
    if k == 'axial': return -.5, .5
    if k == 'temperature': return None, 120
    if k == 'vibration': return None, 89
    return {'75PI808.pv': (9, None), '75PI823.pv': (1.8, None),
            '75PI870.pv': (4.2, None), '75PI845.pv': (.84, None),
            '75PDI853.pv': (None, 3), '75SI865R.pv': (6050, 8500)}[c]


def risk(di):
    return np.select([di >= .85, di >= .65, di >= .4], [1, 2, 3], default=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'data')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    source = HERE.parent / cfg['source']
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert raw_hash == cfg['source_sha256'], 'Unexpected source version'
    raw = pd.read_excel(source, sheet_name='Sheet1')
    t = pd.to_datetime(raw.pop('Timestamp'), errors='raise')
    x = raw.astype(float)
    cols = list(x)
    assert len(cols) == 25 and np.isfinite(x.to_numpy()).all()
    assert t.is_monotonic_increasing and not t.duplicated().any()
    step = pd.Timedelta(minutes=cfg['sample_minutes'])
    dt = cfg['sample_minutes'] / 60
    split = pd.Series(np.select([t < cfg['split_boundaries'][0], t < cfg['split_boundaries'][1]],
                               ['train', 'validation'], default='test'))
    gap = t.diff().ne(step)
    within = pd.Series(True, index=x.index)
    for c in cols:
        lo, hi = limits(c)
        if lo is not None: within &= x[c].ge(lo)
        if hi is not None: within &= x[c].le(hi)
    # A causal 24-hour warm-up after any threshold failure, gap, or split boundary.
    reset = gap | split.ne(split.shift()) | ~within
    run = reset.cumsum()
    streak = within.astype(int).groupby(run).cumsum()
    sustained = within & streak.ge(cfg['stable_history_points'])
    training = x.loc[sustained & split.eq('train')]
    assert len(training) > 200
    units = {'axial': 'mm', 'temperature': 'degC', 'vibration': 'um',
             'pressure': 'kg/cm2 (as reported)', 'differential_pressure': 'kg/cm2 (as reported)', 'speed': 'rpm'}
    floors = {'axial': .001, 'temperature': .1, 'vibration': .1,
              'pressure': .01, 'differential_pressure': .01, 'speed': 10}
    scale_rows = []
    bands = pd.cut(x['75SI865R.pv'], cfg['speed_band_edges'], labels=False, right=False)
    def estimate(frame):
        med = frame.median()
        mad = frame.sub(med).abs().median() * 1.4826
        iqr = (frame.quantile(.75) - frame.quantile(.25)) / 1.349
        return med, pd.Series({c: max(mad[c], iqr[c], floors[kind(c)]) for c in cols})
    pooled_med, pooled_scale = estimate(training)
    scales, medians = {}, {}
    for b in range(len(cfg['speed_band_edges']) - 1):
        part = x.loc[sustained & split.eq('train') & bands.eq(b)]
        fallback = len(part) < cfg['minimum_band_training_points']
        med, sc = (pooled_med, pooled_scale) if fallback else estimate(part)
        scales[b], medians[b] = sc, med
        for c in cols:
            scale_rows.append(dict(speed_band=b, tag=c, median=med[c], scale=sc[c],
                                   training_points=len(part), pooled_fallback=fallback, floor=floors[kind(c)]))
    # Training-only empirical envelope is an audit flag, NOT a hard health filter:
    # a temporal distribution shift is not evidence of mechanical damage.
    envelope = pd.Series(False, index=x.index)
    for b in scales:
        mask = bands.eq(b)
        checks = x.loc[mask, cols[:-1]].sub(medians[b][cols[:-1]]).abs().le(
            cfg['background_robust_envelope'] * scales[b][cols[:-1]])
        envelope.loc[mask] = checks.all(axis=1)
    candidate = sustained
    segments = (gap | split.ne(split.shift()) | candidate.ne(candidate.shift())).cumsum()
    metadata = pd.DataFrame({'source_row_id': x.index, 'source_timestamp': t, 'split': split,
                             'within_paper_limits': within, 'sustained_24h': sustained,
                             'within_training_envelope': envelope, 'baseline_candidate': candidate,
                             'segment_id': segments, 'speed_below100': x['75SI865R.pv'].lt(100)})
    save_csv(out / 'raw_observations.csv.gz', pd.concat([pd.DataFrame({'source_row_id': x.index}), x], axis=1))
    save_csv(out / 'raw_metadata.csv', metadata)
    save_csv(out / 'training_scales.csv', pd.DataFrame(scale_rows))
    dictionary = []
    for c in cols:
        lo, hi = limits(c)
        dictionary.append(dict(tag=c, category=kind(c), unit=units[kind(c)],
                               aggregation={'axial': 'Avg', 'speed': 'Avg', 'temperature': 'Max',
                                            'vibration': 'Max', 'pressure': 'Min', 'differential_pressure': 'Min'}[kind(c)],
                               paper_lower=lo, paper_upper=hi, model_input=True))
    save_csv(out / 'sensor_dictionary.csv', pd.DataFrame(dictionary))
    save_json(out / 'model_input_columns.json', cols)
    blocks, excluded = [], []
    size = cfg['source_block_points']
    for seg, group in metadata.loc[candidate].groupby('segment_id'):
        ids = group.source_row_id.to_numpy()
        for start in range(0, len(ids), size):
            idx = ids[start:start + size]
            if len(idx) < size:
                excluded.append(dict(segment_id=int(seg), reason='candidate_tail_shorter_than_120h', points=len(idx),
                                     first_source_row=int(idx[0]), last_source_row=int(idx[-1])))
                continue
            blocks.append(dict(block_id=f'B{len(blocks):04d}', segment_id=int(seg), split=split.iloc[idx[0]],
                               first_source_row=int(idx[0]), last_source_row=int(idx[-1]), points=len(idx)))
    block_df = pd.DataFrame(blocks)
    assert set(block_df.split) == {'train', 'validation', 'test'}, 'Insufficient source blocks'
    save_csv(out / 'source_blocks.csv', block_df)
    save_csv(out / 'unused_candidate_tails.csv', pd.DataFrame(excluded))
    sensor_groups = {
        'thermal': {'75TI829.pv': 1, '75TI830.pv': 1, '75TI834.pv': 1, '75TI836.pv': 1},
        'oil_thermal': {'75PI808.pv': -1, '75PI823.pv': -1, '75TI834.pv': 1, '75TI836.pv': 1},
        'vibration_axial': {'75XI823BX.pv': 1, '75XI824BX.pv': 1, '75ZI801BA.pv': -1, '75ZI801BB.pv': -1}}
    manifests, rejected, checks, events, summaries, examples = [], [], [], [], [], {}
    for split_name, seed in cfg['seeds'].items():
        rng = np.random.default_rng(seed)
        observations, labels, masks, lineage = [], [], [], []
        for block in blocks:
            if block['split'] != split_name: continue
            src_ids = np.arange(block['first_source_row'], block['last_source_row'] + 1)
            base_full = x.iloc[src_ids].reset_index(drop=True)
            band = int(np.searchsorted(cfg['speed_band_edges'], base_full['75SI865R.pv'].median(), side='right') - 1)
            sc = scales[band]
            specs = [('replay', 'none')] + [(k, 'none') for k in ['spike', 'offset', 'drift_recovery', 'stuck', 'missing', 'coupled_recovery']]
            specs += [(g, f) for g in sensor_groups for f in ['linear', 'accelerating']]
            if split_name == 'test': specs += [(g, 'piecewise') for g in sensor_groups]
            for scenario, family in specs:
                scene_id = f'S{len(manifests) + len(rejected):05d}'
                n = int(rng.integers(192, size + 1))
                onset = int(rng.integers(*[cfg['onset_points'][0], cfg['onset_points'][1] + 1]))
                params = {'speed_band': band}
                d = None
                event_observed = False
                is_deg = scenario in sensor_groups
                if is_deg:
                    nominal = float(rng.uniform(*cfg['degradation_nominal_hours']))
                    exponent = float(rng.uniform(1.3, 2.3)) if family == 'accelerating' else 1.0
                    rate_switch = float(rng.uniform(.2, .6))
                    rate_after = float(rng.uniform(.5, 1.8))
                    d_full = np.zeros(size)
                    for i in range(onset + 1, size):
                        # Integrate monotone damage; future changes are not fed to predictors.
                        elapsed_h = (i - onset) * dt
                        prev_h = (i - onset - 1) * dt
                        if family == 'accelerating':
                            increment = (elapsed_h / nominal)**exponent - (prev_h / nominal)**exponent
                        elif family == 'piecewise':
                            increment = dt / nominal * (rate_after if d_full[i-1] >= rate_switch else 1.0)
                        else: increment = dt / nominal
                        d_full[i] = min(1., d_full[i-1] + increment)
                    hits = np.flatnonzero(d_full >= 1)
                    failure_idx = int(hits[0]) if len(hits) else None
                    censor = rng.random() < cfg['censor_probability'] or failure_idx is None
                    if censor:
                        upper = failure_idx if failure_idx is not None else size
                        n = onset + max(2, int((upper - onset) * rng.uniform(.45, .85)))
                    else:
                        n = failure_idx + 1
                        event_observed = True
                    d = d_full[:n]
                    params.update(nominal_hours=nominal, exponent=exponent, rate_switch=rate_switch,
                                  rate_after=rate_after, censored=censor)
                base = base_full.iloc[:n].copy()
                values = base.copy()
                mask = np.zeros((n, len(cols)), dtype=np.uint8)
                observation_anomaly = np.zeros(n, dtype=np.uint8)
                quality = np.zeros(n, dtype=np.uint8)
                affected = {}
                if is_deg:
                    for c, direction in sensor_groups[scenario].items():
                        cap = cfg['amplitude_caps'][kind(c)]
                        amp = direction * float(np.clip(sc[c] * rng.uniform(*cfg['degradation_scale_multipliers']), *cap))
                        lag = int(rng.integers(0, 9))
                        power = float(rng.uniform(.8, 1.4))
                        delayed = np.concatenate([np.zeros(lag), d])[:n]
                        response = delayed**power
                        values[c] += amp * response
                        mask[:, cols.index(c)] = (response > 0).astype(np.uint8)
                        affected[c] = dict(amplitude=amp, lag_points=lag, response_power=power)
                    event_start = onset + 1
                    event_end = n - 1
                elif scenario != 'replay':
                    strength = float(rng.choice(cfg['anomaly_scale_multipliers']))
                    duration = int(rng.integers(cfg['anomaly_duration_points'][0], cfg['anomaly_duration_points'][1] + 1))
                    if scenario == 'spike': duration = int(rng.integers(1, 3))
                    end = onset + duration
                    selected = sensor_groups['oil_thermal'] if scenario == 'coupled_recovery' else {str(rng.choice(['75TI834.pv', '75XI823BX.pv', '75PI823.pv', '75ZI801BA.pv'])): float(rng.choice([-1, 1]))}
                    for c, direction in selected.items():
                        amp = float(direction * strength * sc[c])
                        shape = np.ones(duration)
                        if scenario in ('drift_recovery', 'coupled_recovery'):
                            shape = np.sin(np.pi * (np.arange(duration) + 1) / (duration + 1))
                        if scenario == 'missing': values.loc[onset:end-1, c] = np.nan
                        elif scenario == 'stuck': values.loc[onset:end-1, c] = values.loc[onset-1, c]
                        else: values.loc[onset:end-1, c] += amp * shape
                        mask[onset:end, cols.index(c)] = 1
                        affected[c] = dict(amplitude=amp if scenario not in ('missing', 'stuck') else None)
                    observation_anomaly[onset:end] = 1
                    if scenario in ('missing', 'stuck'): quality[onset:end] = 1
                    params.update(strength=strength, duration_points=duration)
                    event_start, event_end = onset, end - 1
                params['channels'] = affected
                # Reject instead of clipping: clipping would alter the stated injection rule.
                invalid = []
                for c in cols:
                    lo, hi = cfg['bounds'][kind(c)]
                    if ((values[c] < lo) | (values[c] > hi) | np.isinf(values[c])).any(): invalid.append(c)
                if invalid:
                    rejected.append(dict(scene_id=scene_id, block_id=block['block_id'], split=split_name,
                                         scenario=scenario, reason='experimental_bounds', channels=invalid, parameters=params))
                    continue
                keys = pd.DataFrame({'scene_id': scene_id, 'step': np.arange(n)})
                obs = pd.concat([keys, values], axis=1)
                lab = keys.copy()
                lab['observation_anomaly'] = observation_anomaly
                lab['quality_issue'] = quality
                lab['degradation_active'] = (d > 0).astype(int) if is_deg else 0
                lab['d_ref'] = d if is_deg else np.nan
                lab['DI_ref'] = 1 - d if is_deg else np.nan
                lab['risk_ref'] = pd.array(risk(1-d) if is_deg else [pd.NA]*n, dtype='Int64')
                lab['RUL_ref_hours'] = (n - 1 - np.arange(n)) * dt if event_observed else np.nan
                lab['di_label_valid'] = is_deg
                # Before onset, the randomly scheduled future cannot be inferred
                # from healthy observations. Keep truth but exclude from RUL loss.
                lab['rul_label_valid'] = (d > 0) & event_observed if is_deg else False
                lab['label_provenance'] = 'synthetic_assumption' if is_deg else 'injected_interval' if scenario != 'replay' else 'candidate_only'
                ids = src_ids[:n]
                lin = keys.assign(source_row_id=ids, source_timestamp=t.iloc[ids].to_numpy(),
                                  block_id=block['block_id'], split=split_name)
                change = pd.concat([keys, pd.DataFrame(mask, columns=cols)], axis=1)
                # QA compares observations against original background, independent of serialization.
                a, b = values.to_numpy(), base.to_numpy()
                assert np.array_equal(a[mask == 0], b[mask == 0])
                assert np.all((~np.isfinite(a)) <= (mask.astype(bool)))
                assert scenario == 'missing' or np.isfinite(a).all()
                assert t.iloc[ids].diff().iloc[1:].eq(step).all()
                assert split.iloc[ids].eq(split_name).all()
                if is_deg:
                    assert np.all(np.diff(d) >= 0) and np.all((d >= 0) & (d <= 1))
                    assert np.allclose((lab.DI_ref + lab.d_ref).to_numpy(), 1)
                    assert bool(d[-1] == 1) == event_observed
                    if event_observed: assert np.allclose(np.diff(lab.RUL_ref_hours), -dt)
                    else: assert lab.RUL_ref_hours.isna().all()
                else: assert lab[['d_ref', 'DI_ref', 'RUL_ref_hours', 'risk_ref']].isna().all().all()
                checks.append(dict(scene_id=scene_id, passed=True, rows=n,
                                   intentional_missing_cells=int(values.isna().sum().sum()),
                                   changed_cells=int((~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).sum()),
                                   masked_cells=int(mask.sum())))
                manifest = dict(scene_id=scene_id, split=split_name, block_id=block['block_id'],
                                source_segment_id=block['segment_id'], scenario=scenario, family=family,
                                evaluation_group='unseen_shape' if family == 'piecewise' else 'seen_family',
                                rows=n, first_source_row=int(ids[0]), last_source_row=int(ids[-1]),
                                source_start=str(t.iloc[ids[0]]), source_end=str(t.iloc[ids[-1]]),
                                onset_step=onset if scenario != 'replay' else None,
                                event_observed=event_observed, right_censored=bool(is_deg and not event_observed),
                                failure_step=n-1 if event_observed else None,
                                failure_source_timestamp=str(t.iloc[ids[-1]]) if event_observed else None,
                                seed=seed, parameters=params)
                manifests.append(manifest)
                if scenario != 'replay':
                    events.append(dict(scene_id=scene_id, split=split_name, event_type='degradation' if is_deg else scenario,
                                       start_step=event_start, end_step=event_end,
                                       failure_step=n-1 if event_observed else None,
                                       event_observed=event_observed, quality_only=scenario in ('missing', 'stuck')))
                observations.append(obs); labels.append(lab); masks.append(change); lineage.append(lin)
                if split_name == 'train' and scenario not in examples: examples[scenario] = (base, values, lab, manifest)
        obs, lab = pd.concat(observations, ignore_index=True), pd.concat(labels, ignore_index=True)
        save_csv(out / f'{split_name}_observations.csv.gz', obs)
        save_csv(out / f'{split_name}_labels.csv.gz', lab)
        save_csv(out / f'{split_name}_lineage.csv.gz', pd.concat(lineage, ignore_index=True))
        save_csv(out / f'{split_name}_injection_mask.csv.gz', pd.concat(masks, ignore_index=True))
        summaries.append(dict(split=split_name, scenes=obs.scene_id.nunique(), rows=len(obs),
                              observation_anomaly_points=int(lab.observation_anomaly.sum()),
                              degradation_points=int(lab.degradation_active.sum()),
                              intentional_missing_cells=int(obs[cols].isna().sum().sum()),
                              risk_counts={str(k): int(v) for k, v in lab.risk_ref.value_counts().items()}))
        print(split_name, len(obs), 'rows;', obs.scene_id.nunique(), 'scenes', flush=True)
    save_json(out / 'scene_manifest.json', manifests)
    save_json(out / 'rejected_scenes.json', rejected)
    save_csv(out / 'events.csv', pd.DataFrame(events))
    save_csv(out / 'scene_checks.csv', pd.DataFrame(checks))
    # Verify all source blocks, not only randomly selected scenes, are disjoint.
    all_ids = np.concatenate([np.arange(b['first_source_row'], b['last_source_row'] + 1) for b in blocks])
    assert len(np.unique(all_ids)) == len(all_ids)
    assert np.array_equal(risk(np.array([1, .85, .65, .4, .39999, 0])), [1, 1, 2, 3, 4, 4])
    counts = pd.DataFrame([{k: m[k] for k in ['split', 'scenario', 'family', 'event_observed', 'right_censored', 'rows']} for m in manifests])
    save_csv(out / 'scenario_counts.csv', counts.groupby(['split', 'scenario', 'family']).agg(
        scenes=('rows', 'size'), rows=('rows', 'sum'), observed_endpoints=('event_observed', 'sum'), censored=('right_censored', 'sum')).reset_index())
    report = dict(version=cfg['version'], source_sha256=raw_hash, raw_rows=len(x), candidate_rows=int(candidate.sum()),
                  paper_limit_pass_rows=int(within.sum()), sustained_rows=int(sustained.sum()),
                  splits=summaries, source_blocks=len(blocks), source_block_points=len(all_ids),
                  source_blocks_by_split=block_df.split.value_counts().to_dict(),
                  raw_counts_by_split=split.value_counts().to_dict(),
                  candidate_counts_by_split=metadata.groupby('split').baseline_candidate.sum().astype(int).to_dict(),
                  generated_scenes=len(manifests), rejected_scenes=len(rejected),
                  observed_endpoints=int(counts.event_observed.sum()), right_censored=int(counts.right_censored.sum()),
                  checks_passed=len(checks), cross_split_source_overlap=0,
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  test_used_for_parameter_selection=False, formal_models_trained=False)
    save_json(out / 'quality_report.json', report)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for row, scenario in enumerate(sensor_groups):
        base, values, lab, manifest = examples[scenario]
        time = np.arange(len(base)) * dt
        for c in manifest['parameters']['channels']:
            # Normalize ONLY the visualization by training scale. Saved data keep units.
            axes[row, 0].plot(time, (values[c] - base[c]) / pooled_scale[c], label=c)
        axes[row, 0].set_title(scenario + ': injected change / training scale')
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].plot(time, lab.DI_ref, label='DI reference')
        for threshold in [.85, .65, .4]: axes[row, 1].axhline(threshold, color='gray', ls=':', lw=.8)
        twin = axes[row, 1].twinx()
        twin.plot(time, lab.RUL_ref_hours, color='orange', label='RUL (h)')
        twin.set_ylabel('RUL (h)')
        axes[row, 1].set_title('Synthetic labels; not measured lifetimes')
    for ax in axes.flat: ax.set_xlabel('Elapsed hours (plot only)'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'degradation_examples.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for ax, scenario in zip(axes.flat, ['spike', 'offset', 'drift_recovery', 'stuck', 'missing', 'coupled_recovery']):
        base, values, lab, manifest = examples[scenario]
        c = next(iter(manifest['parameters']['channels']))
        onset = manifest['onset_step']; left, right = max(0, onset-16), min(len(base), onset+40)
        time = np.arange(left, right) * dt
        ax.plot(time, base[c].iloc[left:right], label='Original', lw=1)
        ax.plot(time, values[c].iloc[left:right], label='Constructed', lw=1)
        ax.axvspan(onset*dt, (onset+manifest['parameters']['duration_points'])*dt, color='orange', alpha=.15)
        ax.set_title(scenario + ' | ' + c); ax.legend(fontsize=8); ax.grid(alpha=.2)
        ax.set_xlabel('Elapsed hours (plot only)'); ax.set_ylabel(units[kind(c)])
    fig.tight_layout(); fig.savefig(out / 'anomaly_examples.png', dpi=160); plt.close(fig)
    shutil.copyfile(HERE / 'config.json', out / 'config_snapshot.json')
    save_json(out / 'build_provenance.json', dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              config_sha256=hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest(), source_sha256=raw_hash))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name != 'checksums.json'}
    save_json(out / 'checksums.json', hashes)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == raw_hash
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
