"""RCSD simulation admission, label boundaries and background isolation."""
from importlib import import_module
from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import pytest

study = import_module('02_few_shot_diagnosis.pipelines.rcsd_semisynthetic')
ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / '02_few_shot_diagnosis/configs/rcsd_semisynthetic_v1.json').read_text())


def base(points=480):
    tags = sorted({c for group in config()['groups'] for c in group})
    values = {'temperature': 70., 'pressure': 10., 'vibration': 20., 'axial': 0.}
    return pd.DataFrame({c: np.full(points, values[study.category(c)]) for c in tags})


@pytest.mark.parametrize('class_id', range(4))
def test_only_declared_channels_change_and_prefix_is_not_anomaly(class_id):
    cfg = config();frame = base();before = frame.copy(deep=True)
    params = {'onset': 120, 'duration_hours': 48, 'exponent': 1.5}
    amp = {c: {'temperature': 5., 'pressure': .2, 'vibration': 5., 'axial': .05}[study.category(c)] for c in frame}
    result, mask, response = study.construct_scene(frame, class_id, params, cfg['variants']['main'], amp, cfg)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result.iloc[:121], before.iloc[:121])
    changed = set(result.columns[mask.any(axis=0)])
    assert changed == set(cfg['groups'][class_id])
    assert np.array_equal(result.to_numpy()[~mask], before.to_numpy()[~mask])
    if class_id: assert (response[240:260] > 0).all()


def test_challenge_preserves_background_and_changes_only_response():
    cfg = config();frame = base();params = study.sample_parameters(cfg, 7)
    amp = {c: .1 for c in frame}
    a, _, _ = study.construct_scene(frame, 0, params, cfg['variants']['main'], amp, cfg)
    b, _, _ = study.construct_scene(frame, 0, params, cfg['variants']['weak_lag_challenge'], amp, cfg)
    pd.testing.assert_frame_equal(a, b)
    main, _, _ = study.construct_scene(frame, 1, params, cfg['variants']['main'], amp, cfg)
    weak, _, _ = study.construct_scene(frame, 1, params, cfg['variants']['weak_lag_challenge'], amp, cfg)
    assert np.all(weak.to_numpy() <= main.to_numpy())


def test_amplitude_fits_only_source_candidates():
    cfg = config()
    frame = pd.DataFrame({'75TI834.pv': [60., 61., 62., 70., 80., 90.]})
    meta = pd.DataFrame({'split': ['source'] * 3 + ['query'] * 3, 'baseline_candidate': True})
    expected, table = study.training_amplitudes(frame, meta, cfg)
    modified = frame.copy();modified.iloc[3:] = 99999.
    actual, _ = study.training_amplitudes(modified, meta, cfg)
    assert actual == expected and table.fit_observations.tolist() == [3]


def test_generation_parameters_are_deterministic_background_nuisances():
    cfg = config()
    assert study.sample_parameters(cfg, 4) == study.sample_parameters(copy.deepcopy(cfg), 4)
    draws = [study.sample_parameters(cfg, b) for b in range(20)]
    assert len({json.dumps(v, sort_keys=True) for v in draws}) > 10
    assert all(96 <= v['onset'] <= 144 for v in draws)


def test_paper_filter_warmup_and_real_gap_split_backgrounds():
    cfg = config();frame = base(700)
    clock = pd.date_range('2022-01-01', periods=700, freq='15min')
    blocks, _, tails = study.background_plan(frame, clock, cfg)
    assert len(blocks) == 1 and blocks.first_source_row.iloc[0] == 95
    assert blocks.last_source_row.iloc[0] == 574 and len(tails) == 1
    broken = pd.Series(clock);broken.iloc[350:] += pd.Timedelta(hours=1)
    other, _, _ = study.background_plan(frame, broken, cfg)
    assert other.empty


def manifest():
    rows = []
    for role, count in [('source', 6), ('support_pool', 11), ('query', 2)]:
        for b in range(count):
            for variant in (['main', 'weak_lag_challenge'] if role == 'query' else ['main']):
                for c in range(4):
                    rows.append({'split': role, 'block_id': f'{role}_{b:02d}', 'variant': variant, 'class_id': c})
    return pd.DataFrame(rows)


def test_support_is_nested_and_budget_counts_original_backgrounds():
    meta = manifest();source, tasks = study.support_tasks(meta, config())
    assert len(source) == 24 and len(tasks) == 9
    for seed in config()['support_seeds']:
        same = sorted([t for t in tasks if t['support_seed'] == seed], key=lambda t: t['k'])
        assert set(same[0]['support_backgrounds']) <= set(same[1]['support_backgrounds']) <= set(same[2]['support_backgrounds'])
    for task in tasks:
        assert len(task['support_ids']) == 4*task['k']
        assert meta.loc[task['support_ids'], 'block_id'].nunique() == task['k']
        assert not set(meta.loc[source, 'block_id']) & set(task['support_backgrounds'])


def test_global_cross_class_background_leakage_is_rejected():
    meta = manifest()
    meta.loc[meta['split'].eq('query') & meta.class_id.eq(2), 'block_id'] = 'support_pool_00'
    with pytest.raises(ValueError, match='crosses'): study.support_tasks(meta, config())


def test_insufficient_original_support_budget_is_not_faked_by_more_windows():
    meta = manifest();meta = meta[~(meta['split'].eq('support_pool') & meta.block_id.eq('support_pool_10'))].reset_index(drop=True)
    cfg = config();cfg['k_values'] = [11]
    with pytest.raises(ValueError, match='Not enough'): study.support_tasks(meta, cfg)


def test_bounds_violation_fails_instead_of_clipping_or_resampling():
    cfg = config();frame = base();params = {'onset': 1, 'duration_hours': 1, 'exponent': 1.}
    amp = {c: 10000. for c in frame}
    with pytest.raises(ValueError, match='bounds'):
        study.construct_scene(frame, 1, params, cfg['variants']['main'], amp, cfg)


def test_source_time_aggregation_contract_is_not_field_six_minutes():
    cfg = config();contract = study.input_contract(list(base()), cfg)
    assert contract['interval_seconds'] == 900 and contract['window_points'] == 20
    channel = {c['name']: c for c in contract['channels']}
    assert channel['75TI834.pv']['aggregation'] == 'maximum'
    assert channel['75XI823BX.pv']['unit'] == 'um'
    assert channel['75PI823.pv']['aggregation'] == 'minimum'
