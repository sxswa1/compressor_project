"""Independent saved-file audit. Run after build_dataset.py."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from load_dataset import load_scenes, load_baseline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
manifest = json.loads((DATA / 'scene_manifest.json').read_text(encoding='utf-8'))
raw = pd.read_csv(DATA / 'raw_observations.csv.gz').set_index('source_row_id')
meta = pd.read_csv(DATA / 'raw_metadata.csv').set_index('source_row_id')
hashes = json.loads((DATA / 'checksums.json').read_text(encoding='utf-8'))
for name, digest in hashes.items():
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
source_sets, results = {}, []
for split in ['train', 'validation', 'test']:
    x, y, lineage = load_scenes(split)
    mask = pd.read_csv(DATA / f'{split}_injection_mask.csv.gz').set_index(['scene_id', 'step'])
    assert x.index.is_unique and x.index.equals(mask.index)
    assert len(x.columns) == 25 and all(c.endswith('.pv') for c in x.columns)
    source_sets[split] = set(lineage.source_row_id)
    assert meta.loc[list(source_sets[split]), 'split'].eq(split).all()
    assert meta.loc[list(source_sets[split]), 'baseline_candidate'].all()
    reference = raw.loc[lineage.source_row_id].to_numpy()
    current = x.to_numpy()
    unmasked = mask.to_numpy() == 0
    np.testing.assert_allclose(current[unmasked], reference[unmasked], rtol=0, atol=0)
    assert not np.isinf(current).any()
    for m in [m for m in manifest if m['split'] == split]:
        sid = m['scene_id']; obs, lab, lin = x.loc[sid], y.loc[sid], lineage.loc[sid]
        assert len(obs) == m['rows'] and obs.index.tolist() == list(range(m['rows']))
        assert pd.to_datetime(lin.source_timestamp).diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
        if m['scenario'] == 'missing':
            expected = len(m['parameters']['channels']) * m['parameters']['duration_points']
            assert obs.isna().sum().sum() == expected
        else: assert obs.notna().all().all()
        if m['family'] != 'none':
            np.testing.assert_array_equal(lab.rul_label_valid, (lab.d_ref > 0) & m['event_observed'])
            np.testing.assert_allclose(lab.DI_ref, 1 - lab.d_ref, atol=1e-14)
            expected_risk = np.where(lab.DI_ref >= .85, 1, np.where(lab.DI_ref >= .65, 2, np.where(lab.DI_ref >= .4, 3, 4)))
            np.testing.assert_array_equal(lab.risk_ref, expected_risk)
            assert (lab.d_ref.diff().dropna() >= -1e-14).all()
            if m['event_observed']:
                expected_rul = (pd.Timestamp(m['failure_source_timestamp']) - pd.to_datetime(lin.source_timestamp)).dt.total_seconds()/3600
                np.testing.assert_allclose(lab.RUL_ref_hours, expected_rul)
                assert lab.RUL_ref_hours.iloc[-1] == 0 and lab.DI_ref.iloc[-1] == 0
            else:
                assert lab.RUL_ref_hours.isna().all() and lab.DI_ref.iloc[-1] > 0
            # Reconstruct saved channel responses directly from labels/parameters.
            for c, p in m['parameters']['channels'].items():
                delayed = lab.d_ref.shift(p['lag_points'], fill_value=0).to_numpy()
                expected = raw.loc[lin.source_row_id, c].to_numpy() + p['amplitude'] * delayed ** p['response_power']
                np.testing.assert_allclose(obs[c], expected, rtol=1e-12, atol=1e-12)
        else:
            assert lab[['d_ref', 'DI_ref', 'risk_ref', 'RUL_ref_hours']].isna().all().all()
    baseline, bm = load_baseline(split)
    assert bm['split'].eq(split).all() and len(baseline) == bm.baseline_candidate.sum()
    results.append(dict(split=split, saved_rows=len(x), scenes=x.index.get_level_values(0).nunique(),
                        unique_source_points=len(source_sets[split]), baseline_candidates=len(baseline)))
assert not source_sets['train'] & source_sets['validation']
assert not source_sets['train'] & source_sets['test']
assert not source_sets['validation'] & source_sets['test']
assert all(m['split'] == 'test' for m in manifest if m['family'] == 'piecewise')
report = dict(status='PASS', verified_files=len(hashes), splits=results,
              checks=['file hashes', 'saved keys and sensor-only inputs', 'raw lineage and split separation',
                      'unaltered cells', 'intentional missingness', '15-minute continuity',
                      'DI and risk', 'RUL from timestamps', 'censoring', 'channel-response reconstruction',
                      'baseline loader', 'test-only held-out family'])
(ROOT / 'verification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
