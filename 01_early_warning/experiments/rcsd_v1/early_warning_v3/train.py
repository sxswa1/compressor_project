from pathlib import Path
import sys, json, pickle, hashlib
import numpy as np
import pandas as pd
from monitor_v3 import cumulative_evidence, score_segment, retain_incident, execute
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DATA = ROOT.parent / 'RCSD-1YD Data Set/dataset_v1/data'
STRESS = ROOT.parent / 'early_warning_v2/stress_test/results'
sys.path.insert(0, str(DATA.parent))
from load_dataset import load_scenes

def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'config.json').read_text())
    with (ROOT.parent / 'early_warning_v2/monitor_bundle.pkl').open('rb') as f:
        base = pickle.load(f)
    b = {'config': cfg, 'base': base}
    raw = pd.read_csv(DATA / 'raw_observations.csv.gz').set_index('source_row_id')
    meta = pd.read_csv(DATA / 'raw_metadata.csv', parse_dates=['source_timestamp']).set_index('source_row_id')
    keep = meta['split'].isin(['train', 'validation'])
    raw = raw.loc[keep]
    meta = meta.loc[keep]
    cal = meta.baseline_candidate & (meta.source_timestamp >= '2022-05-01') & (meta.source_timestamp < '2022-07-01')
    segments = [raw.loc[g.index].reset_index(drop=True) for _, g in meta[cal].groupby('segment_id')]
    accumulated = pd.concat([cumulative_evidence(x, b) for x in segments], ignore_index=True)
    b['group_scales'] = {c: float(max(1, accumulated[c].quantile(cfg['calibration_quantile']))) for c in accumulated}
    normalized = accumulated / np.array([b['group_scales'][c] for c in accumulated])
    b['cusum_threshold'] = float(normalized.max(axis=1, skipna=False).quantile(cfg['calibration_quantile']))
    with (ROOT / 'monitor_bundle.pkl').open('wb') as f:
        pickle.dump(b, f, protocol=5)
if __name__ == '__main__':
    main()
