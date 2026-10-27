from pathlib import Path
import sys
import pickle
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from adaptive_monitor import residual_signals, score_segment, execute
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DATA = ROOT.parent / 'RCSD-1YD Data Set/dataset_v1/data'
sys.path.insert(0, str(DATA.parent))
from load_dataset import load_scenes

def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'config.json').read_text())
    with (ROOT.parent / 'early_warning_v1/monitor_bundle.pkl').open('rb') as f:
        base = pickle.load(f)
    raw = pd.read_csv(DATA / 'raw_observations.csv.gz').set_index('source_row_id')
    meta = pd.read_csv(DATA / 'raw_metadata.csv', parse_dates=['source_timestamp']).set_index('source_row_id')
    keep = meta['split'].isin(['train', 'validation'])
    raw = raw.loc[keep]
    meta = meta.loc[keep]
    fit = meta.baseline_candidate & (meta.source_timestamp < cfg['fit_before'])
    cal = meta.baseline_candidate & (meta.source_timestamp >= cfg['fit_before']) & (meta.source_timestamp < cfg['calibration_before'])
    targets = base['config']['targets']
    inputs = base['reference_inputs']
    band = np.searchsorted(cfg['speed_edges'], raw['75SI865R.pv'], side='right') - 1
    models = {}
    band_info = []
    for i in range(3):
        subset = raw.loc[fit & (band == i)]
        fallback = len(subset) < cfg['minimum_band_fit_points']
        models[i] = base['reference_model'] if fallback else make_pipeline(StandardScaler(), Ridge(alpha=cfg['ridge_alpha'])).fit(subset[inputs].iloc[::4], subset[targets].iloc[::4])
        band_info.append({'band': i, 'points': len(subset), 'pooled_fallback': fallback})
    b = {'config': cfg, 'base': base, 'band_models': models, 'scales': {}, 'band_info': band_info}
    segments = [raw.loc[g.index].reset_index(drop=True) for _, g in meta[cal].groupby('segment_id')]
    arrays = {k: [] for k in ['speed_binned', 'change6', 'change24']}
    for segment in segments:
        signals = residual_signals(segment, b)
        for method, a in signals.items():
            arrays[method].append(a[96:])
    for method, values in arrays.items():
        a = np.concatenate(values)
        a = a[np.isfinite(a).all(axis=1)]
        center = np.median(a, axis=0)
        floors = np.array([0.001 if 'ZI' in c else 0.01 if 'PI' in c else 0.1 for c in targets])
        scale = np.maximum(1.4826 * np.median(np.abs(a - center), axis=0), floors)
        tail = np.maximum(np.quantile(np.abs((a - center) / scale), cfg['threshold_quantile'], axis=0), 1)
        b['scales'][method] = {'center': center.tolist(), 'scale': scale.tolist(), 'tail': tail.tolist()}
    cal_trace = pd.concat([score_segment(x, b, False) for x in segments], ignore_index=True)
    b['thresholds'] = {method: float(cal_trace.loc[cal_trace[method + '_valid'], method + '_score'].quantile(cfg['threshold_quantile'])) for method in cfg['methods']}
    with (ROOT / 'monitor_bundle.pkl').open('wb') as f:
        pickle.dump(b, f, protocol=5)
if __name__ == '__main__':
    main()
