"""Fit Jan-Apr, calibrate May-Jun, evaluate Jul-Aug; never score final test."""
from pathlib import Path
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import sklearn
from monitor import score_segment, quality_flags, raw_residuals, alert_state, execute, make_features
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'RCSD-1YD Data Set/dataset_v1/data'
OUT = ROOT / 'results'
sys.path.insert(0, str(DATA.parent))
from load_dataset import load_scenes

def regression(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def residual_stats(values, targets):
    a = np.concatenate(values)
    a = a[np.isfinite(a).all(axis=1)]
    center = np.median(a, axis=0)
    floor = np.array([0.001 if 'ZI' in c else 0.01 if 'PI' in c else 0.1 for c in targets])
    scale = np.maximum(1.4826 * np.median(np.abs(a - center), axis=0), floor)
    return (center.tolist(), scale.tolist())

def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'config.json').read_text())
    targets = cfg['targets']
    raw = pd.read_csv(DATA / 'raw_observations.csv.gz').set_index('source_row_id')
    meta = pd.read_csv(DATA / 'raw_metadata.csv', parse_dates=['source_timestamp']).set_index('source_row_id')
    dev = meta['split'].isin(['train', 'validation'])
    raw = raw.loc[dev]
    meta = meta.loc[dev]
    candidate = meta.baseline_candidate
    fit_mask = candidate & (meta.source_timestamp < cfg['fit_before'])
    cal_mask = candidate & (meta.source_timestamp >= cfg['fit_before']) & (meta.source_timestamp < cfg['calibration_before'])
    inputs = [c for c in raw if c not in targets]
    fit_x = raw.loc[fit_mask].iloc[::cfg['training_stride']]
    ref = regression(cfg['ridge_alpha']).fit(fit_x[inputs], fit_x[targets])
    fs = []
    ys = []
    run_lengths = {c: [] for c in raw}
    run_counts = []
    for _, g in meta.loc[fit_mask].groupby('segment_id'):
        x = raw.loc[g.index]
        f = make_features(x, targets).iloc[95:-1]
        fs.append(f)
        ys.append((x[targets].shift(-1) - x[targets]).loc[f.index])
        for c in x:
            lengths = x.groupby(x[c].ne(x[c].shift()).cumsum()).size()
            run_lengths[c].extend(lengths.tolist())
        run_counts.append(quality_flags(x, {c: 10 ** 9 for c in x})[2])
    features = pd.concat(fs)
    delta = pd.concat(ys)
    cols = [c for c in features if '__context_' not in c]
    forecast = regression(cfg['ridge_alpha']).fit(features[cols].iloc[::4], delta.iloc[::4])
    counts = np.concatenate(run_counts)
    quantile = 1 - cfg['stuck_point_tail_budget'] / len(raw.columns)
    stuck = {c: max(cfg['minimum_stuck_points'], int(np.floor(np.quantile(counts[:, j], quantile))) + 1) for j, c in enumerate(raw)}
    bundle = {'config': cfg, 'reference_inputs': inputs, 'sensor_columns': list(raw), 'reference_model': ref, 'forecast_model': forecast, 'forecast_features': cols, 'stuck_points': stuck}
    calibrations = []
    fr = []
    rr = []
    for _, g in meta.loc[cal_mask].groupby('segment_id'):
        x = raw.loc[g.index].reset_index(drop=True)
        calibrations.append(x)
        f, r = raw_residuals(x, bundle)
        missing, st, _ = quality_flags(x, stuck)
        usable = (np.arange(len(x)) >= 96) & ~missing
        fr.append(f[usable])
        rr.append(r[usable])
    bundle['forecast_center'], bundle['forecast_scale'] = residual_stats(fr, targets)
    bundle['reference_center'], bundle['reference_scale'] = residual_stats(rr, targets)
    for name, values in [('forecast', fr), ('reference', rr)]:
        values = np.concatenate(values)
        values = values[np.isfinite(values).all(axis=1)]
        z = np.abs((values - np.array(bundle[name + '_center'])) / np.array(bundle[name + '_scale']))
        bundle[name + '_tail_scale'] = np.maximum(np.quantile(z, cfg['channel_tail_scaling_quantile'], axis=0), 1).tolist()
    calibration = pd.concat([score_segment(x, bundle, False) for x in calibrations], ignore_index=True)
    bundle['thresholds'] = {m: float(calibration.loc[calibration[m + '_valid'], m + '_score'].quantile(cfg['threshold_quantile'])) for m in cfg['methods']}
    with (ROOT / 'monitor_bundle.pkl').open('wb') as f:
        pickle.dump(bundle, f, protocol=5)
if __name__ == '__main__':
    main()
