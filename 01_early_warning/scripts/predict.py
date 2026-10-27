"""Predict DI for a single equipment CSV, preserving invalid/warm-up rows."""
from pathlib import Path
import argparse
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from common_algorithms.models.lstm import LSTMRegressor
from common_algorithms.models.gradient_boosting import GradientBoostingRegressor
from pipelines.baseline_data import windows, tabular_features, risk_levels


def predict_frame(frame, run_dir, model_name, timestamp_column='Timestamp'):
    if model_name not in ('lstm', 'gradient_boosting'):
        raise ValueError('unknown model')
    schema = json.loads((Path(run_dir)/'input_schema.json').read_text(encoding='utf-8'))
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError('a nonempty CSV with unique columns is required')
    missing = set(schema['sensors']+[timestamp_column])-set(frame.columns)
    if missing:
        raise ValueError('missing required columns: '+str(sorted(missing)))
    x = frame[schema['sensors']].apply(pd.to_numeric, errors='raise')
    values = x.to_numpy(dtype=np.float32)
    values[~x['75SI865R.pv'].between(6050, 8500).to_numpy()] = np.nan
    X, ends = windows(values, frame[timestamp_column], schema['history_points'], schema['sampling_minutes'],
                      frame['segment_id'] if 'segment_id' in frame else None)
    out = pd.DataFrame(dict(timestamp=pd.to_datetime(frame[timestamp_column]).to_numpy(),
                            DI_pred=np.nan, valid=False, status='insufficient_valid_running_history'))
    out['risk_pred'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    if len(ends):
        if model_name == 'lstm':
            model = LSTMRegressor.load(Path(run_dir)/'lstm.pt')
            raw = model.predict(X)
        else:
            model = GradientBoostingRegressor.load(Path(run_dir)/'gradient_boosting.pkl')
            raw = model.predict(tabular_features(X))
        p = np.clip(raw, 0, 1)
        out.loc[ends, 'DI_pred'] = p
        out.loc[ends, 'risk_pred'] = risk_levels(p)
        out.loc[ends, 'valid'] = True
        out.loc[ends, 'status'] = 'synthetic_DI_pilot_not_field_health'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', choices=['lstm','gradient_boosting'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timestamp-column', default='Timestamp')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists; choose a new file')
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result = predict_frame(pd.read_csv(args.input), args.run_dir, args.model, args.timestamp_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
