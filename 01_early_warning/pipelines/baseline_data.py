"""RCSD v2 DI pilot. Reads only the existing January-June training files.

Synthetic labels and lineage never enter the model feature matrix.
Source times, gaps, stopped operation and missing sensors constrain each window.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def windows(values, timestamps, window=96, minutes=15, segments=None):
    """Return causal windows and end-row positions; retain no invalid windows."""
    values = np.asarray(values, dtype=np.float32)
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps, errors='raise'))
    if values.ndim != 2 or len(values) != len(timestamps):
        raise ValueError('sensor rows and timestamps must align')
    if not isinstance(window, int) or window < 1 or minutes <= 0:
        raise ValueError('window and sampling period must be positive')
    if timestamps.hasnans or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError('timestamps must be valid, unique and increasing within a scene')
    boundary = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        boundary[1:] = np.diff(timestamps.asi8) != pd.Timedelta(minutes=minutes).value
    if segments is not None:
        segments = np.asarray(segments)
        if len(segments) != len(values) or pd.isna(segments).any():
            raise ValueError('segments must align and cannot be missing')
        boundary[1:] |= segments[1:] != segments[:-1]
    ends, consecutive = [], 0
    for i, valid in enumerate(np.isfinite(values).all(axis=1)):
        if boundary[i]:
            consecutive = 0
        consecutive = consecutive + 1 if valid else 0
        if consecutive >= window:
            ends.append(i)
    ends = np.asarray(ends, dtype=int)
    if not len(ends):
        return np.empty((0, window, values.shape[1]), dtype=np.float32), ends
    return np.stack([values[i-window+1:i+1] for i in ends]), ends


def tabular_features(X):
    """125 features from the same 96-point windows used by the LSTM."""
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1] != 96:
        raise ValueError('this research feature recipe requires exactly 96 history points')
    return np.concatenate([X[:, -1], X[:, -24:].mean(axis=1), X[:, -1]-X[:, -24],
                           X.mean(axis=1), X[:, -1]-X[:, 0]], axis=1)


def load_training_pilot(data_dir, config):
    data_dir = Path(data_dir)
    sensors = json.loads((data_dir/'model_input_columns.json').read_text(encoding='utf-8'))
    if len(sensors) != 25 or any(not s.startswith('75') or not s.endswith('.pv') for s in sensors):
        raise ValueError('expected the documented 25 RCSD sensors')
    manifest = pd.DataFrame(json.loads((data_dir/'scene_manifest.json').read_text(encoding='utf-8')))
    selected = manifest.loc[manifest.split.eq('train') & manifest.benchmark.eq('main')].copy()
    blocks = selected.groupby('block_id').first_source_row.min().sort_values().index.tolist()
    a, b = config['train_backgrounds'], config['evaluation_backgrounds']
    if a < 1 or b < 1 or a+b > len(blocks) or config['training_stride'] < 1:
        raise ValueError('invalid chronological split or stride')
    train_blocks, eval_blocks = blocks[:a], blocks[a:a+b]
    selected = selected.loc[selected.block_id.isin(train_blocks+eval_blocks)]
    fit_max = int(selected.loc[selected.block_id.isin(train_blocks), 'last_source_row'].max())
    eval_min = int(selected.loc[selected.block_id.isin(eval_blocks), 'first_source_row'].min())
    if fit_max >= eval_min:
        raise ValueError('training and evaluation source ranges overlap')
    tables = []
    for kind in ('observations', 'labels', 'lineage'):
        table = pd.read_csv(data_dir/f'train_{kind}.csv.gz').set_index(['scene_id', 'step'])
        if table.index.has_duplicates:
            raise ValueError('duplicate scene/step keys')
        tables.append(table)
    obs, labels, lineage = tables
    if not obs.index.equals(labels.index) or not obs.index.equals(lineage.index):
        raise ValueError('observation, label and lineage keys differ')
    batches = {'train': [], 'evaluation': []}
    for meta in selected.to_dict('records'):
        sid = meta['scene_id']
        x, y, source = obs.loc[sid], labels.loc[sid], lineage.loc[sid]
        if not x.index.is_monotonic_increasing or not source.block_id.eq(meta['block_id']).all():
            raise ValueError('invalid scene order or source background')
        values = x[sensors].to_numpy(dtype=np.float32)
        values[~x['75SI865R.pv'].between(6050, 8500).to_numpy()] = np.nan
        X, ends = windows(values, source.source_timestamp, config['history_points'], config['sampling_minutes'])
        keep = y.iloc[ends].di_label_valid.eq(True).to_numpy() & np.isfinite(y.iloc[ends].DI_ref.to_numpy())
        X, ends = X[keep], ends[keep]
        role = 'train' if meta['block_id'] in train_blocks else 'evaluation'
        if role == 'train':
            X, ends = X[::config['training_stride']], ends[::config['training_stride']]
        if not len(ends):
            continue
        keys = pd.DataFrame(dict(scene_id=sid, block_id=meta['block_id'], step=x.index[ends],
                                  source_timestamp=source.iloc[ends].source_timestamp.to_numpy(),
                                  DI_ref=y.iloc[ends].DI_ref.to_numpy(dtype=float)))
        if not keys.DI_ref.between(0, 1).all():
            raise ValueError('DI labels must lie in [0,1]')
        batches[role].append((X, keys))
    result = {}
    for role, batch in batches.items():
        if not batch:
            raise ValueError(f'no valid {role} windows')
        result[role] = (np.concatenate([x for x, _ in batch]), pd.concat([k for _, k in batch], ignore_index=True))
    split = dict(train_blocks=train_blocks, evaluation_blocks=eval_blocks,
                 train_last_source_row=fit_max, evaluation_first_source_row=eval_min,
                 development_validation_read=False, final_test_read=False,
                 limitation='Fixed v2 generator amplitudes use all Jan-Jun training backgrounds; not fully prospective generation.')
    return result, sensors, split


def risk_levels(di):
    di = np.asarray(di)
    return np.where(di >= .85, 1, np.where(di >= .65, 2, np.where(di >= .4, 3, 4)))


def metrics(frame):
    if frame.empty or not np.isfinite(frame[['DI_ref', 'DI_pred']].to_numpy()).all():
        raise ValueError('evaluation requires finite nonempty predictions and references')
    error = frame.DI_pred-frame.DI_ref
    r, p = risk_levels(frame.DI_ref), risk_levels(frame.DI_pred)
    scene = pd.DataFrame(dict(scene=frame.scene_id, absolute=error.abs())).groupby('scene').absolute.mean()
    return dict(points=len(frame), scenes=int(frame.scene_id.nunique()),
                mae=float(error.abs().mean()), rmse=float(np.sqrt(np.mean(error**2))),
                scene_macro_mae=float(scene.mean()), risk_accuracy=float(np.mean(r == p)),
                risk_recall={str(k):float(np.mean(p[r == k] == k)) if np.any(r == k) else None for k in range(1,5)})
