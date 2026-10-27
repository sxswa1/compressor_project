"""Batch inference for one device; labels and scenario metadata are not inputs."""
from pathlib import Path
import copy
import pickle
import sys
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'experiments/rcsd_v1'
for directory in [ROOT,ROOT/'final_model_fit_v1',ROOT/'warning_matched_baselines_v1']:
    sys.path.insert(0,str(directory))
from inference import predict_scene
from portable import read
import experiment as warning


def predict(frame,seed,di_model='lstm_window',rul_model='trees_history',
            rul_variant='residual_augmented',warning_method='change_ewma',root=ROOT):
    if seed not in [20261023,20261024,20261025]:raise ValueError('Unknown recorded seed')
    if warning_method not in ['change_ewma','cusum','v3_core_limits']:raise ValueError('Unknown warning method')
    if rul_variant!='raw_original' and rul_model not in ['trees_history','gradient_boosting_history']:
        raise ValueError('This RUL model supports only raw_original; residual/augmented variants use trees or gradient boosting')
    if len(frame)==0:raise ValueError('Input has no rows')
    meta=read(root/'final_model_fit_v1/results/preprocessing.json');sensors=meta['sensors']
    missing=set(['Timestamp',*sensors])-set(frame)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    timestamp=pd.DatetimeIndex(pd.to_datetime(frame.Timestamp,errors='raise'))
    if timestamp.hasnans or timestamp.has_duplicates or not timestamp.is_monotonic_increasing:
        raise ValueError('Timestamp must be unique, valid and increasing')
    frame=frame.reset_index(drop=True)
    values=frame[sensors].astype(float)
    boundary=np.ones(len(frame),dtype=bool)
    boundary[1:]=np.diff(timestamp.asi8)!=pd.Timedelta(minutes=15).value
    if 'segment_id' in frame:
        if frame.segment_id.isna().any():raise ValueError('segment_id cannot be missing')
        boundary[1:] |= frame.segment_id.to_numpy()[1:]!=frame.segment_id.to_numpy()[:-1]
    with (root/'warning_lstm_budget_v1/results/bundle.pkl').open('rb') as f:
        bundle=copy.copy(pickle.load(f))
    # These three warning choices do not need the LSTM / Isolation Forest branches.
    bundle['lstm']={};bundle['iforest']={}
    output=[]
    di_seed=seed if di_model.startswith('trees_') or di_model=='lstm_window' else 20261024
    rul_seed=seed if rul_model in ['trees_history','lstm_window'] else 20261024
    for _,indices in pd.Series(np.arange(len(frame))).groupby(boundary.cumsum()):
        ids=indices.to_numpy();x=values.iloc[ids].reset_index(drop=True);t=timestamp[ids]
        di=predict_scene('di',di_model,di_seed,x,t,root=root/'final_model_fit_v1')
        rul=predict_scene('rul',rul_model,rul_seed,x,t,variant=rul_variant,root=root/'final_model_fit_v1')
        alerts=warning.score(x[bundle['sensors']],bundle)
        out=pd.DataFrame(dict(Timestamp=t,seed=seed,DI=di.prediction,
            risk_level=di.risk_level.astype('Int64'),di_valid=di.valid,
            RUL_hours=rul.prediction,rul_valid=rul.valid,
            warning_valid=alerts.valid,warning_active=alerts[warning_method+'_active'],
            warning_new=alerts[warning_method+'_new']))
        for key in ['lower_hours','upper_hours']:
            out['RUL_'+key]=rul[key] if key in rul else np.nan
        output.append(out)
    return pd.concat(output,ignore_index=True)
