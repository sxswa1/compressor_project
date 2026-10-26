"""Causal feature construction and inference. No training at import time."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def make_features(frame, targets):
    parts = [frame[targets].add_suffix('__now')]
    for lag in [1,4,8,24]:
        parts.append(frame[targets].shift(lag).add_suffix(f'__lag{lag}'))
    speed = '75SI865R.pv'
    parts.append(frame[[speed]].add_suffix('__now'))
    for lag in [1,4,8,24]: parts.append(frame[[speed]].shift(lag).add_suffix(f'__lag{lag}'))
    for window in [4,24,96]:
        parts.append(frame[targets].rolling(window,min_periods=window).mean().add_suffix(f'__mean{window}'))
    parts.append(frame[targets].rolling(24,min_periods=24).std(ddof=0).add_suffix('__std24'))
    for lag in [4,24]: parts.append((frame[targets]-frame[targets].shift(lag)).add_suffix(f'__delta{lag}'))
    others = [c for c in frame if c not in targets and c != speed and c.endswith('.pv')]
    parts.append(frame[others].add_suffix('__context_now'))
    parts.append(frame[others].shift(8).add_suffix('__context_lag8'))
    return pd.concat(parts,axis=1)


def columns_for(features, variant, targets):
    group=variant.replace('ridge_','').replace('extra_trees_','')
    if group=='full': return list(features)
    own=[c for c in features if c.split('__')[0] in targets]
    base=[c for c in own if c.endswith('__now') or '__lag' in c]
    speed=[c for c in features if c.startswith('75SI865R.pv__')]
    if group=='self': return base
    if group=='speed': return base+speed
    if group=='trend': return own+speed
    raise ValueError(variant)


def execute(df: pd.DataFrame, model_path: str, timestamp_column='Timestamp', **kwargs):
    """Predict at each arrival time from continuous sensor histories.

    Only load a locally trusted model pickle. Predictions are not alarms.
    The caller may provide segment_id to force resets at known state boundaries.
    Missing/nonfinite input history suppresses output; no backward filling.
    """
    with Path(model_path).open('rb') as f: bundle=pickle.load(f)
    targets=bundle['targets']; horizon=bundle['horizon']; sensor_cols=bundle['sensor_columns']
    if timestamp_column not in df: raise ValueError('Timestamp column missing')
    missing=set(sensor_cols)-set(df)
    if missing: raise ValueError(f'Missing sensor columns: {sorted(missing)}')
    time=pd.to_datetime(df[timestamp_column],errors='raise')
    if time.isna().any() or not time.is_monotonic_increasing or time.duplicated().any():
        raise ValueError('Timestamps must be present, unique and increasing')
    values=df[sensor_cols].astype(float).reset_index(drop=True)
    time=time.reset_index(drop=True)
    reset=time.diff().ne(pd.Timedelta(minutes=15))
    if 'segment_id' in df:
        seg=df.segment_id.reset_index(drop=True)
        reset |= seg.ne(seg.shift())
    output=[]
    for _,idx in reset.cumsum().groupby(reset.cumsum()).groups.items():
        v=values.loc[idx]; f=make_features(v,targets)
        history_ok=np.isfinite(v).all(axis=1).rolling(96,min_periods=96).sum().eq(96)
        ok=history_ok.copy(); ok.iloc[:95]=False
        pred=np.full((len(v),len(targets)),np.nan)
        if ok.any():
            current=v.loc[ok,targets].to_numpy()
            pred[ok.to_numpy()]=current if bundle['model'] is None else current+bundle['model'].predict(f.loc[ok,bundle['features']])
        result=pd.DataFrame(pred,columns=targets,index=v.index)
        result.insert(0,'origin_timestamp',time.loc[idx].to_numpy())
        result.insert(1,'forecast_timestamp',(time.loc[idx]+pd.Timedelta(minutes=15*horizon)).to_numpy())
        result['prediction_valid']=ok.to_numpy()
        status=np.where(ok,'ok','invalid_history').astype(object)
        status[:min(95,len(v))]='warmup'
        result['status']=status
        output.append(result)
    if not output:
        return {'out_1':pd.DataFrame(columns=['origin_timestamp','forecast_timestamp',*targets,'prediction_valid','status'])}
    return {'out_1':pd.concat(output).reset_index(drop=True)}
