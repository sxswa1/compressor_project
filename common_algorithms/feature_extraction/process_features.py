"""Level and temporal-change features for multichannel process windows.

Preserves channel order and physical magnitude. No FFT, waveforms, implicit
unit conversion or per-window centring is used. Only finite inputs accepted.
"""
import numpy as np
import pandas as pd
from common_algorithms.data_processing.process_clock import normalize_process_time


def process_features(windows, channel_names, interval_seconds):
    x=np.asarray(windows,dtype=float)
    names=list(channel_names)
    if x.ndim!=3 or x.shape[1]!=len(names) or not names or len(set(names))!=len(names):
        raise ValueError("Use N x channels x time and unique channel_names")
    if x.shape[-1]<2 or not np.isfinite(x).all():
        raise ValueError("Each window needs at least two finite time points")
    if not np.isfinite(interval_seconds) or interval_seconds<=0:
        raise ValueError("interval_seconds must be positive")
    t=np.arange(x.shape[-1])*interval_seconds/3600
    centred=t-t.mean()
    slope=(x*centred).sum(axis=-1)/(centred**2).sum()
    diff=np.diff(x,axis=-1)
    blocks=[x.mean(-1),x.std(-1),x.min(-1),x.max(-1),x[:,:,-1],slope,np.abs(diff).mean(-1)]
    suffixes=["mean","std","min","max","last","slope_per_hour","mean_abs_step"]
    values=np.stack(blocks,axis=-1).reshape(len(x),len(names)*len(suffixes))
    feature_names=[f"{c}__{s}" for c in names for s in suffixes]
    return values,feature_names


def execute(df: pd.DataFrame, value_columns, window_id_column="window_id",
            time_column="timestamp", interval_seconds=360., **kwargs):
    """out_1 one feature row/window; out_2 schema. Complete windows required.

    Each window_id must identify one device, split and continuous interval.
    Input timestamps must be unique with exactly interval_seconds spacing.
    """
    values=[value_columns] if isinstance(value_columns,str) else list(value_columns)
    if not isinstance(df,pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("A DataFrame with unique columns is required")
    if not values or len(set(values))!=len(values) or set(values)&{time_column,window_id_column}:
        raise ValueError("Distinct numeric value columns are required")
    if any(c not in df for c in values+[window_id_column,time_column]):
        raise ValueError("Missing required columns")
    if df[window_id_column].isna().any():
        raise ValueError("Missing window IDs")
    rows=[]
    feature_names=[f"{c}__{s}" for c in values for s in ["mean","std","min","max","last","slope_per_hour","mean_abs_step"]]
    for key,part in df.groupby(window_id_column,sort=False):
        part=part.copy()
        part[time_column]=normalize_process_time(part[time_column])
        part=part.sort_values(time_column)
        delta=part[time_column].diff().dt.total_seconds().iloc[1:].to_numpy()
        if part[time_column].isna().any() or not np.allclose(delta,interval_seconds,rtol=0,atol=1e-6):
            raise ValueError("Window has invalid, duplicate or discontinuous timestamps")
        a=part[values].to_numpy(dtype=float).T[None]
        f,feature_names=process_features(a,values,interval_seconds)
        rows.append({window_id_column:key,**dict(zip(feature_names,f[0]))})
    result_df=pd.DataFrame(rows,columns=[window_id_column]+feature_names)
    info_df=pd.DataFrame({"feature_name":feature_names})
    return {"out_1":result_df,"out_2":info_df}
