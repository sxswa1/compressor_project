"""Causal monitoring; observations only. Never accepts true labels or event times."""
from pathlib import Path
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'health_baseline_v1'))
from baseline import make_features


def quality_flags(x, thresholds):
    missing=~np.isfinite(x.to_numpy()).all(axis=1)
    counts=np.zeros(x.shape,dtype=int)
    for j,c in enumerate(x):
        a=x[c].to_numpy()
        for i,v in enumerate(a):
            counts[i,j]=0 if not np.isfinite(v) else counts[i-1,j]+1 if i>0 and v==a[i-1] else 1
    stuck=counts>=np.array([thresholds[c] for c in x])[None,:]
    return missing,stuck.any(axis=1),counts


def alert_state(score, valid, threshold, cfg):
    """Issue only on the third available exceedance; never backdate."""
    state=False; high=0; low=0
    active=np.zeros(len(score),bool); new=np.zeros(len(score),bool)
    for i,(s,ok) in enumerate(zip(score,valid)):
        if not ok or not np.isfinite(s):
            state=False; high=0; low=0
            continue
        if not state:
            high=high+1 if s>threshold else 0
            if high>=cfg['alarm_consecutive_points']:
                state=True;new[i]=True;high=0
        else:
            low=low+1 if s<=threshold*cfg['recovery_threshold_ratio'] else 0
            if low>=cfg['recovery_consecutive_points']:
                state=False;low=0
        active[i]=state
    return active,new


def raw_residuals(x,bundle):
    targets=bundle['config']['targets']; inputs=bundle['reference_inputs']
    reference=np.full((len(x),len(targets)),np.nan)
    good=np.isfinite(x[inputs]).all(axis=1)
    if good.any(): reference[good]=bundle['reference_model'].predict(x.loc[good,inputs])
    f=make_features(x,targets).shift(1)
    previous=x[targets].shift(1)
    forecast=np.full_like(reference,np.nan)
    usable=np.isfinite(f[bundle['forecast_features']]).all(axis=1)&np.isfinite(previous).all(axis=1)
    if usable.any():
        forecast[usable]=previous.loc[usable].to_numpy()+bundle['forecast_model'].predict(f.loc[usable,bundle['forecast_features']])
    actual=x[targets].to_numpy()
    return actual-forecast,actual-reference


def score_segment(x,bundle,apply_thresholds=True):
    cfg=bundle['config']; n=len(x)
    forecast,reference=raw_residuals(x,bundle)
    missing,stuck,counts=quality_flags(x,bundle['stuck_points'])
    speed=x['75SI865R.pv'].between(*cfg['operating_speed_bounds']).to_numpy()
    warmup=np.arange(n)<cfg['history_points']
    quality=missing|stuck
    # Constant values are only a suspicion. Do not veto all monitoring on that alone.
    allowed=speed & ~missing & ~warmup
    zf=(forecast-np.array(bundle['forecast_center']))/np.array(bundle['forecast_scale'])
    zr=(reference-np.array(bundle['reference_center']))/np.array(bundle['reference_scale'])
    zf=zf/np.array(bundle.get('forecast_tail_scale',np.ones(len(cfg['targets']))))
    zr=zr/np.array(bundle.get('reference_tail_scale',np.ones(len(cfg['targets']))))
    sf=np.max(np.abs(zf),axis=1); sr=np.max(np.abs(zr),axis=1)
    ema=np.zeros(reference.shape[1]); se=np.full(n,np.nan)
    for i in range(n):
        if not allowed[i] or not np.isfinite(zr[i]).all():
            ema[:]=0
        else:
            ema=cfg['ewma_alpha']*zr[i]+(1-cfg['ewma_alpha'])*ema
            se[i]=np.abs(ema).max()
    result=pd.DataFrame({'step':np.arange(n),'missing_flag':missing,'stuck_flag':stuck,
                         'quality_flag':quality,'running':speed,'warmup':warmup,
                         'forecast_score':sf,'reference_score':sr,'reference_ewma_score':se})
    for name in cfg['methods']:
        scores=result[name+'_score'].to_numpy()
        valid=allowed & np.isfinite(scores)
        result[name+'_valid']=valid
        if apply_thresholds:
            threshold=bundle['thresholds'][name]
            result[name+'_point_alarm']=valid&(scores>threshold)
            active,new=alert_state(scores,valid,threshold,cfg)
            result[name+'_active']=active;result[name+'_new_alarm']=new
    return result


def execute(df: pd.DataFrame,model_path:str,timestamp_column='Timestamp',**kwargs):
    """Return out_1 monitoring trace, out_2 event starts; full history per call."""
    with Path(model_path).open('rb') as f: bundle=pickle.load(f)
    x=df[bundle['sensor_columns']].astype(float).reset_index(drop=True)
    t=pd.to_datetime(df[timestamp_column],errors='raise').reset_index(drop=True)
    if t.isna().any() or t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError('Timestamps must be present, unique and increasing')
    breaks=t.diff().ne(pd.Timedelta(minutes=15))
    if 'segment_id' in df:
        s=df.segment_id.reset_index(drop=True); breaks|=s.ne(s.shift())
    traces=[];events=[]
    for _,ids in breaks.cumsum().groupby(breaks.cumsum()).groups.items():
        r=score_segment(x.loc[ids].reset_index(drop=True),bundle)
        r.insert(0,'timestamp',t.loc[ids].to_numpy())
        for method in bundle['config']['methods']:
            for timestamp in r.loc[r[method+'_new_alarm'],'timestamp']:
                events.append({'timestamp':timestamp,'method':method,'event':'alarm_start'})
        traces.append(r)
    return {'out_1':pd.concat(traces,ignore_index=True) if traces else pd.DataFrame(),
            'out_2':pd.DataFrame(events,columns=['timestamp','method','event'])}
