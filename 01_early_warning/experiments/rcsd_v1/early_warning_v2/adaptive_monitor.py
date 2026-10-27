"""Condition grouping and causal residual-change diagnostics; no label inputs."""
from pathlib import Path
import sys
import pickle
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'early_warning_v1'))
from monitor import alert_state,quality_flags,raw_residuals


def residual_signals(x,b):
    old=b['base'];_,reference=raw_residuals(x,old)
    targets=old['config']['targets'];inputs=old['reference_inputs']
    speed=x['75SI865R.pv'].to_numpy()
    band=np.searchsorted(b['config']['speed_edges'],speed,side='right')-1
    grouped=np.full_like(reference,np.nan)
    good=np.isfinite(x[inputs]).all(axis=1).to_numpy()
    for i,model in b['band_models'].items():
        use=(band==i)&good
        if use.any():grouped[use]=x.loc[use,targets].to_numpy()-model.predict(x.loc[use,inputs])
    r=pd.DataFrame(reference)
    # The current residual is never part of its own historical median.
    return {'speed_binned':grouped,
            'change6':reference-r.shift(1).rolling(24,min_periods=24).median().to_numpy(),
            'change24':reference-r.shift(1).rolling(96,min_periods=96).median().to_numpy()}


def score_segment(x,b,apply_thresholds=True):
    cfg=b['config'];old=b['base'];n=len(x)
    signals=residual_signals(x,b)
    missing,stuck,_=quality_flags(x,old['stuck_points'])
    running=x['75SI865R.pv'].between(6050,8500).to_numpy()
    warmup=np.arange(n)<96
    eligible=running&~missing&~warmup
    result=pd.DataFrame({'step':np.arange(n),'warmup':warmup,'running':running,
                         'missing_flag':missing,'stuck_flag':stuck})
    normalized={}
    for method,values in signals.items():
        p=b['scales'][method]
        z=(values-np.array(p['center']))/np.array(p['scale'])/np.array(p['tail'])
        normalized[method]=z
    z=normalized['change24'];ema=np.zeros(z.shape[1]);s=np.full_like(z,np.nan)
    for i in range(n):
        if eligible[i] and np.isfinite(z[i]).all():
            ema=cfg['ewma_alpha']*z[i]+(1-cfg['ewma_alpha'])*ema;s[i]=ema
        else:ema[:]=0
    normalized['change24_ewma']=s
    for method,z in normalized.items():
        score=np.max(np.abs(z),axis=1)
        valid=eligible&np.isfinite(score)
        result[method+'_score']=score;result[method+'_valid']=valid
        dominant=np.argmax(np.nan_to_num(np.abs(z),nan=-1),axis=1)
        result[method+'_dominant_channel']=np.array(old['config']['targets'])[dominant]
        if apply_thresholds:
            threshold=b['thresholds'][method]
            result[method+'_point_alarm']=valid&(score>threshold)
            active,new=alert_state(score,valid,threshold,cfg)
            result[method+'_active']=active;result[method+'_new_alarm']=new
    return result


def execute(df:pd.DataFrame,model_path:str,timestamp_column='Timestamp',**kwargs):
    with Path(model_path).open('rb') as f:b=pickle.load(f)
    x=df[b['base']['sensor_columns']].astype(float).reset_index(drop=True)
    t=pd.to_datetime(df[timestamp_column],errors='raise').reset_index(drop=True)
    if t.isna().any() or t.duplicated().any() or not t.is_monotonic_increasing:
        raise ValueError('Timestamps must be present, unique and increasing')
    reset=t.diff().ne(pd.Timedelta(minutes=15))
    if 'segment_id' in df:
        seg=df.segment_id.reset_index(drop=True);reset|=seg.ne(seg.shift())
    results=[];events=[]
    for _,ids in reset.cumsum().groupby(reset.cumsum()).groups.items():
        result=score_segment(x.loc[ids].reset_index(drop=True),b)
        result.insert(0,'timestamp',t.loc[ids].to_numpy())
        for method in b['config']['methods']:
            for row in result.loc[result[method+'_new_alarm']].itertuples(index=False):
                # Column names contain no punctuation, so getattr is safe here.
                events.append({'timestamp':row.timestamp,'method':method,
                               'dominant_channel':getattr(row,method+'_dominant_channel')})
        results.append(result)
    return {'out_1':pd.concat(results,ignore_index=True) if results else pd.DataFrame(),
            'out_2':pd.DataFrame(events,columns=['timestamp','method','dominant_channel'])}
