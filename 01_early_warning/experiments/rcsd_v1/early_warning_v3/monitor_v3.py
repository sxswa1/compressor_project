from pathlib import Path
import sys,pickle
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'early_warning_v2'))
from adaptive_monitor import score_segment as score_v2,residual_signals
from monitor import score_segment as score_v1,alert_state


def cumulative_evidence(x,b):
    base=b['base'];targets=base['base']['config']['targets']
    a=residual_signals(x,base)['change24'];p=base['scales']['change24']
    z=(a-np.array(p['center']))/np.array(p['scale'])/np.array(p['tail'])
    eligible=(np.arange(len(x))>=96)&np.isfinite(x).all(axis=1).to_numpy()&x['75SI865R.pv'].between(6050,8500).to_numpy()
    columns={}
    for group,directions in b['config']['groups'].items():
        indices=[targets.index(c) for c in directions]
        values=np.median(z[:,indices]*np.array(list(directions.values()))[None,:],axis=1)
        evidence=np.full(len(x),np.nan);pos=0.;neg=0.
        for i,v in enumerate(values):
            if not eligible[i] or not np.isfinite(v):pos=0.;neg=0.
            else:
                pos=max(0.,pos+v-b['config']['cusum_allowance'])
                neg=max(0.,neg-v-b['config']['cusum_allowance'])
                evidence[i]=max(pos,neg)
        columns[group]=evidence
    return pd.DataFrame(columns)


def retain_incident(new_alarm, resolved=None, initially_open=False):
    """Workflow state: absence of new evidence does NOT resolve an existing incident."""
    opened=bool(initially_open);out=[]
    resets=np.zeros(len(new_alarm),bool) if resolved is None else np.asarray(resolved,bool)
    for event,reset in zip(new_alarm,resets):
        if reset:opened=False
        if event:opened=True
        out.append(opened)
    return np.array(out,bool)


def score_segment(x,b):
    previous=score_v2(x,b['base'])
    instantaneous=score_v1(x,b['base']['base'])
    evidence=cumulative_evidence(x,b)
    scaled=evidence/np.array([b['group_scales'][g] for g in evidence])
    score=scaled.max(axis=1,skipna=False).to_numpy()
    valid=previous['change24_ewma_valid'].to_numpy()&np.isfinite(score)
    active,new=alert_state(score,valid,b['cusum_threshold'],b['config'])
    # Paper limits provide independent evidence, not a replacement for a health assessment.
    breach=np.zeros(len(x),bool)
    for c in b['base']['base']['config']['targets']:
        if 'TI' in c:breach|=x[c].gt(120).to_numpy()
        elif 'XI' in c:breach|=x[c].gt(89).to_numpy()
        elif 'ZI' in c:breach|=x[c].abs().gt(.5).to_numpy()
        elif c=='75PI808.pv':breach|=x[c].lt(9).to_numpy()
        elif c=='75PI823.pv':breach|=x[c].lt(1.8).to_numpy()
    current_valid=np.isfinite(x).all(axis=1).to_numpy()&x['75SI865R.pv'].between(6050,8500).to_numpy()
    limit_active,limit_new=alert_state(breach.astype(float)*2,current_valid,1.,b['config'])
    result=pd.DataFrame({'step':np.arange(len(x)),
                         'quality_missing':previous.missing_flag,'quality_stuck_suspected':previous.stuck_flag,
                         'instant_hint':instantaneous.forecast_point_alarm,
                         'change_active':previous.change24_ewma_active,'change_new':previous.change24_ewma_new_alarm,
                         'cusum_score':score,'cusum_valid':valid,'cusum_active':active,'cusum_new':new,
                         'paper_limit_breach':breach&current_valid,'limit_active':limit_active,'limit_new':limit_new})
    result['equipment_active']=result.change_active|result.cusum_active|result.limit_active
    # A new branch starting while another remains active is not a new equipment episode.
    result['equipment_new']=result.equipment_active&~result.equipment_active.shift(1,fill_value=False)
    result['incident_open']=retain_incident(result.equipment_new)
    result['health_assessment']='not_established_by_these_detectors'
    return result


def execute(df:pd.DataFrame,model_path:str,timestamp_column='Timestamp',resolved_column=None,initial_incident_open=False,**kwargs):
    with Path(model_path).open('rb') as f:b=pickle.load(f)
    x=df[b['base']['base']['sensor_columns']].astype(float).reset_index(drop=True)
    t=pd.to_datetime(df[timestamp_column],errors='raise').reset_index(drop=True)
    if t.isna().any() or t.duplicated().any() or not t.is_monotonic_increasing:raise ValueError('Invalid timestamp order')
    reset=t.diff().ne(pd.Timedelta(minutes=15))
    if 'segment_id' in df:
        s=df.segment_id.reset_index(drop=True);reset|=s.ne(s.shift())
    frames=[]
    for _,ids in reset.cumsum().groupby(reset.cumsum()).groups.items():
        result=score_segment(x.loc[ids].reset_index(drop=True),b)
        result.insert(0,'timestamp',t.loc[ids].to_numpy());frames.append(result)
    r=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not r.empty:
        resolved=None
        if resolved_column is not None:
            value=df[resolved_column]
            if value.isna().any() or not value.isin([True,False,0,1]).all():raise ValueError('Resolution input must contain explicit booleans')
            resolved=value.astype(bool).to_numpy()
            if np.any(resolved&r.equipment_active.to_numpy()):raise ValueError('Cannot resolve while equipment evidence remains active')
        r['incident_open']=retain_incident(r.equipment_new,resolved,initial_incident_open)
    return {'out_1':r,'out_2':r.loc[r.equipment_new,['timestamp']].copy() if not r.empty else pd.DataFrame(columns=['timestamp'])}
