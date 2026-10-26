"""One fixed evaluator for development replay and the later reserved run."""
import argparse
import gzip
import pickle
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from common import (ROOT,WORK,FIT,D1,D2,PROTOCOL,read,write,digest,check_frozen,assert_release,warning_module,
    raw_windows,tabular_features,features,risk_levels,load_model,predict_array)
from generate import load_raw,controlled,stress
from metrics import di_score,rul_score,warning_summary,bootstrap_primary

class CsvStream:
    def __init__(self,path):
        self.stream=gzip.open(path,'wt',encoding='utf-8',newline='',compresslevel=1);self.first=True
    def append(self,frame):
        frame.to_csv(self.stream,index=False,header=self.first);self.first=False
    def close(self):self.stream.close()

def warning_metadata(source,m):
    if source=='v2':
        return dict(source=source,scene_id=m['scene_id'],block_id=m['block_id'],kind='replay' if m['benchmark']=='replay' else 'ramp',
            benchmark=m['benchmark'],group=m['group'],strength=m['amplitude_multiplier'],duration=m['nominal_hours'],
            onset=m['onset'],endpoint=m['failure_step'] if m['observed'] else None)
    return dict(source=source,scene_id=m['scene_id'],block_id=m['block_id'],kind=m['kind'],benchmark=m['kind'],group=m['group'],
        strength=m['strength'],duration=m['duration'],onset=m['onset_step'],endpoint=m['reference_endpoint_step'])

def stratum(source,m):
    if source=='v2':return m['variant']
    if m['kind']=='ramp_plateau':return f'ramp_{m["duration"]}h_amp{m["strength"]:g}'
    return m['kind']

def feature_inputs(x,timestamps,sensors,reference,preprocess):
    X,ends=raw_windows(x,timestamps,sensors)
    raw=tabular_features(X)
    residual,rends=features(x,timestamps,sensors,reference,preprocess)
    np.testing.assert_array_equal(ends,rends)
    # Reasons are mutually exclusive in this declared priority order.
    finite=pd.Series(np.isfinite(x[sensors]).all(axis=1))
    running=x['75SI865R.pv'].between(6050,8500).reset_index(drop=True)
    continuous=pd.Series(pd.DatetimeIndex(pd.to_datetime(timestamps))).diff().eq(pd.Timedelta(minutes=15))
    reason=np.full(len(x),'valid',dtype=object)
    reason[~continuous.rolling(95).sum().eq(95)]='time_gap_history'
    reason[~running.rolling(96).sum().eq(96)]='stopped_history'
    reason[~finite.rolling(96).sum().eq(96)]='missing_sensor_history'
    reason[:95]='insufficient_history'
    assert np.array_equal(np.flatnonzero(reason=='valid'),ends)
    return X,raw,residual,ends,reason

def regress_scene(m,x,y,lineage,registry,models,reference,radii,preprocess):
    X,raw,residual,ends,reason=feature_inputs(x,lineage.source_timestamp,preprocess['sensors'],reference,preprocess)
    outputs={}
    for r in registry.to_dict('records'):
        task,name,variant,asset=r['task'],r['name'],r['variant'],r['asset']
        F=X if name=='lstm_window' else (X[:,-1] if name.endswith('_current') else residual if variant.startswith('residual_') else raw)
        values=predict_array(models[(task,asset)],F) if len(ends) else np.empty(0)
        assert np.isfinite(values).all()
        full=np.full(len(x),np.nan);full[ends]=values
        pred=np.clip(full,0,1) if task=='di' else np.maximum(full,0)
        outputs[(task,asset)]=(full,pred)
    primary=outputs[('di','ridge_history__20261024')][1]
    primary_risk=np.where(np.isfinite(primary),risk_levels(primary),np.nan)
    reference_risk=y.risk_ref.to_numpy(dtype=float,na_value=np.nan)
    for r in registry.to_dict('records'):
        task,asset=r['task'],r['asset'];rawpred,pred=outputs[(task,asset)]
        valid=reason=='valid';family=r['name'] if task=='di' else r['name']+'__'+r['variant']
        label_ok=y.di_label_valid.to_numpy()&np.isfinite(y.DI_ref) if task=='di' else (
            y.rul_label_valid.to_numpy()&y.event_observed.to_numpy()&~y.right_censored.to_numpy()&np.isfinite(y.RUL_ref_hours)&y.RUL_ref_hours.ge(0))
        lo,hi=np.full(len(x),np.nan),np.full(len(x),np.nan)
        if task=='rul' and asset in radii:
            q=radii[asset]['radius_hours'];q=float('inf') if q is None else q
            lo=np.maximum(0,pred-q);hi=pred+q
        f=pd.DataFrame(dict(task=task,asset=asset,family=family,seed=r['seed'],scene_id=m['scene_id'],block_id=m['block_id'],
            benchmark=m['benchmark'],stratum=m['variant'],step=np.arange(len(x)),source_row_id=lineage.source_row_id,
            source_timestamp=lineage.source_timestamp,valid=valid,invalid_reason=reason,label_valid=label_ok,
            evaluation_eligible=valid&label_ok,DI_ref=y.DI_ref,RUL_ref_hours=y.RUL_ref_hours,
            raw_prediction=rawpred,prediction=pred,primary_predicted_risk=primary_risk,reference_risk=reference_risk,
            lower_hours=lo,upper_hours=hi,event_observed=y.event_observed,right_censored=y.right_censored))
        yield f

def summarize_regression(frames,out):
    summary=[];scene_rows=[];background_rows=[];stratified=[];family_rows=[]
    for (task,asset),batches in frames.items():
        frame=pd.concat(batches,ignore_index=True);fn=di_score if task=='di' else rul_score
        identity=dict(task=task,asset=asset,family=frame.family.iloc[0],seed=int(frame.seed.iloc[0]))
        for label,g in frame.groupby('stratum',sort=False):
            summary.append(dict(**identity,stratum=label,**fn(g)))
        for (label,sid,bid),g in frame.groupby(['stratum','scene_id','block_id'],sort=False):
            scene_rows.append(dict(**identity,stratum=label,scene_id=sid,block_id=bid,**fn(g)))
        for (label,bid),g in frame.groupby(['stratum','block_id'],sort=False):
            background_rows.append(dict(**identity,stratum=label,block_id=bid,**fn(g)))
        if task=='rul':
            for risk_type,column in [('predicted','primary_predicted_risk'),('reference','reference_risk')]:
                for label,g in frame.groupby('stratum',sort=False):
                    for level in range(1,5):
                        subset=g.loc[g[column].eq(level)]
                        stratified.append(dict(**identity,stratum=label,risk_type=risk_type,risk_level=level,**fn(subset)))
        del frame
    summary=pd.DataFrame(summary);scenes=pd.DataFrame(scene_rows)
    summary.to_csv(out/'regression_metrics.csv',index=False)
    scenes.to_csv(out/'regression_scene_metrics.csv',index=False)
    pd.DataFrame(background_rows).to_csv(out/'regression_background_metrics.csv',index=False)
    pd.DataFrame(stratified).to_csv(out/'rul_risk_strata.csv',index=False)
    for (task,family,label),g in summary.groupby(['task','family','stratum'],sort=False):
        metric='scene_macro_mae' if task=='di' else 'scene_macro_rmse_hours'
        family_rows.append(dict(task=task,family=family,stratum=label,seeds=len(g),metric=metric,
            seed_mean=float(g[metric].mean()),seed_min=float(g[metric].min()),seed_max=float(g[metric].max())))
    pd.DataFrame(family_rows).to_csv(out/'regression_family_summary.csv',index=False)
    bootstrap=[]
    for task,primary in [('di','ridge_history'),('rul','gradient_boosting_history__residual_augmented')]:
        bootstrap.append(bootstrap_primary(scenes.loc[scenes.task.eq(task)],task,primary,[20261023,20261024,20261025]))
    pd.concat(bootstrap,ignore_index=True).to_csv(out/'regression_background_bootstrap.csv',index=False)

def empirical_warning(raw,meta,b,w,split,out):
    meta=meta.copy();meta['source_timestamp']=pd.to_datetime(meta.source_timestamp)
    a,z=('2022-07-01','2022-09-01') if split=='validation' else ('2022-09-01','2023-01-01')
    traces=[]
    for k,(x,m) in enumerate(w.parts(raw,meta,a,z,candidate=False)):
        r=w.score(x,b);r.insert(0,'source_row_id',m.source_row_id);r.insert(1,'timestamp',m.source_timestamp);r['segment']=k;traces.append(r)
    frame=pd.concat(traces,ignore_index=True);frame.to_csv(out/'empirical_trace.csv.gz',index=False)
    rows=[]
    for c in frame:
        if c.endswith('_active'):
            method=c[:-7];valid=frame.original_v3_valid if method=='original_v3' else frame.valid
            rows.append(dict(method=method,valid_points=int(valid.sum()),total_points=len(frame),active_fraction=float(frame.loc[valid,c].mean()),
                new_events=int(frame[method+'_new'].sum()),events_per_valid_day=float(frame[method+'_new'].sum()/(valid.sum()/96))))
    pd.DataFrame(rows).to_csv(out/'empirical_metrics.csv',index=False)

def evaluate(split,out):
    check_frozen()
    if split=='test':
        assert_release()
        directory=WORK/read(PROTOCOL)['dataset']['output_directory']
        verification=read(ROOT/'results/reserved_dataset_verification.json')
        assert verification['status']=='PASS' and verification['dataset_checksums_sha256']==digest(directory/'checksums.json')
        if (ROOT/'reserved_run_started.json').exists():raise FileExistsError('Reserved scoring already started; any bug-driven rerun requires a separate versioned record')
    if out.exists():raise FileExistsError('Use a new output directory; previous results must be retained')
    if split=='test':write(ROOT/'reserved_run_started.json',dict(output=str(out),release_sha256=digest(ROOT/'release_manifest.json')))
    out.mkdir(parents=True);start=time.monotonic()
    raw,meta=load_raw(split);w=warning_module()
    with (WORK/read(PROTOCOL)['warning']['frozen_bundle']).open('rb') as f:b=pickle.load(f)
    registry=pd.read_csv(FIT/'results/model_registry.csv')
    models={(r.task,r.asset):load_model(WORK/r.model_path,r['name']) for _,r in registry.iterrows()}
    with (FIT/'models/rul/reference.pkl').open('rb') as f:reference=pickle.load(f)
    radii=read(FIT/'models/rul/interval_radii.json');preprocess=read(FIT/'results/preprocessing.json')
    streams={name:CsvStream(out/f'{name}.csv.gz') for name in ['warning_scene_traces','warning_control_traces','regression_traces','regression_evaluation']}
    ledger=[];index=[];events=[];cache={};frames={};eligibility=[]
    if split=='test':
        from reserved_io import reserved_scenes
        directory=WORK/read(PROTOCOL)['dataset']['output_directory']
        ledger=read(directory/'inclusion_ledger.json')
        iterators=[('v2',reserved_scenes('controlled')),('stress',reserved_scenes('stress'))]
    else:
        iterators=[('v2',controlled(raw,meta,split,ledger)),('stress',stress(raw,meta,split,ledger))]
    try:
        for source,iterator in iterators:
            for m,x,y,l,mask in iterator:
                sid=m['scene_id'];wm=warning_metadata(source,m);label=stratum(source,m)
                ids=l.source_row_id.to_numpy();timestamps=pd.to_datetime(l.source_timestamp)
                assert timestamps.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
                assert meta.loc[ids,'split'].eq(split).all()
                r=w.score(x[b['sensors']].reset_index(drop=True),b)
                key=(int(ids[0]),len(ids));assert np.array_equal(ids,np.arange(ids[0],ids[0]+len(ids)))
                if key not in cache:
                    cache[key]=w.score(raw.loc[ids,b['sensors']].reset_index(drop=True),b)
                    control=cache[key].copy();control.insert(0,'first_source_row',key[0]);control.insert(1,'scene_rows',key[1])
                    streams['warning_control_traces'].append(control)
                ev=w.event_metrics(r,cache[key],wm)
                for item in ev:item['stratum']=label
                events.extend(ev)
                trace=r.copy();trace.insert(0,'scene_id',sid);trace.insert(0,'source',source);streams['warning_scene_traces'].append(trace)
                index.append(dict(**wm,stratum=label,first_source_row=key[0],last_source_row=int(ids[-1]),rows=len(ids),
                    censored=m.get('censored',False),di_rul_label_support=source=='v2' and m['benchmark']!='replay' if source=='v2' else False))
                if source=='v2':
                    for frame in regress_scene(m,x,y,l,registry,models,reference,radii,preprocess):
                        streams['regression_traces'].append(frame)
                        selected=frame.loc[frame.evaluation_eligible].copy()
                        if len(selected):
                            streams['regression_evaluation'].append(selected)
                            frames.setdefault((frame.task.iloc[0],frame.asset.iloc[0]),[]).append(selected)
                        # Availability is model-independent; record once per task.
                        if frame.asset.iloc[0] in ['ridge_history__20261024','gradient_boosting_history__residual_augmented__20261024']:
                            eligibility.append(dict(task=frame.task.iloc[0],scene_id=sid,block_id=m['block_id'],stratum=label,
                                rows=len(frame),valid_outputs=int(frame.valid.sum()),label_valid=int(frame.label_valid.sum()),
                                evaluated_points=len(selected),label_valid_but_no_output=int((frame.label_valid&~frame.valid).sum()),
                                event_observed=m['observed'],right_censored=m['censored'],
                                **{reason:int(frame.invalid_reason.eq(reason).sum()) for reason in ['insufficient_history','time_gap_history','missing_sensor_history','stopped_history']}))
                if len(index)%25==0:print('Evaluated scenes',len(index),flush=True)
    finally:
        for stream in streams.values():stream.close()
    events=pd.DataFrame(events);events.to_csv(out/'warning_scene_metrics.csv',index=False)
    pd.DataFrame(index).to_csv(out/'scene_index.csv',index=False)
    warning_summary(events).to_csv(out/'warning_event_summary.csv',index=False)
    pd.DataFrame(eligibility).to_csv(out/'regression_eligibility.csv',index=False)
    summarize_regression(frames,out)
    empirical_warning(raw,meta,b,w,split,out)
    from summaries import supplementary
    supplementary(out)
    write(out/'inclusion_ledger.json',ledger)
    check_frozen()
    write(out/'run_status.json',dict(status='COMPLETED_PENDING_VERIFICATION',split=split,scenes=len(index),
        warning_configurations=26,regression_point_models=34,seconds=time.monotonic()-start,
        method_or_threshold_changes=False,final_test_read=split=='test',
        stress_di_rul='Not scored: original stress data has no DI/RUL reference labels',
        input_source_last_timestamp=str(meta.source_timestamp.iloc[-1]),
        code_sha256={p.name:digest(p) for p in ROOT.glob('*.py')}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--output',type=__import__('pathlib').Path)
    args=parser.parse_args()
    if args.split=='test':assert_release()
    out=args.output or ROOT/('development_results' if args.split=='validation' else 'reserved_results')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):evaluate(args.split,out)

if __name__=='__main__':main()
