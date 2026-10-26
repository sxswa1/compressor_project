"""Matched warning comparison. Models never receive scenario labels/lineage."""
from pathlib import Path
import sys, json, pickle, hashlib, gzip, time, os
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; OUT=ROOT/'results'
D1=PROJECT/'RCSD-1YD Data Set/dataset_v1/data'
D2=PROJECT/'RCSD-1YD Data Set/controlled_dataset_v2/data'
ST=PROJECT/'early_warning_v2/stress_test/results'
sys.path.insert(0,str(PROJECT/'early_warning_v3'))
from monitor_v3 import score_segment as original_v3, cumulative_evidence
from adaptive_monitor import residual_signals
from monitor import raw_residuals, alert_state

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def load_raw():
    meta=pd.read_csv(D1/'raw_metadata.csv',parse_dates=['source_timestamp']).set_index('source_row_id')
    meta=meta.loc[meta.source_timestamp.lt('2022-09-01')]
    # Source file is chronological: nrows prevents parsing final-test observations.
    assert np.array_equal(meta.index,np.arange(len(meta)))
    raw=pd.read_csv(D1/'raw_observations.csv.gz',nrows=len(meta)).set_index('source_row_id')
    assert raw.index.equals(meta.index)
    return raw,meta

def parts(raw,meta,start,stop,candidate=True):
    m=meta.loc[meta.source_timestamp.ge(start)&meta.source_timestamp.lt(stop)]
    if candidate: m=m.loc[m.baseline_candidate]
    reset=m.source_timestamp.diff().ne(pd.Timedelta(minutes=15))
    if candidate: reset|=m.segment_id.ne(m.segment_id.shift())
    return [(raw.loc[g.index].reset_index(drop=True),g.reset_index()) for _,g in m.groupby(reset.cumsum())]

def support(x):
    finite=pd.Series(np.isfinite(x.to_numpy()).all(axis=1))
    return (finite.rolling(97,min_periods=97).sum().eq(97).to_numpy()
            &x['75SI865R.pv'].between(6050,8500).to_numpy())

def windows(x,targets):
    a=x.to_numpy(dtype=np.float32); ids=np.flatnonzero(support(x))
    X=np.stack([a[i-96:i] for i in ids]) if len(ids) else np.empty((0,96,a.shape[1]),np.float32)
    return X,x[targets].to_numpy(dtype=np.float32)[ids],ids

class Net(nn.Module):
    def __init__(self,n_in,n_out,hidden=24):
        super().__init__(); self.rnn=nn.LSTM(n_in,hidden,batch_first=True); self.head=nn.Linear(hidden,n_out)
    def forward(self,x):return self.head(self.rnn(x)[0][:,-1])

def predict_lstm(bundle,X):
    net=Net(len(bundle['xmean']),len(bundle['ymean']),bundle['hidden'])
    net.load_state_dict(bundle['state']); net.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(X),256):
            z=(X[i:i+256]-bundle['xmean'])/bundle['xscale']
            preds.append(net(torch.from_numpy(z)).numpy()*bundle['yscale']+bundle['ymean'])
    return np.concatenate(preds) if preds else np.empty((0,len(bundle['ymean'])))

def train_lstm(X,Y,seed,cfg,epochs=None,validation=None):
    torch.manual_seed(seed)
    p=cfg['lstm']; xm=X.mean(axis=(0,1),dtype=np.float64).astype(np.float32)
    xs=np.maximum(X.std(axis=(0,1),dtype=np.float64),1e-6).astype(np.float32)
    ym=Y.mean(axis=0,dtype=np.float64).astype(np.float32)
    ys=np.maximum(Y.std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
    tx=torch.from_numpy((X-xm)/xs); ty=torch.from_numpy((Y-ym)/ys)
    net=Net(X.shape[-1],Y.shape[-1],p['hidden']); opt=torch.optim.Adam(net.parameters(),lr=p['lr'])
    history=[]; best=float('inf'); chosen=0; wait=0
    if validation is not None:
        vx=torch.from_numpy((validation[0]-xm)/xs); vy=torch.from_numpy((validation[1]-ym)/ys)
    for ep in range(1,(epochs or p['max_epochs'])+1):
        net.train(); order=torch.randperm(len(tx)); losses=[]
        for ix in order.split(p['batch_size']):
            opt.zero_grad(); loss=((net(tx[ix])-ty[ix])**2).mean(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step();losses.append((float(loss.detach()),len(ix)))
        row={'epoch':ep,'train_mse':sum(v*n for v,n in losses)/len(tx)}
        if validation is not None:
            net.eval()
            with torch.no_grad():
                mse=sum(float(((net(vx[i:i+256])-vy[i:i+256])**2).sum()) for i in range(0,len(vx),256))/vy.numel()
            row['validation_mse']=mse
            if mse<best-p['min_delta']:best=mse;chosen=ep;wait=0
            else:wait+=1
        history.append(row)
        if validation is not None and wait>=p['patience']:break
    return dict(xmean=xm,xscale=xs,ymean=ym,yscale=ys,hidden=p['hidden'],state=net.state_dict(),seed=seed),history,chosen or ep

def floor(targets):return np.array([.001 if 'ZI' in c else .01 if 'PI' in c else .1 for c in targets])

def residual_scale(a,targets,q,center=None,scale=None):
    center=np.median(a,axis=0) if center is None else center
    scale=np.maximum(1.4826*np.median(abs(a-center),axis=0),floor(targets)) if scale is None else scale
    tail=np.maximum(1,np.quantile(abs((a-center)/scale),q,axis=0))
    return dict(center=center,scale=scale,tail=tail)

def multiscale(a,p):return np.max(abs((a-p['center'])/p['scale']/p['tail']),axis=1)

def raw_scores(x,b):
    old=b['v3']['base']['base']; target=old['config']['targets']; valid=support(x)
    values=x[target].to_numpy(); f,_=raw_residuals(x,old)
    scores={'static':multiscale(values,b['scalers']['static']),
            'persistence':multiscale(values-x[target].shift(1).to_numpy(),b['scalers']['persistence']),
            'ridge_forecast':multiscale(f,b['scalers']['ridge_forecast'])}
    change=residual_signals(x,b['v3']['base'])['change24']; p=b['v3']['base']['scales']['change24']
    z=(change-np.asarray(p['center']))/np.asarray(p['scale'])/np.asarray(p['tail'])
    ema=np.zeros(len(target)); change_score=np.full(len(x),np.nan)
    for i in range(len(x)):
        if valid[i] and np.isfinite(z[i]).all():
            ema=.2*z[i]+.8*ema;change_score[i]=np.max(abs(ema))
        else:ema[:]=0
    scores['change_ewma']=change_score
    evidence=cumulative_evidence(x,b['v3'])
    scores['cusum']=np.max(evidence.to_numpy()/np.array([b['v3']['group_scales'][c] for c in evidence]),axis=1)
    X,Y,ids=windows(x,target)
    for name,model in b['lstm'].items():
        score=np.full(len(x),np.nan)
        if len(ids):score[ids]=multiscale(Y-predict_lstm(model,X),b['scalers'][name])
        scores[name]=score
    for name,model in b['iforest'].items():
        score=np.full(len(x),np.nan)
        if valid.any():score[valid]=np.maximum(0,-model.score_samples(x.to_numpy()[valid])-b['if_centers'][name])
        scores[name]=score
    return scores,valid

def score(x,b):
    scores,valid=raw_scores(x,b);out=pd.DataFrame({'step':np.arange(len(x)),'valid':valid})
    for name,s in scores.items():
        assert np.isfinite(s[valid]).all(),name
        out[name+'_score']=s
        out[name+'_active']=alert_state(s,valid,b['thresholds'][name],b['cfg'])[0]
    out['v3_core_active']=out.change_ewma_active|out.cusum_active
    breach=np.zeros(len(x),bool)
    for c in b['targets']:
        if 'TI' in c:breach|=x[c].gt(120).to_numpy()
        elif 'XI' in c:breach|=x[c].gt(89).to_numpy()
        elif 'ZI' in c:breach|=x[c].abs().gt(.5).to_numpy()
        elif c=='75PI808.pv':breach|=x[c].lt(9).to_numpy()
        elif c=='75PI823.pv':breach|=x[c].lt(1.8).to_numpy()
    out['limits_active']=alert_state(breach.astype(float)*2,valid,1,b['cfg'])[0]
    for name in list(scores)+['v3_core']:
        out[name+'_limits_active']=out[name+'_active']|out.limits_active
    anchor=original_v3(x,b['v3'])
    out['original_v3_active']=anchor.equipment_active
    out['original_v3_valid']=anchor.cusum_valid
    for c in list(out):
        if c.endswith('_active'):out[c[:-7]+'_new']=out[c]&~out[c].shift(1,fill_value=False)
    return out

def inventory():
    paths=[ROOT/'plan.json',ROOT/'experiment.py',PROJECT/'early_warning_v3/monitor_bundle.pkl',
           PROJECT/'early_warning_v3/monitor_v3.py',PROJECT/'early_warning_v2/adaptive_monitor.py',
           PROJECT/'early_warning_v1/monitor.py',PROJECT/'health_baseline_v1/baseline.py',
           D1/'raw_observations.csv.gz',D1/'raw_metadata.csv',
           D2/'scene_manifest.json',D2/'validation_observations.csv.gz',D2/'validation_lineage.csv.gz',
           ST/'manifest.json',ST/'observations.csv.gz']
    paths+=list((PROJECT/'two_algorithm_delivery_v2').glob('**/*.pkl'))
    return {str(p.relative_to(PROJECT)):digest(p) for p in paths}

def fit(raw,meta,cfg):
    with (PROJECT/'early_warning_v3/monitor_bundle.pkl').open('rb') as f:v3=pickle.load(f)
    targets=v3['base']['base']['config']['targets'];sensors=v3['base']['base']['sensor_columns']
    assert list(raw)==sensors
    b=dict(cfg=cfg,v3=v3,targets=targets,sensors=sensors,scalers={},lstm={},iforest={},if_centers={})
    roles={name:parts(raw,meta,a,z) for name,a,z in [
        ('inner_fit',cfg['fit_start'],cfg['inner_fit_before']),
        ('inner_validation',cfg['inner_fit_before'],cfg['fit_before']),
        ('fit',cfg['fit_start'],cfg['fit_before']),
        ('calibration',cfg['fit_before'],cfg['calibration_before'])]}
    batches={}; keys=[]
    for role,seq in roles.items():
        xx=[];yy=[]
        for k,(x,m) in enumerate(seq):
            X,Y,ids=windows(x,targets)
            take=np.arange(len(ids))[::cfg['stride']] if role.endswith('fit') else np.arange(len(ids))
            xx.append(X[take]);yy.append(Y[take]);ids=ids[take]
            keys.append(m.iloc[ids][['source_row_id','source_timestamp']].assign(role=role,segment=k))
        batches[role]=(np.concatenate(xx),np.concatenate(yy))
    key=pd.concat(keys,ignore_index=True);key.to_csv(OUT/'sample_keys.csv',index=False)
    dump(OUT/'split_counts.json',{r:dict(rows=sum(len(x) for x,_ in roles[r]),segments=len(roles[r]),windows=len(batches[r][0])) for r in roles})
    histories=[];selected=[]
    for seed in cfg['seeds']:
        _,hist,ep=train_lstm(*batches['inner_fit'],seed,cfg,validation=batches['inner_validation'])
        histories.extend(dict(seed=seed,phase='inner',**r) for r in hist)
        model,hist,_=train_lstm(*batches['fit'],seed,cfg,epochs=ep)
        histories.extend(dict(seed=seed,phase='refit',**r) for r in hist)
        b['lstm'][f'lstm_{seed}']=model
        selected.append(dict(seed=seed,epoch=ep,epochs_run=len([r for r in histories if r['seed']==seed and r['phase']=='inner'])))
        print('LSTM',seed,'chosen',ep,flush=True)
    pd.DataFrame(histories).to_csv(OUT/'learning_curves.csv',index=False);dump(OUT/'selected_epochs.json',selected)
    # Same endpoint sampling as LSTM for current-sensor baselines.
    fit_values=raw.loc[key.loc[key.role.eq('fit'),'source_row_id']].to_numpy()
    current=fit_values[:,[sensors.index(t) for t in targets]]
    center=np.median(current,axis=0);scale=np.maximum(1.4826*np.median(abs(current-center),axis=0),floor(targets))
    cal_current=np.concatenate([x.loc[support(x),targets].to_numpy() for x,_ in roles['calibration']])
    b['scalers']['static']=residual_scale(cal_current,targets,cfg['quantile'],center,scale)
    for name in ['persistence','ridge_forecast']:
        arrays=[]
        for x,_ in roles['calibration']:
            a=x[targets].to_numpy()-x[targets].shift(1).to_numpy() if name=='persistence' else raw_residuals(x,v3['base']['base'])[0]
            arrays.append(a[support(x)])
        b['scalers'][name]=residual_scale(np.concatenate(arrays),targets,cfg['quantile'])
    for name,model in b['lstm'].items():
        X,Y=batches['calibration'];b['scalers'][name]=residual_scale(Y-predict_lstm(model,X),targets,cfg['quantile'])
    cal_values=np.concatenate([x.to_numpy()[support(x)] for x,_ in roles['calibration']])
    for seed in cfg['seeds']:
        name=f'iforest_{seed}'; model=IsolationForest(**cfg['isolation_forest'],random_state=seed,n_jobs=1).fit(fit_values)
        b['iforest'][name]=model;b['if_centers'][name]=float(np.median(-model.score_samples(cal_values)))
    cal=[raw_scores(x,b) for x,_ in roles['calibration']]
    b['thresholds']={name:float(np.quantile(np.concatenate([s[name][v] for s,v in cal]),cfg['quantile'])) for name in cal[0][0]}
    assert all(t>0 for t in b['thresholds'].values())
    dump(OUT/'thresholds.json',b['thresholds'])
    with (OUT/'bundle.pkl').open('wb') as f:pickle.dump(b,f,protocol=5)
    return b

def scene_inputs():
    manifest=json.loads((D2/'scene_manifest.json').read_text(encoding='utf-8'))
    obs=pd.read_csv(D2/'validation_observations.csv.gz').set_index(['scene_id','step'])
    lineage=pd.read_csv(D2/'validation_lineage.csv.gz').set_index(['scene_id','step'])
    for m in manifest:
        if m['split']!='validation':continue
        sid=m['scene_id'];x=obs.loc[sid]
        yield dict(source='v2',scene_id=sid,block_id=m['block_id'],kind='replay' if m['benchmark']=='replay' or m['group']=='none' else 'ramp',
                   benchmark=m['benchmark'],group=m['group'],strength=m.get('amplitude_multiplier'),duration=m.get('nominal_hours'),
                   onset=m.get('onset'),endpoint=m.get('failure_step') if m.get('observed') else None),x,lineage.loc[sid,'source_row_id'].to_numpy()
    obs=pd.read_csv(ST/'observations.csv.gz').set_index(['scene_id','step'])
    for m in json.loads((ST/'manifest.json').read_text()):
        yield dict(source='stress',scene_id=m['scene_id'],block_id=m['block_id'],kind=m['kind'],benchmark=m['kind'],group=m['group'],
                   strength=m['strength'],duration=m['duration'],onset=m['onset_step'],endpoint=m['reference_endpoint_step']),obs.loc[m['scene_id']],np.arange(m['first_row'],m['first_row']+m['rows'])

def event_metrics(r,control,meta):
    rows=[]
    for c in r:
        if not c.endswith('_active'):continue
        name=c[:-7];active=r[c].to_numpy();new=r[name+'_new'].to_numpy();onset=meta['onset'];end=meta['endpoint']
        valid=r.original_v3_valid.to_numpy() if name=='original_v3' else r.valid.to_numpy()
        item=dict(**meta,method=name,valid_points=int(valid.sum()),rows=len(r),active_points=int((active&valid).sum()),
                  new_events=int(new.sum()),control_active_points=int((control[c].to_numpy()&valid).sum()))
        if end is not None:
            starts=np.flatnonzero(new);starts=starts[(starts>onset)&(starts<=end)]
            paired=[int(k) for k in starts if not control[c].iloc[max(0,k-2):k+1].any()]
            first=int(starts[0]) if len(starts) else None;pfirst=paired[0] if paired else None
            item.update(first=first,paired_first=pfirst,hit=bool(first is not None),paired_hit=bool(pfirst is not None),
                        two_hour=bool(first is not None and (end-first)*.25>=2),
                        paired_two_hour=bool(pfirst is not None and (end-pfirst)*.25>=2),
                        lead_hours=(end-first)*.25 if first is not None else None,
                        paired_lead_hours=(end-pfirst)*.25 if pfirst is not None else None)
        if onset is not None:item['pre_onset_events']=int(new[:onset+1].sum())
        if meta['kind']=='spike':item['spike_event_hit']=bool(new[onset:onset+int(meta['duration'])].any())
        if meta['kind']=='abnormal_start':item['additional_start_events']=int((new&~control[c].to_numpy()).sum())
        rows.append(item)
    return rows

def evaluate(raw,meta,b):
    empirical=[]
    for k,(x,m) in enumerate(parts(raw,meta,'2022-07-01','2022-09-01',candidate=False)):
        r=score(x,b);r.insert(0,'source_row_id',m.source_row_id);r.insert(1,'timestamp',m.source_timestamp);r['segment']=k;empirical.append(r)
    empirical=pd.concat(empirical,ignore_index=True);empirical.to_csv(OUT/'empirical_trace.csv.gz',index=False,compression='gzip')
    records=[]
    for c in empirical:
        if c.endswith('_active'):
            name=c[:-7];valid=empirical.original_v3_valid if name=='original_v3' else empirical.valid
            records.append(dict(method=name,valid_points=int(valid.sum()),total_points=len(empirical),
                active_fraction=float(empirical.loc[valid,c].mean()),new_events=int(empirical[name+'_new'].sum()),
                events_per_valid_day=float(empirical[name+'_new'].sum()/(valid.sum()/96))))
    pd.DataFrame(records).to_csv(OUT/'empirical_metrics.csv',index=False)
    rows=[];index=[];cache={}
    with gzip.open(OUT/'scene_traces.csv.gz','wt',encoding='utf-8',newline='') as f:
        for i,(m,x,ids) in enumerate(scene_inputs()):
            assert len(ids)==len(x) and meta.loc[ids,'split'].eq('validation').all()
            x=x[b['sensors']].reset_index(drop=True);r=score(x,b)
            key=(int(ids[0]),len(ids));assert np.array_equal(ids,np.arange(ids[0],ids[0]+len(ids)))
            if key not in cache:cache[key]=score(raw.loc[ids].reset_index(drop=True),b)
            control=cache[key]; rows.extend(event_metrics(r,control,m))
            r.insert(0,'scene_id',m['scene_id']);r.insert(0,'source',m['source']);r.to_csv(f,index=False,header=i==0)
            index.append(dict(**m,first_source_row=int(ids[0]),last_source_row=int(ids[-1]),rows=len(ids)))
            if (i+1)%40==0:print('Scored scenes',i+1,flush=True)
    pd.DataFrame(index).to_csv(OUT/'scene_index.csv',index=False)
    with gzip.open(OUT/'control_traces.csv.gz','wt',encoding='utf-8',newline='') as f:
        for i,((first,n),r) in enumerate(cache.items()):
            r=r.copy();r.insert(0,'first_source_row',first);r.insert(1,'scene_rows',n);r.to_csv(f,index=False,header=i==0)
    ev=pd.DataFrame(rows);ev.to_csv(OUT/'scene_metrics.csv',index=False)
    complete=ev.loc[ev.endpoint.notna()].copy()
    complete['stratum']=np.where(complete.source.eq('v2'),complete.benchmark,
        complete.duration.astype(str)+'h_amp'+complete.strength.astype(str))
    summary=complete.groupby(['source','stratum','method']).agg(events=('scene_id','size'),backgrounds=('block_id','nunique'),
        hits=('hit','sum'),paired_hits=('paired_hit','sum'),two_hour=('two_hour','sum'),paired_two_hour=('paired_two_hour','sum'),
        median_paired_lead=('paired_lead_hours','median')).reset_index()
    summary.to_csv(OUT/'event_summary.csv',index=False)

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);threadpool_limits(limits=2)
    OUT.mkdir(exist_ok=True);cfg=json.loads((ROOT/'plan.json').read_text(encoding='utf-8'));before=inventory()
    dump(OUT/'provenance.json',dict(input_sha256=before,python=sys.version,numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,final_test_scored=False))
    raw,meta=load_raw();start=time.monotonic();b=fit(raw,meta,cfg);print('Fit complete',flush=True)
    evaluate(raw,meta,b)
    assert inventory()==before
    dump(OUT/'run_status.json',dict(status='COMPLETED',seconds=time.monotonic()-start,source_files_unchanged=True,final_test_scored=False))
    print('Completed',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
