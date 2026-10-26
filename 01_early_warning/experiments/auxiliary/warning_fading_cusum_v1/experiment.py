"""One bounded fading-CUSUM experiment, raw observations restricted to Jan-Jun."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]/'rcsd_v1'
D1=WORK/'RCSD-1YD Data Set/dataset_v1/data'
D2=WORK/'RCSD-1YD Data Set/controlled_dataset_v2/data'
E,C,F,L,U,V='ewma','cusum','fading','limits','combined','fading_combined'
METHODS=[E,C,F,L,U,V]


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(1024*1024),b''): h.update(a)
    return h.hexdigest()


def parts(raw,meta,a,z,candidate):
    m=meta.loc[meta.source_timestamp.ge(a)&meta.source_timestamp.lt(z)]
    if candidate: m=m.loc[m.baseline_candidate]
    reset=m.source_timestamp.diff().ne(pd.Timedelta(minutes=15))
    if candidate: reset|=m.segment_id.ne(m.segment_id.shift())
    return [(raw.loc[g.index].reset_index(drop=True),g.reset_index()) for _,g in m.groupby(reset.cumsum(),sort=False)]


def load_raw():
    meta=pd.read_csv(D1/'raw_metadata.csv',parse_dates=['source_timestamp']).set_index('source_row_id')
    meta=meta.loc[meta.source_timestamp.lt('2022-07-01')]
    assert np.array_equal(meta.index,np.arange(len(meta)))
    raw=pd.read_csv(D1/'raw_observations.csv.gz',nrows=len(meta)).set_index('source_row_id')
    assert raw.index.equals(meta.index)
    return raw,meta


def raw_change(x,b):
    valid=pd.Series(np.isfinite(x[b['sensors']]).all(axis=1)).rolling(97).sum().eq(97).to_numpy() & x['75SI865R.pv'].between(6050,8500).to_numpy()
    good=np.isfinite(x[b['context']]).all(axis=1)
    residual=np.full((len(x),len(b['targets'])),np.nan)
    if good.any(): residual[good]=x.loc[good,b['targets']].to_numpy()-b['reference'].predict(x.loc[good,b['context']])
    change=residual-pd.DataFrame(residual).shift(1).rolling(96,min_periods=96).median().to_numpy()
    assert np.isfinite(change[valid]).all()
    return change,valid


def evidence(values,valid,decay,allowance):
    pos=np.zeros(values.shape[1]);neg=pos.copy();result=np.full_like(values,np.nan)
    for i,v in enumerate(values):
        if not valid[i] or not np.isfinite(v).all(): pos[:]=0;neg[:]=0
        else:
            pos=np.maximum(0,decay*pos+v-allowance)
            neg=np.maximum(0,decay*neg-v-allowance)
            result[i]=np.maximum(pos,neg)
    return result


def signals(x,b):
    change,valid=raw_change(x,b)
    z=(change-b['center'])/b['scale']/b['tail']
    values=np.column_stack([np.median(z[:,[b['targets'].index(c) for c in directions]]*np.array(list(directions.values())),axis=1) for directions in b['groups'].values()])
    old=evidence(values,valid,1,b['allowance'])
    new=evidence(values,valid,b['decay'],b['allowance'])
    ema=np.zeros(z.shape[1]);ewma=np.full(len(x),np.nan)
    for i,row in enumerate(z):
        if valid[i]: ema=.2*row+.8*ema;ewma[i]=np.max(abs(ema))
        else: ema[:]=0
    return valid,z,values,ewma,old,new


def state(score,valid,threshold):
    active=np.zeros(len(score),bool);opened=False;high=low=0
    for i,(v,ok) in enumerate(zip(score,valid)):
        if not ok or not np.isfinite(v): opened=False;high=low=0;continue
        if not opened:
            high=high+1 if v>threshold else 0
            if high>=3: opened=True;high=0
        else:
            low=low+1 if v<=threshold*.7 else 0
            if low>=4: opened=False;low=0
        active[i]=opened
    return active


def score(x,b):
    valid,z,values,ewma,old,new=signals(x,b)
    scaled_old=old/b['group_scales'][C]
    scaled_new=new/b['group_scales'][F]
    result=pd.DataFrame(dict(step=np.arange(len(x)),valid=valid,ewma_score=ewma,
        cusum_score=np.max(scaled_old,axis=1),fading_score=np.max(scaled_new,axis=1)))
    for j,name in enumerate(b['groups']):
        result[name+'_increment']=values[:,j]
        result[name+'_cusum_evidence']=old[:,j]
        result[name+'_fading_evidence']=new[:,j]
    for name,values in [(C,scaled_old),(F,scaled_new)]:
        result[name+'_dominant']=np.array(list(b['groups']))[np.argmax(np.nan_to_num(values,nan=-1),axis=1)]
    for method in [E,C,F]: result[method+'_active']=state(result[method+'_score'],valid,b['thresholds'][method])
    breach=np.zeros(len(x),bool)
    for c in b['targets']:
        if 'TI' in c: breach|=x[c].gt(120).to_numpy()
        elif 'XI' in c: breach|=x[c].gt(89).to_numpy()
        elif 'ZI' in c: breach|=x[c].abs().gt(.5).to_numpy()
        elif c=='75PI808.pv': breach|=x[c].lt(9).to_numpy()
        elif c=='75PI823.pv': breach|=x[c].lt(1.8).to_numpy()
    result[L+'_active']=state(breach.astype(float)*2,valid,1)
    result[U+'_active']=result[E+'_active']|result[C+'_active']|result[L+'_active']
    result[V+'_active']=result[E+'_active']|result[F+'_active']|result[L+'_active']
    for method in METHODS:
        a=result[method+'_active'];result[method+'_new']=a&~a.shift(1,fill_value=False)
    return result


def fit(raw,meta,plan,out):
    targets=read(WORK/'early_warning_v1/config.json')['targets']
    groups=read(WORK/'early_warning_v3/config.json')['groups']
    b=dict(sensors=list(raw),targets=targets,context=[c for c in raw if c not in targets],groups=groups,
           allowance=.05,decay=2**(-.25/24))
    roles={role:parts(raw,meta,*dates,True) for role,dates in plan['calendar'].items() if role!='evaluation'}
    keys=[]
    fitframe=pd.concat([x for x,_ in roles['reference_fit']],ignore_index=True).iloc[::4]
    b['reference']=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(fitframe[b['context']],fitframe[targets])
    for role,segs in roles.items():
        for segment,(x,m) in enumerate(segs): keys.append(m[['source_row_id','source_timestamp']].assign(role=role,segment=segment))
    key=pd.concat(keys,ignore_index=True);key.to_csv(out/'role_source_rows.csv',index=False)
    assert key.groupby('source_row_id').role.nunique().max()==1
    a=np.concatenate([raw_change(x,b)[0][raw_change(x,b)[1]] for x,_ in roles['residual_scaling']])
    b['center']=np.median(a,axis=0)
    floor=np.array([.001 if 'ZI' in c else .01 if 'PI' in c else .1 for c in targets])
    b['scale']=np.maximum(1.4826*np.median(abs(a-b['center']),axis=0),floor)
    b['tail']=np.maximum(1,np.quantile(abs((a-b['center'])/b['scale']),.995,axis=0))
    cal=[signals(x,b) for x,_ in roles['threshold_calibration']]
    b['group_scales']={};b['thresholds']={E:float(np.quantile(np.concatenate([s[3][s[0]] for s in cal]),.995))}
    for method,col in [(C,4),(F,5)]:
        vals=np.concatenate([s[col][s[0]] for s in cal])
        scales=np.maximum(1,np.quantile(vals,.995,axis=0))
        b['group_scales'][method]=scales
        b['thresholds'][method]=float(np.quantile(np.max(vals/scales,axis=1),.995))
    assert all(v>0 for v in b['thresholds'].values())
    with (out/'candidate_bundle.pkl').open('wb') as f:pickle.dump(b,f,protocol=5)
    write(out/'calibration.json',dict(decay=b['decay'],thresholds=b['thresholds'],
        group_scales={m:dict(zip(groups,map(float,v))) for m,v in b['group_scales'].items()},
        role_rows={role:sum(len(x) for x,_ in segs) for role,segs in roles.items()},reference_fit_sampled_rows=len(fitframe),
        residual_scaling_valid_rows=len(a),threshold_valid_rows=sum(int(s[0].sum()) for s in cal)))
    return b


def category(c):
    return 'temperature' if 'TI' in c else 'vibration' if 'XI' in c else 'axial' if 'ZI' in c else 'pressure'


def scenes(raw,meta):
    # Reuse pre-existing training scenes; never parse validation/test observation files.
    manifest=read(D2/'scene_manifest.json')
    chosen=[m for m in manifest if m['split']=='train' and m['benchmark']=='main' and
        meta.loc[m['first_source_row'],'source_timestamp']>=pd.Timestamp('2022-05-01') and
        meta.loc[m['last_source_row'],'source_timestamp']<pd.Timestamp('2022-07-01')]
    ids={m['scene_id'] for m in chosen}
    batches=[]
    for chunk in pd.read_csv(D2/'train_observations.csv.gz',chunksize=50000):
        batches.append(chunk.loc[chunk.scene_id.isin(ids)])
    observations=pd.concat(batches,ignore_index=True).set_index('scene_id')
    for m in chosen:
        x=observations.loc[m['scene_id']].sort_values('step').drop(columns='step').reset_index(drop=True)
        yield dict(scene_id=m['scene_id'],source='v2',stratum='main',block_id=m['block_id'],group=m['group'],
            onset=int(m['onset']),endpoint=m['failure_step'] if m['observed'] else None,
            censored=m['censored'],first_source_row=m['first_source_row'],rows=len(x)),x
    cfg=read(WORK/'early_warning_v2/stress_test/config.json')
    original=read(D1/'config_snapshot.json');groups=read(WORK/'RCSD-1YD Data Set/controlled_dataset_v2/config.json')['groups']
    scales=pd.read_csv(D1/'training_scales.csv').set_index(['speed_band','tag'])
    block=0
    for x,m in parts(raw,meta,'2022-05-01','2022-07-01',True):
        for start in range(0,len(x)-cfg['block_points']+1,cfg['block_points']):
            base=x.iloc[start:start+cfg['block_points']].reset_index(drop=True)
            first=int(m.source_row_id.iloc[start]);bid=f'TS{block:02d}';block+=1
            band=int(np.searchsorted(original['speed_band_edges'],base['75SI865R.pv'].median(),side='right')-1)
            yield dict(scene_id=bid+'_replay',source='stress',stratum='replay',block_id=bid,group='none',
                onset=None,endpoint=None,censored=False,first_source_row=first,rows=len(base)),base
            for group,directions in groups.items():
                for amp in cfg['amplitude_multipliers']:
                    for duration in cfg['ramp_hours']:
                        onset=cfg['prelude_points'];end=onset+duration*4;n=end+1+cfg['plateau_hours']*4
                        d=np.clip((np.arange(n)-onset)/(duration*4),0,1)
                        obs=base.iloc[:n].copy();amplitudes={}
                        for c,sign in directions.items():
                            amplitude=sign*amp*float(np.clip(cfg['training_scale_multiplier']*scales.loc[(band,c),'scale'],*original['amplitude_caps'][category(c)]))
                            obs[c]+=amplitude*d;amplitudes[c]=amplitude
                            assert obs[c].between(*original['bounds'][category(c)]).all(),(bid,group,c)
                        yield dict(scene_id=f'{bid}_{group}_{duration}h_amp{amp:g}',source='stress',
                            stratum=f'ramp_{duration}h_amp{amp:g}',block_id=bid,group=group,onset=onset,
                            endpoint=end,censored=False,first_source_row=first,rows=n,amplitudes=amplitudes),obs
    assert block>0,'No long stress backgrounds; cannot advance this pilot'


def event_rows(t,b,m):
    rows=[]
    for method in METHODS:
        a=t[method+'_active'];valid=t.valid
        row=dict(**{k:v for k,v in m.items() if k!='amplitudes'},method=method,valid_points=int(valid.sum()),
            active_points=int((a&valid).sum()),new_events=int(t[method+'_new'].sum()),paired_two_hour=None,paired_first=None,
            paired_lead_hours=None,cusum_unique=None)
        if m['endpoint'] is not None:
            starts=np.flatnonzero(t[method+'_new'].to_numpy())
            starts=starts[(starts>m['onset'])&(starts<=m['endpoint'])]
            paired=[int(k) for k in starts if not b[method+'_active'].iloc[max(0,k-2):k+1].any()]
            row.update(paired_first=paired[0] if paired else None,
                       paired_lead_hours=(m['endpoint']-paired[0])*.25 if paired else None,
                       paired_two_hour=bool(paired and (m['endpoint']-paired[0])*.25>=2))
            if m['source']=='stress':row['late_plateau_active_fraction']=float(a.iloc[-96:].mean())
        rows.append(row)
    return rows


def episode_rows(frame):
    rows=[]
    for segment,g in frame.groupby('segment',sort=False):
        g=g.reset_index(drop=True)
        assert pd.to_datetime(g.timestamp).diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
        for method in METHODS:
            a=(g[method+'_active']&g.valid).to_numpy(bool)
            starts=np.flatnonzero(a&~np.r_[False,a[:-1]])
            ends=np.flatnonzero(a&~np.r_[a[1:],False])
            for i,j in zip(starts,ends):
                rows.append(dict(method=method,segment=int(segment),start=str(g.timestamp.iloc[i]),last_active=str(g.timestamp.iloc[j]),
                    points=int(j-i+1),observed_hours=(j-i+1)*.25,end_reason='boundary' if j==len(g)-1 else 'invalid_reset' if not g.valid.iloc[j+1] else 'valid_inactive'))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results');args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Preserve previous results; use a new directory')
    plan=read(ROOT/'plan.json')
    paths=[ROOT/'plan.json',Path(__file__),D1/'raw_metadata.csv',D1/'raw_observations.csv.gz',D1/'training_scales.csv',
        D1/'config_snapshot.json',D2/'train_observations.csv.gz',D2/'scene_manifest.json',D2/'training_amplitudes.csv',
        WORK/'early_warning_v1/config.json',WORK/'early_warning_v3/config.json',WORK/'early_warning_v2/stress_test/config.json',
        WORK/'RCSD-1YD Data Set/controlled_dataset_v2/config.json',WORK/'final_evaluation_pipeline_v1/release_manifest.json']
    before={os.path.relpath(p,WORK):digest(p) for p in paths}
    out.mkdir(parents=True);write(out/'run_started.json',dict(plan=plan,inputs=before))
    raw,meta=load_raw()
    with threadpool_limits(limits=1):
        bundle=fit(raw,meta,plan,out)
        empirical=[]
        for segment,(x,m) in enumerate(parts(raw,meta,*plan['calendar']['evaluation'],False)):
            r=score(x,bundle);r.insert(0,'source_row_id',m.source_row_id);r.insert(1,'timestamp',m.source_timestamp);r['segment']=segment;empirical.append(r)
        empirical=pd.concat(empirical,ignore_index=True)
        empirical.to_csv(out/'empirical_trace.csv.gz',index=False)
        episodes=episode_rows(empirical);episodes.to_csv(out/'empirical_episodes.csv',index=False)
        erows=[]
        for method in METHODS:
            g=episodes.loc[episodes.method.eq(method)]
            erows.append(dict(method=method,valid_points=int(empirical.valid.sum()),active_points=int(empirical.loc[empirical.valid,method+'_active'].sum()),
                active_fraction=float(empirical.loc[empirical.valid,method+'_active'].mean()),new_events=int(empirical[method+'_new'].sum()),
                longest_observed_hours=float(g.observed_hours.max()) if len(g) else 0))
        empirical_metrics=pd.DataFrame(erows).set_index('method');empirical_metrics.to_csv(out/'empirical_metrics.csv')
        manifests=[];traces=[];controls={};events=[];obs=[]
        for m,x in scenes(raw,meta):
            ids=np.arange(m['first_source_row'],m['first_source_row']+len(x))
            assert meta.loc[ids,'source_timestamp'].between('2022-05-01','2022-06-30 23:59:59').all()
            assert meta.loc[ids,'source_timestamp'].diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
            t=score(x,bundle);key=(int(ids[0]),len(ids))
            if key not in controls:
                control=score(raw.loc[ids].reset_index(drop=True),bundle);control['first_source_row']=key[0];control['scene_rows']=key[1];controls[key]=control
            events.extend(event_rows(t,controls[key],m));t['scene_id']=m['scene_id'];traces.append(t)
            copied=x.copy();copied.insert(0,'step',np.arange(len(x)));copied.insert(0,'scene_id',m['scene_id']);obs.append(copied)
            m['last_source_row']=int(ids[-1]);manifests.append(m)
        pd.concat(traces,ignore_index=True).to_csv(out/'scene_traces.csv.gz',index=False)
        pd.concat(controls.values(),ignore_index=True).to_csv(out/'control_traces.csv.gz',index=False)
        pd.concat(obs,ignore_index=True).to_csv(out/'observations.csv.gz',index=False)
        write(out/'scene_manifest.json',manifests)
        ev=pd.DataFrame(events);ev.to_csv(out/'scene_metrics.csv',index=False)
        complete=ev.loc[ev.endpoint.notna()]
        summary=complete.groupby(['source','stratum','method']).agg(scenes=('scene_id','size'),backgrounds=('block_id','nunique'),hits=('paired_two_hour','sum')).reset_index()
        summary.to_csv(out/'summary.csv',index=False)
        piv=complete.pivot(index=['source','stratum','scene_id'],columns='method',values='paired_two_hour').astype(bool)
        pairs=[]
        for (source,stratum),g in piv.groupby(level=[0,1]):
            unique=g[C]&~g[E]
            pairs.append(dict(source=source,stratum=stratum,scenes=len(g),baseline_combined=int(g[U].sum()),candidate_combined=int(g[V].sum()),
                gained=int((g[V]&~g[U]).sum()),lost=int((g[U]&~g[V]).sum()),baseline_cusum_only=int(unique.sum()),
                baseline_cusum_only_retained=int((unique&g[V]).sum())))
        pairs=pd.DataFrame(pairs);pairs.to_csv(out/'paired_comparison.csv',index=False)
        mainrow=pairs.loc[pairs.source.eq('v2')&pairs.stratum.eq('main')].iloc[0]
        stress=pairs.loc[pairs.source.eq('stress')]
        weak=stress.loc[stress.stratum.isin(['ramp_168h_amp0.5','ramp_288h_amp0.5'])]
        checks=dict(empirical_fraction_lower=bool(empirical_metrics.loc[V,'active_fraction']<empirical_metrics.loc[U,'active_fraction']),
            longest_not_higher=bool(empirical_metrics.loc[V,'longest_observed_hours']<=empirical_metrics.loc[U,'longest_observed_hours']),
            main_hits_not_lower=bool(mainrow.candidate_combined>=mainrow.baseline_combined),
            every_stress_stratum_not_lower=bool(len(stress)==8 and stress.candidate_combined.ge(stress.baseline_combined).all()),
            weak_slow_unique_retained=bool(len(weak)==2 and weak.baseline_cusum_only_retained.eq(weak.baseline_cusum_only).all()))
        write(out/'decision.json',dict(status='ADVANCE_CANDIDATE_ONLY' if all(checks.values()) else 'DO_NOT_ADVANCE',checks=checks,
            meaning='Training-period auxiliary screen, no replacement or independent final-test claim',no_additional_search=True))
        # Preserve group dominance on raw replay without inferring physical faults.
        dominant=[]
        empirical['month']=pd.to_datetime(empirical.timestamp).dt.strftime('%Y-%m')
        for method in [C,F]:
            for (month,group),g in empirical.loc[empirical.valid&empirical[method+'_active']].groupby(['month',method+'_dominant']):
                dominant.append(dict(method=method,month=month,dominant_group=group,active_points=len(g)))
        pd.DataFrame(dominant).to_csv(out/'dominant_group_counts.csv',index=False)
    assert before=={os.path.relpath(p,WORK):digest(p) for p in paths}
    write(out/'run_status.json',dict(status='COMPLETED_PENDING_VERIFICATION',scenes=len(manifests),method_scene_rows=len(ev),
        parsed_observations_before='2022-07-01',input_files_unchanged=True,frozen_models_changed=False))
    print(empirical_metrics.to_string());print(pairs.to_string(index=False));print(read(out/'decision.json'))


if __name__=='__main__': main()
