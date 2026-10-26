"""One fixed 125-vs-50-feature DI ablation, using Jan-Jun files only."""
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
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]/'rcsd_v1'
DATA=WORK/'RCSD-1YD Data Set/controlled_dataset_v2/data'
sys.path.insert(0,str(WORK.parents[1]/'pipelines'))
from baseline_data import windows,tabular_features,risk_levels
COLS=np.r_[50:75,100:125]
METHODS=['history125','changes50','constant']

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load():
    sensors=read(DATA/'model_input_columns.json');assert len(sensors)==25
    manifest=pd.DataFrame(read(DATA/'scene_manifest.json')).set_index('scene_id')
    manifest=manifest.loc[manifest.split.eq('train')&manifest.benchmark.eq('main')]
    tables=[pd.read_csv(DATA/f'train_{name}.csv.gz').set_index(['scene_id','step']) for name in ['observations','labels','lineage']]
    x,y,lineage=tables
    assert x.index.equals(y.index) and x.index.equals(lineage.index) and x.index.is_unique
    assert lineage.split.eq('train').all()
    assert pd.to_datetime(lineage.source_timestamp).lt('2022-07-01').all()
    assert list(x)==sensors
    return sensors,manifest,x,y,lineage

def build(sensors,manifest,x,y,lineage):
    features=[];keys=[];support=[]
    for sid,m in manifest.iterrows():
        obs,lab,lin=x.loc[sid],y.loc[sid],lineage.loc[sid]
        assert lin.block_id.eq(m.block_id).all() and np.array_equal(obs.index,np.arange(m.n))
        vals=obs[sensors].to_numpy(dtype=np.float32).copy()
        vals[~obs['75SI865R.pv'].between(6050,8500).to_numpy()]=np.nan
        X,ends=windows(vals,lin.source_timestamp,96,15)
        keep=lab.iloc[ends].di_label_valid.to_numpy(bool)&np.isfinite(lab.iloc[ends].DI_ref)
        X,ends=X[keep],ends[keep]
        assert len(X)>0,sid
        f=tabular_features(X);assert f.shape[1]==125
        key=lin.iloc[ends].reset_index().copy();key.insert(0,'scene_id',sid)
        key['DI_ref']=lab.iloc[ends].DI_ref.to_numpy();key['onset']=int(m.onset)
        for c in ['nominal_hours','curve_exponent','group','censored']:key[c]=m[c]
        key['sampled_for_fit']=np.arange(len(key))%4==0
        assert key.DI_ref.between(0,1).all()
        features.append(f);keys.append(key)
        support.append(dict(scene_id=sid,rows=len(obs),valid_label_windows=len(ends),fit_sampled_windows=int(key.sampled_for_fit.sum()),
            first_step=int(ends[0]),last_step=int(ends[-1]),block_id=m.block_id))
    return np.concatenate(features),pd.concat(keys,ignore_index=True),pd.DataFrame(support)

def metrics(p):
    error=p.prediction-p.DI_ref
    truth=risk_levels(p.DI_ref);pred=risk_levels(p.prediction)
    r=dict(points=len(p),scenes=int(p.scene_id.nunique()),backgrounds=int(p.block_id.nunique()),
        point_mae=float(abs(error).mean()),bias=float(error.mean()),
        scene_mae=float(abs(error).groupby(p.scene_id).mean().mean()),
        background_mae=float(abs(error).groupby(p.block_id).mean().mean()),
        risk_accuracy=float(np.mean(truth==pred)),risk_macro_f1=float(f1_score(truth,pred,labels=[1,2,3,4],average='macro',zero_division=0)),
        risk_overestimate=float(np.mean(pred>truth)),risk_underestimate=float(np.mean(pred<truth)),
        below_zero=float(p.raw_prediction.lt(0).mean()),above_one=float(p.raw_prediction.gt(1).mean()))
    for k in range(1,5):
        r[f'support_{k}']=int((truth==k).sum())
        r[f'recall_{k}']=float(np.mean(pred[truth==k]==k)) if np.any(truth==k) else None
    return r

def summarize(preds):
    aggregate=[];strata=[];confusion=[];pre=[]
    for (fold,model),p in preds.groupby(['fold','model'],sort=False):
        aggregate.append(dict(fold=int(fold),model=model,**metrics(p)))
        for name in ['nominal_hours','group','curve_exponent','block_id']:
            for value,g in p.groupby(name,sort=False):strata.append(dict(fold=int(fold),model=model,stratum=name,value=str(value),**metrics(g)))
        truth=risk_levels(p.DI_ref);pr=risk_levels(p.prediction)
        for a in range(1,5):
            for b in range(1,5):confusion.append(dict(fold=int(fold),model=model,reference=a,predicted=b,count=int(((truth==a)&(pr==b)).sum())))
        g=p.loc[p.step.le(p.onset)].copy();assert g.DI_ref.eq(1).all()
        assert g.groupby('source_row_id').raw_prediction.agg(lambda x:x.max()-x.min()).max()<1e-6
        g=g.drop_duplicates('source_row_id')
        pre.append(g)
    return pd.DataFrame(aggregate),pd.DataFrame(strata),pd.DataFrame(confusion),pd.concat(pre,ignore_index=True)

def decide(aggregate,strata,pre,preds):
    a=aggregate.loc[aggregate.model.eq('history125')].set_index('fold')
    b=aggregate.loc[aggregate.model.eq('changes50')].set_index('fold')
    pooled={m:metrics(preds.loc[preds.model.eq(m)]) for m in METHODS}
    slow=strata.loc[strata.stratum.eq('nominal_hours')&pd.to_numeric(strata.value,errors='coerce').isin([48,72])].pivot(index=['fold','value'],columns='model',values='scene_mae')
    healthy=pre.groupby(['fold','model']).apply(lambda p:abs(float((p.prediction-p.DI_ref).mean())),include_groups=False).unstack()
    checks=dict(mean_scene_mae_lower_every_fold_not_higher=bool(b.scene_mae.mean()<a.scene_mae.mean() and b.scene_mae.le(a.scene_mae+1e-12).all()),
        pooled_each_class_recall_preserved=bool(all(pooled['changes50'][f'recall_{k}'] is not None and pooled['changes50'][f'recall_{k}']>=pooled['history125'][f'recall_{k}']-.01 for k in range(1,5))),
        every_fold_high_risk_recall_preserved=bool(b.recall_4.ge(a.recall_4-.01).all()),
        every_slow_duration_fold_mae_not_higher=bool(len(slow)==6 and slow.changes50.le(slow.history125+1e-12).all()),
        pre_injection_absolute_mean_bias_not_higher=bool(healthy.changes50.mean()<=healthy.history125.mean()+1e-12))
    return dict(status='ADVANCE_CANDIDATE_ONLY' if all(checks.values()) else 'DO_NOT_ADVANCE',checks=checks,
        no_additional_search=True,meaning='Training-side mechanism screen, not a publication acceptance target'),pooled

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results');args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Preserve prior outputs; use a new output directory')
    plan=read(ROOT/'plan.json')
    paths=[ROOT/'plan.json',Path(__file__),WORK.parents[1]/'pipelines/baseline_data.py',
        DATA/'model_input_columns.json',DATA/'scene_manifest.json',DATA/'training_amplitudes.csv',DATA.parent/'config.json',
        WORK/'paper_baseline_comparison_v1/plan.json',WORK/'final_evaluation_pipeline_v1/release_manifest.json']
    paths += [DATA/f'train_{n}.csv.gz' for n in ['observations','labels','lineage']]
    before={os.path.relpath(p,WORK):digest(p) for p in paths}
    out.mkdir(parents=True);(out/'models').mkdir()
    write(out/'run_started.json',dict(plan=plan,inputs=before))
    sensors,manifest,obs,lab,lin=load();F,keys,support=build(sensors,manifest,obs,lab,lin)
    keys.to_csv(out/'all_main_keys.csv.gz',index=False);support.to_csv(out/'scene_support.csv',index=False)
    np.save(out/'features125.npy',F)
    order=manifest.groupby('block_id').first_source_row.min().sort_values().index.tolist();assert len(order)==31
    predictions=[];splits=[];probes=[];fitkeys=[]
    with threadpool_limits(limits=1):
        for fold,(a,z) in enumerate(plan['fold_block_ranges']):
            fb,eb=order[:a],order[a:z]
            train=keys.block_id.isin(fb)&keys.sampled_for_fit;ev=keys.block_id.isin(eb)
            assert not set(fb)&set(eb)
            assert manifest.loc[manifest.block_id.isin(fb),'last_source_row'].max()<manifest.loc[manifest.block_id.isin(eb),'first_source_row'].min()
            trainkeys=keys.loc[train].copy();trainkeys['fold']=fold;fitkeys.append(trainkeys)
            k=keys.loc[ev].copy();k['fold']=fold
            splits.append(dict(fold=fold,fit_blocks=fb,evaluation_blocks=eb,fit_points=int(train.sum()),evaluation_points=int(ev.sum()),
                fit_first_time=str(trainkeys.source_timestamp.min()),fit_last_time=str(trainkeys.source_timestamp.max()),
                evaluation_first_time=str(k.source_timestamp.min()),evaluation_last_time=str(k.source_timestamp.max())))
            for method in METHODS:
                if method=='constant':
                    model=dict(value=float(np.median(keys.loc[train,'DI_ref'])))
                    prediction=np.full(ev.sum(),model['value'])
                else:
                    cols=np.arange(125) if method=='history125' else COLS
                    model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(F[train][:,cols],keys.loc[train,'DI_ref'])
                    prediction=model.predict(F[ev][:,cols])
                with (out/'models'/f'fold{fold}_{method}.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
                p=k.copy();p['model']=method;p['raw_prediction']=prediction;p['prediction']=np.clip(prediction,0,1)
                p['risk_prediction']=risk_levels(p.prediction);predictions.append(p)
                if method!='constant':
                    # An algebraic constant-level probe, not a scored fault scenario.
                    first=k.iloc[0];v=obs.loc[first.scene_id].iloc[int(first.step)-95:int(first.step)+1][sensors].to_numpy(dtype=float)[None]
                    shift=F[train,:25].std(axis=0).astype(float)
                    normal=tabular_features(v)[:,cols];shifted=tabular_features(v+shift)[:,cols]
                    probes.append(dict(fold=fold,model=method,scene_id=first.scene_id,step=int(first.step),
                        max_feature_change=float(abs(shifted-normal).max()),raw_prediction_change=float((model.predict(shifted)-model.predict(normal))[0])))
            print('Completed fold',fold,'fit',int(train.sum()),'evaluation',int(ev.sum()),flush=True)
    preds=pd.concat(predictions,ignore_index=True);preds.to_csv(out/'predictions.csv.gz',index=False)
    pd.concat(fitkeys,ignore_index=True).to_csv(out/'fit_keys.csv.gz',index=False)
    write(out/'folds.json',splits);pd.DataFrame(probes).to_csv(out/'constant_offset_probe.csv',index=False)
    aggregate,strata,confusion,pre=summarize(preds)
    aggregate.to_csv(out/'fold_metrics.csv',index=False);strata.to_csv(out/'stratified_metrics.csv',index=False)
    confusion.to_csv(out/'risk_confusion.csv',index=False);pre.to_csv(out/'pre_injection_unique.csv.gz',index=False)
    pr=[]
    for (fold,model),g in pre.groupby(['fold','model']):
        pr.append(dict(fold=int(fold),model=model,unique_source_points=len(g),backgrounds=int(g.block_id.nunique()),
            mean_prediction=float(g.prediction.mean()),bias=float((g.prediction-g.DI_ref).mean()),reference_I_recall=float(g.risk_prediction.eq(1).mean())))
    pd.DataFrame(pr).to_csv(out/'pre_injection_summary.csv',index=False)
    decision,pooled=decide(aggregate,strata,pre,preds)
    write(out/'decision.json',decision);write(out/'pooled_metrics.json',pooled)
    for path,h in before.items():assert digest(WORK/path)==h,path
    write(out/'run_status.json',dict(status='COMPLETED_PENDING_VERIFICATION',source_main_scenes=len(manifest),
        evaluation_scenes=int(preds.scene_id.nunique()),evaluation_backgrounds=int(preds.block_id.nunique()),
        predictions=len(preds),models=9,input_files_unchanged=True,observations_read='train only'))
    print(aggregate.to_string(index=False));print(decision)

if __name__=='__main__':main()
