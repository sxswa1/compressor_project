"""Fixed equal-weight seed ensemble: same labels, 3x training/inference cost.

No data augmentation, model weighting, query tuning or sample removal. Final
target results remain exploratory after prior target inspection.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'02_few_shot_diagnosis'),str(Path(__file__).parent)]
from run_rcsd_optimization import write_json,digest,source_models,half_strength
from models.process_learning import train_process_classifier,process_prototypes,predict_process_classifier,predict_process_prototypes
from pipelines.rcsd_semisynthetic import sha,support_tasks
from common_algorithms.models.process_cnn import ChannelStandardizer,save_process_model

METHODS=['cnn_scratch','transfer','meta_adapted_prototype']


def target(method,x,y,qx,seed,cfg,source,meta):
    encoder=None if method=='cnn_scratch' else source.encoder if method=='transfer' else meta
    model,history=train_process_classifier(x,y,seed=seed,head_seed=seed+100000,steps=cfg['target_steps'],lr=cfg['learning_rate'],encoder=encoder)
    proto=process_prototypes(model.encoder,x,y) if method=='meta_adapted_prototype' else None
    scores=predict_process_prototypes(model.encoder,proto,qx) if proto is not None else predict_process_classifier(model,qx)
    return model,proto,scores,history


def source_check(x,y,g,contract,cfg,out):
    rows=[];roles=[];columns=[c['name'] for c in contract['channels']]
    for fold,limits in enumerate(cfg['source_forward_folds']):
        ordered=np.unique(g);training=ordered[:limits['train_end']];held=ordered[limits['train_end']:limits['validation_end']]
        ti=np.flatnonzero(np.isin(g,training));vi=np.flatnonzero(np.isin(g,held));vg=g[vi];vy=y[vi]
        scaler=ChannelStandardizer().fit(x[ti]);tx=scaler.transform(x[ti]);vx=scaler.transform(x[vi]);weak=half_strength(vx,vy,vg)
        cached={}
        for seed in cfg['final_train_seeds']:
            source,meta,_,history=source_models(tx,y[ti],g[ti],seed,cfg,False);frozen=digest(source),digest(meta)
            for ss in cfg['selection_support_seeds']:
                order=np.random.default_rng(np.random.SeedSequence([fold,ss])).permutation(held);pool=order[:5];query=order[5:];qi=np.flatnonzero(np.isin(vg,query))
                for k in cfg['selection_k']:
                    si=np.flatnonzero(np.isin(vg,pool[:k]));qx=np.concatenate([vx[qi],weak[qi]])
                    if seed==42:roles.append({'fold':fold,'support_seed':ss,'k':k,'training_backgrounds':training.tolist(),'support_backgrounds':pool[:k].tolist(),'query_backgrounds':query.tolist()})
                    for method in METHODS:
                        _,_,scores,h=target(method,vx[si],vy[si],qx,seed,cfg,source,meta)
                        key=ss,k,method;cached.setdefault(key,[]).append(scores)
                        for variant,part in [('main',scores[:len(qi)]),('half_amplitude',scores[len(qi):])]:
                            rows.append({'fold':fold,'support_seed':ss,'k':k,'method':method,'train_seed':seed,'variant':variant,
                                'accuracy':float(accuracy_score(vy[qi],part.argmax(axis=1)))})
            if frozen!=(digest(source),digest(meta)):raise RuntimeError('Source mutation')
        for (ss,k,method),items in cached.items():
            order=np.random.default_rng(np.random.SeedSequence([fold,ss])).permutation(held);qi=np.flatnonzero(np.isin(vg,order[5:]));scores=np.mean(items,axis=0)
            for variant,part in [('main',scores[:len(qi)]),('half_amplitude',scores[len(qi):])]:
                rows.append({'fold':fold,'support_seed':ss,'k':k,'method':method+'_seed_ensemble','train_seed':-1,'variant':variant,
                    'accuracy':float(accuracy_score(vy[qi],part.argmax(axis=1)))})
        pd.DataFrame(rows).to_csv(out/'source_validation_metrics.csv',index=False)
        print(f'ENSEMBLE FORWARD CHECK fold={fold+1}/3 complete',flush=True)
    pd.DataFrame(rows).groupby('method').accuracy.mean().to_csv(out/'source_validation_summary.csv')
    write_json(out/'source_validation_roles.json',roles)


def evaluate(x,meta,contract,cfg,out,prepared):
    basecfg=json.loads((prepared/'preparation.json').read_text())['config'];source,tasks=support_tasks(meta[meta.variant.isin(basecfg['variants'])],basecfg)
    y=meta.class_id.to_numpy(dtype=np.int64);g=meta.block_id.to_numpy();qi=np.flatnonzero(meta.split.eq('query'));scaler=ChannelStandardizer().fit(x[source]);sx=scaler.transform(x)
    variants={v:meta.index[meta.split.eq('query')&meta.variant.eq(v)].to_numpy() for v in meta.loc[qi,'variant'].unique()};classes=basecfg['class_names']
    metrics=[];predictions=[];expected=[];resources=[];cached={};members={};logs=[]
    write_json(out/'evaluation_protocol.json',{'config':cfg,'source_ids':source.tolist(),'tasks':tasks,'query_ids_by_variant':{k:v.tolist() for k,v in variants.items()},
        'ensemble_weights':[1/3]*3,'source_original_labels':124,'target_labels_per_member':'same 4*K original windows',
        'target_evaluation':'exploratory_after_v1','query_fitting_rows':0,'additional_real_fault_events':0})
    def record(method,scores,k,ss,seed,support):
        identity={'method':method,'k':k,'support_seed':ss,'train_seed':seed};pred=scores.argmax(axis=1)
        for variant,ids in variants.items():
            pos=np.flatnonzero(np.isin(qi,ids));truth=y[qi[pos]];p=pred[pos];mat=confusion_matrix(truth,p,labels=range(4));recall=np.diag(mat)/mat.sum(axis=1)
            metrics.append({**identity,'variant':variant,'accuracy':float(accuracy_score(truth,p)),'balanced_accuracy':float(recall.mean()),
                'macro_f1':float(f1_score(truth,p,average='macro',zero_division=0)),**{f'class_{c}_recall':float(v) for c,v in enumerate(recall)},
                'confusion_matrix':json.dumps(mat.tolist()),'original_target_labels':len(support),'query_windows':len(pos),'query_backgrounds':22,'members':3 if method.endswith('seed_ensemble') else 1})
        for i,p,score in zip(qi,pred,scores.max(axis=1)):predictions.append({**identity,'variant':meta.loc[i,'variant'],'sample_id':int(i),
            'window_id':meta.loc[i,'window_id'],'block_id':g[i],'true_class':int(y[i]),'predicted_class':int(p),'model_score':float(score)})
        return pred
    def export_case(key,method,k,path,operator,pred,scores):
        resources.append({'id':key,'method':method,'k':k,'model':path.relative_to(out).as_posix(),'operator':operator,'sha256':sha(path)})
        for i,p,score in zip(qi,pred,scores.max(axis=1)):expected.append({'case':key,'window_id':meta.loc[i,'window_id'],'predicted_label':classes[p],'model_score':float(score)})
    for seed in cfg['final_train_seeds']:
        classifier,encoder,_,history=source_models(sx[source],y[source],g[source],seed,cfg,False);frozen=digest(classifier),digest(encoder);log={'seed':seed,'source':history,'target':[]}
        for task in tasks:
            support=np.asarray(task['support_ids']);k=task['k'];ss=task['support_seed']
            for method in METHODS:
                model,proto,scores,h=target(method,sx[support],y[support],sx[qi],seed,cfg,classifier,encoder)
                pred=record(method,scores,k,ss,seed,support);cached.setdefault((k,ss,method),[]).append(scores)
                log['target'].append({'method':method,'k':k,'support_seed':ss,'history':h})
                if ss==11:
                    key=f'k{k}_{method}_seed{seed}';path=out/'resources'/(key+'.pt');path.parent.mkdir(exist_ok=True)
                    kind='prototype' if proto is not None else 'transfer' if method=='transfer' else 'cnn'
                    save_process_model(path,model.encoder if proto is not None else model,scaler,contract,classes,kind,proto,
                        {'semi_synthetic':True,'method':method,'train_seed':seed,'support_ids':support.tolist(),'natural_fault_incidents':0,'target_evaluation':'exploratory'})
                    members.setdefault((k,method),[]).append(torch.load(path,weights_only=True,map_location='cpu'))
                    operator='process_meta_learning.py' if proto is not None else 'process_transfer_learning.py' if method=='transfer' else 'process_cnn_diagnosis.py'
                    export_case(key,method,k,path,operator,pred,scores)
            pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
            print(f'ENSEMBLE MEMBERS seed={seed} K={k} support={ss} complete',flush=True)
        if frozen!=(digest(classifier),digest(encoder)):raise RuntimeError('Source mutation')
        logs.append(log);write_json(out/'training_logs.json',logs)
    for (k,ss,method),items in cached.items():
        scores=np.mean([a.astype(float) for a in items],axis=0);support=np.asarray(next(t['support_ids'] for t in tasks if t['k']==k and t['support_seed']==ss))
        name=method+'_seed_ensemble';pred=record(name,scores,k,ss,-1,support)
        if ss==11:
            key=f'k{k}_{name}';path=out/'resources'/(key+'.pt')
            torch.save({'kind':'process_seed_ensemble','format_version':1,'contract':contract,'class_names':classes,'members':members[k,method],
                'train_seeds':cfg['final_train_seeds'],'provenance':{'semi_synthetic':True,'same_support_ids':support.tolist(),'natural_fault_incidents':0,
                    'target_evaluation':'exploratory','equal_weights_fixed_before_target_scoring':True}},path)
            export_case(key,name,k,path,'process_seed_ensemble.py',pred,scores)
    table=pd.DataFrame(metrics);table.to_csv(out/'metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    table.groupby(['variant','k','method']).agg(accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),
        macro_f1_mean=('macro_f1','mean'),background_recall=('class_0_recall','mean'),temperature_recall=('class_1_recall','mean'),
        oil_temperature_recall=('class_2_recall','mean'),vibration_axial_recall=('class_3_recall','mean'),n=('accuracy','size')).reset_index().to_csv(out/'aggregate_metrics.csv',index=False)
    pd.DataFrame(expected).to_csv(out/'expected_resource_predictions.csv',index=False);write_json(out/'resource_manifest.json',{'cases':resources})
    meta.to_csv(out/'sample_manifest.csv',index=False);np.savez_compressed(out/'all_windows.npz',x=x)
    if len(table)!=432 or len(predictions)!=38016 or len(resources)!=36:raise RuntimeError('Ensemble study incomplete')
    write_json(out/'completion.json',{'metric_rows':len(table),'prediction_rows':len(predictions),'resources':len(resources),'natural_fault_incidents':0,
        'semi_synthetic':True,'query_backgrounds':22,'independent_target_confirmation':False,'remote_pushed':False,'platform_registered':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'02_few_shot_diagnosis/outputs/rcsd_seed_ensemble_v2');args=p.parse_args();out=args.output_dir
    if out.exists():raise FileExistsError('Preserve prior evidence; choose new output directory')
    prepared=ROOT/'data/RCSD_semisynthetic_diagnosis_v1';initial=ROOT/'02_few_shot_diagnosis/outputs/rcsd_optimization_v2'
    cfg=json.loads((ROOT/'02_few_shot_diagnosis/configs/rcsd_support_optimization_v2.json').read_text(encoding='utf-8'))
    cfg.update(version='rcsd-equal-seed-ensemble-v2',augmentation_levels=[1.],ensemble_weights=[1/3]*3,
        default_delivery={'method':'meta_adapted_prototype_seed_ensemble','k':5,'support_seed':11},
        runner_sha256=sha(__file__),helper_sha256=sha(ROOT/'02_few_shot_diagnosis/scripts/run_rcsd_optimization.py'))
    for name,checksum in json.loads((prepared/'checksums.json').read_text()).items():
        if sha(prepared/name)!=checksum:raise ValueError('Original prepared evidence changed')
    x=np.load(prepared/'windows.npz',allow_pickle=False)['x'];meta=pd.read_csv(prepared/'sample_manifest.csv');source=np.flatnonzero(meta.split.eq('source'))
    contract=json.loads((prepared/'preparation.json').read_text())['input_contract'];out.mkdir(parents=True);write_json(out/'plan_locked.json',cfg);torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        source_check(x[source],meta.loc[source,'class_id'].to_numpy(),meta.loc[source,'block_id'].to_numpy(),contract,cfg,out)
        x=np.load(initial/'all_windows.npz',allow_pickle=False)['x'];meta=pd.read_csv(initial/'sample_manifest.csv')
        evaluate(x,meta,contract,cfg,out,prepared)
    if sha(__file__)!=cfg['runner_sha256'] or sha(ROOT/'02_few_shot_diagnosis/scripts/run_rcsd_optimization.py')!=cfg['helper_sha256']:raise RuntimeError('Frozen runner changed')
    print((out/'completion.json').read_text(),flush=True)


if __name__=='__main__':main()
