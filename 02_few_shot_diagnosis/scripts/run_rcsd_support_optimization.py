"""Forward source selection of paired-support augmentation and full ablations.

Uses only public semi-synthetic labels. Same original background stays in one
role; no target query controls are supplied to augmentation, fitting or inference.
All post-v1 target findings are explicitly exploratory, including new stresses.
"""
from pathlib import Path
import argparse,json,sys
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'02_few_shot_diagnosis'),str(Path(__file__).parent)]
from run_rcsd_optimization import write_json,digest,half_strength,source_models
from models.rcsd_optimization import augment_source_pairs
from models.process_learning import train_process_classifier,process_prototypes,predict_process_classifier,predict_process_prototypes
from pipelines.rcsd_semisynthetic import sha,support_tasks
from common_algorithms.models.process_cnn import ChannelStandardizer,save_process_model
from common_algorithms.models.random_forest import train_random_forest
from common_algorithms.feature_extraction.process_features import process_features

METHODS=['random_forest','cnn','transfer','meta']


def train_target(method,x,y,groups,levels,seed,cfg,source_classifier,source_meta):
    ax,ay,ag=augment_source_pairs(x,y,groups,levels)
    if method=='random_forest':return None,None,(ax,ay),[]
    initial=source_classifier.encoder if method=='transfer' else source_meta if method=='meta' else None
    model,history=train_process_classifier(ax,ay,seed=seed,head_seed=seed+100000,steps=cfg['target_steps'],
        lr=cfg['learning_rate'],encoder=initial)
    proto=process_prototypes(model.encoder,ax,ay) if method=='meta' else None
    return model,proto,(ax,ay),history


def scores_target(method,model,proto,aug,qx,columns,interval,cfg):
    if method=='random_forest':
        sf,names=process_features(aug[0],columns,interval);qf,_=process_features(qx,columns,interval)
        forest=train_random_forest(sf,aug[1],n_estimators=cfg['forest_trees'],random_state=42)
        return forest.predict_proba(qf),forest,names
    if method=='meta':return predict_process_prototypes(model.encoder,proto,qx),None,None
    return predict_process_classifier(model,qx),None,None


def selection(x,y,g,contract,cfg,out):
    rows=[];roles=[];logs=[];columns=[c['name'] for c in contract['channels']];interval=contract['interval_seconds'];ordered=np.unique(g)
    for fold,limits in enumerate(cfg['source_forward_folds']):
        training=ordered[:limits['train_end']];held=ordered[limits['train_end']:limits['validation_end']]
        ti=np.flatnonzero(np.isin(g,training));vi=np.flatnonzero(np.isin(g,held))
        scaler=ChannelStandardizer().fit(x[ti]);tx=scaler.transform(x[ti]);vx=scaler.transform(x[vi]);vy=y[vi];vg=g[vi]
        weak=half_strength(vx,vy,vg)
        for seed in cfg['selection_train_seeds']:
            classifier,meta,_,history=source_models(tx,y[ti],g[ti],seed,cfg,False)
            logs.append({'fold':fold,'train_seed':seed,**history});frozen=digest(classifier),digest(meta)
            for ss in cfg['selection_support_seeds']:
                order=np.random.default_rng(np.random.SeedSequence([fold,ss])).permutation(held);pool=order[:5];query=order[5:]
                qi=np.flatnonzero(np.isin(vg,query))
                for k in cfg['selection_k']:
                    si=np.flatnonzero(np.isin(vg,pool[:k]));identity={'fold':fold,'train_seed':seed,'support_seed':ss,'k':k}
                    roles.append({**identity,'training_backgrounds':training.tolist(),'support_backgrounds':pool[:k].tolist(),'query_backgrounds':query.tolist()})
                    for method in METHODS:
                        if method=='random_forest' and seed!=cfg['selection_train_seeds'][0]:continue
                        for candidate,levels in cfg['augmentation_candidates'].items():
                            model,proto,aug,history=train_target(method,vx[si],vy[si],vg[si],levels,seed,cfg,classifier,meta)
                            for variant,qx in [('main',vx),('half_amplitude',weak)]:
                                scores,_,_=scores_target(method,model,proto,aug,qx[qi],columns,interval,cfg)
                                rows.append({**identity,'method':method,'candidate':candidate,'variant':variant,
                                    'balanced_accuracy':float(accuracy_score(vy[qi],scores.argmax(axis=1)))})
                    pd.DataFrame(rows).to_csv(out/'source_selection_metrics.csv',index=False)
            if frozen!=(digest(classifier),digest(meta)):raise RuntimeError('Source encoder mutation')
            print(f'FORWARD SOURCE fold={fold+1}/3 seed={seed} complete',flush=True)
    means=pd.DataFrame(rows).groupby(['method','candidate'],sort=False).balanced_accuracy.mean().reset_index()
    means.to_csv(out/'source_selection_summary.csv',index=False)
    choices={method:str(part.sort_values('balanced_accuracy',ascending=False,kind='stable').iloc[0].candidate) for method,part in means.groupby('method',sort=False)}
    result={'choices':choices,'selection_data':'Only January-June public source windows; expanding chronological source training',
        'target_query_used':False,'fresh_encoders_each_fold':True,'fold_roles':roles,'metric_rows':len(rows),
        'config_sha256':sha(ROOT/'02_few_shot_diagnosis/configs/rcsd_support_optimization_v2.json'),
        'runner_sha256':sha(__file__),'augmentation_code_sha256':sha(ROOT/'02_few_shot_diagnosis/models/rcsd_optimization.py'),
        'objective':cfg['source_selection_objective'],'target_confirmation_is_independent':False}
    write_json(out/'selection_locked.json',result);write_json(out/'source_selection_logs.json',logs);return result


def evaluate(x,meta,contract,cfg,choices,out,prepared):
    basecfg=json.loads((prepared/'preparation.json').read_text())['config'];source,tasks=support_tasks(meta[meta.variant.isin(basecfg['variants'])],basecfg)
    labels=meta.class_id.to_numpy(dtype=np.int64);g=meta.block_id.to_numpy();qi=np.flatnonzero(meta.split.eq('query'))
    columns=[c['name'] for c in contract['channels']];interval=contract['interval_seconds'];classes=basecfg['class_names']
    scaler=ChannelStandardizer().fit(x[source]);sx=scaler.transform(x);metrics=[];predictions=[];resources=[];expected=[];logs=[]
    variants={v:meta.index[meta.split.eq('query')&meta.variant.eq(v)].to_numpy() for v in meta.loc[qi,'variant'].unique()}
    write_json(out/'evaluation_protocol.json',{'config':cfg,'choices':choices,'selection_sha256':sha(out/'selection_locked.json'),
        'source_ids':source.tolist(),'tasks':tasks,'query_ids_by_variant':{k:v.tolist() for k,v in variants.items()},
        'target_evaluation':'exploratory_after_v1_and_unsuccessful_source_only_trial','paired_support_requirement':cfg['assumption'],
        'query_normal_reference_used':False,'default_delivery_fixed_before_scoring':cfg['delivery_choice']})
    def record(method,scores,k,ss,seed,support,model,proto,forest,names,levels,base):
        identity={'method':method,'k':k,'support_seed':ss,'train_seed':seed};pred=scores.argmax(axis=1)
        for variant,ids in variants.items():
            pos=np.flatnonzero(np.isin(qi,ids));truth=labels[qi[pos]];p=pred[pos];matrix=confusion_matrix(truth,p,labels=range(4));recall=np.diag(matrix)/matrix.sum(axis=1)
            metrics.append({**identity,'variant':variant,'accuracy':float(accuracy_score(truth,p)),'macro_f1':float(f1_score(truth,p,average='macro',zero_division=0)),
                'balanced_accuracy':float(recall.mean()),**{f'class_{c}_recall':float(v) for c,v in enumerate(recall)},
                'confusion_matrix':json.dumps(matrix.tolist()),'original_support_labels':len(support),'derived_training_rows':k*(1+3*len(levels)),
                'query_windows':len(pos),'query_backgrounds':22,'augmentation_levels':str(levels)})
        for i,p,score in zip(qi,pred,scores.max(axis=1)):predictions.append({**identity,'variant':meta.loc[i,'variant'],'sample_id':int(i),
            'window_id':meta.loc[i,'window_id'],'block_id':g[i],'true_class':int(labels[i]),'predicted_class':int(p),'model_score':float(score)})
        if seed in {42,-1} and ss==11:
            key=f'k{k}_{method}';path=out/'resources'/(key+('.joblib' if forest is not None else '.pt'));path.parent.mkdir(exist_ok=True)
            provenance={'semi_synthetic':True,'natural_fault_incidents':0,'method':method,'original_target_support_ids':support.tolist(),
                'derived_target_rows':k*(1+3*len(levels)),'source_backgrounds':31,'source_original_label_windows':124,'augmentation_levels':levels,
                'support_pairing_required_at_training':True,'normal_pair_required_at_inference':False,
                'target_evaluation':'exploratory_after_v1','selection_sha256':sha(out/'selection_locked.json')}
            if forest is not None:
                joblib.dump({'model':forest,'feature_columns':names,'scaler':scaler.state(),'process_contract':contract,'class_names':classes,'provenance':provenance},path)
                operator='process_random_forest.py'
            else:
                kind='prototype' if base=='meta' else 'transfer' if base=='transfer' else 'cnn'
                save_process_model(path,model.encoder if base=='meta' else model,scaler,contract,classes,kind,proto,provenance)
                operator='process_meta_learning.py' if base=='meta' else 'process_transfer_learning.py' if base=='transfer' else 'process_cnn_diagnosis.py'
            resources.append({'id':key,'method':method,'k':k,'model':path.relative_to(out).as_posix(),'operator':operator,'sha256':sha(path)})
            for i,p,score in zip(qi,pred,scores.max(axis=1)):expected.append({'case':key,'window_id':meta.loc[i,'window_id'],'predicted_label':classes[p],'model_score':float(score)})
    for seed in cfg['final_train_seeds']:
        classifier,encoder,_,history=source_models(sx[source],labels[source],g[source],seed,cfg,False);frozen=digest(classifier),digest(encoder)
        log={'seed':seed,'source_training':history,'target':[]}
        for task in tasks:
            support=np.asarray(task['support_ids']);ss=task['support_seed'];k=task['k']
            for base in METHODS:
                if base=='random_forest' and seed!=42:continue
                # Include explicit no-augmentation ablations under the same implementation.
                for variant,levels in [('support_augmented',cfg['augmentation_candidates'][choices[base]]),('original_support',[1.])]:
                    model,proto,aug,h=train_target(base,sx[support],labels[support],g[support],levels,seed,cfg,classifier,encoder)
                    scores,forest,names=scores_target(base,model,proto,aug,sx[qi],columns,interval,cfg)
                    method=base+'_'+variant
                    record(method,scores,k,ss,-1 if base=='random_forest' else seed,support,model,proto,forest,names,levels,base)
                    log['target'].append({'method':method,'k':k,'support_seed':ss,'history':h})
            if frozen!=(digest(classifier),digest(encoder)):raise RuntimeError('Source changed after target adaptation')
            pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
            print(f'SUPPORT EXPLORATORY seed={seed} K={k} support={ss} complete',flush=True)
        logs.append(log);write_json(out/'training_logs.json',logs)
    table=pd.DataFrame(metrics)
    table.groupby(['variant','k','method']).agg(accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
        accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),macro_f1_mean=('macro_f1','mean'),
        background_recall=('class_0_recall','mean'),temperature_recall=('class_1_recall','mean'),oil_temperature_recall=('class_2_recall','mean'),
        vibration_axial_recall=('class_3_recall','mean'),n=('accuracy','size')).reset_index().to_csv(out/'aggregate_metrics.csv',index=False)
    pd.DataFrame(expected).to_csv(out/'expected_resource_predictions.csv',index=False);write_json(out/'resource_manifest.json',{'cases':resources})
    meta.to_csv(out/'sample_manifest.csv',index=False);np.savez_compressed(out/'all_windows.npz',x=x)
    for variant,ids in variants.items():
        pieces=[]
        for i in ids:
            frame=pd.DataFrame(x[i].T,columns=columns);frame.insert(0,'timestamp',pd.date_range(meta.loc[i,'start_time'],periods=20,freq='15min'))
            frame.insert(0,'window_id',meta.loc[i,'window_id']);pieces.append(frame)
        pd.concat(pieces,ignore_index=True).to_csv(out/f'query_{variant}.csv',index=False)
    if len(table)!=720 or len(predictions)!=63360 or len(resources)!=24:raise RuntimeError('Full comparison incomplete')
    write_json(out/'completion.json',{'metric_rows':len(table),'prediction_rows':len(predictions),'resources_exported':len(resources),
        'source_backgrounds':31,'support_pool_backgrounds':11,'query_backgrounds':22,'natural_fault_incidents':0,'semi_synthetic':True,
        'target_results':'exploratory','query_pairing_at_inference':False,'platform_registered':False,'remote_pushed':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'02_few_shot_diagnosis/outputs/rcsd_support_optimization_v2');args=parser.parse_args()
    out=args.output_dir
    if out.exists():raise FileExistsError('Preserve completed/partial evidence; choose new output directory')
    prepared=ROOT/'data/RCSD_semisynthetic_diagnosis_v1';previous=ROOT/'02_few_shot_diagnosis/outputs/rcsd_optimization_v2'
    if not (previous/'completion.json').exists():raise ValueError('Preserve and finish the initial optimization trial first')
    cfg=json.loads((ROOT/'02_few_shot_diagnosis/configs/rcsd_support_optimization_v2.json').read_text(encoding='utf-8'))
    for name,checksum in json.loads((prepared/'checksums.json').read_text()).items():
        if sha(prepared/name)!=checksum:raise ValueError('Prepared v1 input changed')
    original=np.load(prepared/'windows.npz',allow_pickle=False)['x'];manifest=pd.read_csv(prepared/'sample_manifest.csv')
    contract=json.loads((prepared/'preparation.json').read_text())['input_contract'];source=np.flatnonzero(manifest.split.eq('source'))
    out.mkdir(parents=True);write_json(out/'optimization_plan.json',cfg);torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        locked=selection(original[source],manifest.loc[source,'class_id'].to_numpy(),manifest.loc[source,'block_id'].to_numpy(),contract,cfg,out)
        x=np.load(previous/'all_windows.npz',allow_pickle=False)['x'];meta=pd.read_csv(previous/'sample_manifest.csv')
        if any(sha(ROOT/path)!=checksum for path,checksum in []):raise RuntimeError('Unreachable')
        evaluate(x,meta,contract,cfg,locked['choices'],out,prepared)
    print((out/'completion.json').read_text(),flush=True)


if __name__=='__main__':main()
