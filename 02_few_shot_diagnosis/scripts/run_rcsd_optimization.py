"""Source-only selection followed by explicitly exploratory target evaluation.

The v1 target has already been inspected. This run does not relabel that target
as an untouched holdout. All source folds train fresh encoders from scratch.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'02_few_shot_diagnosis')]
from pipelines.rcsd_semisynthetic import sha,support_tasks,construct_scene,sample_parameters
from models.process_learning import (train_process_classifier,process_prototypes,
    predict_process_classifier,predict_process_prototypes)
from models.global_record_meta import train_global_record_protonet
from models.rcsd_optimization import (augment_source_pairs,raw_representation,
    fit_source_geometry,target_prototypes,geometry_scores,preserve_head_adaptation)
from common_algorithms.models.process_cnn import ChannelStandardizer,save_process_model


def write_json(path,item):
    Path(path).write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):h.update(key.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def source_models(x,y,g,seed,cfg,augmented=True):
    ax,ay,ag=augment_source_pairs(x,y,g,cfg['source_augmentation_levels']) if augmented else (x,y,g)
    model,h1=train_process_classifier(ax,ay,seed=seed,steps=cfg['source_steps'],lr=cfg['learning_rate'])
    encoder,h2=train_global_record_protonet(ax,ay,ag,seed=seed,episodes=cfg['meta_episodes'],
        support=cfg['episode_support'],query=cfg['episode_query'],ways=4,lr=cfg['learning_rate'])
    return model,encoder,(ax,ay,ag),{'supervised':h1,'meta':h2,'derived_source_rows':len(ax),'original_backgrounds':len(set(g))}


def half_strength(x,y,g):
    out=x.copy()
    for group in np.unique(g):
        ids=np.flatnonzero(g==group);control=x[ids[y[ids]==0][0]]
        out[ids]=control+.5*(x[ids]-control)
    return out


def select_source_only(x,y,g,columns,interval,cfg,out):
    """This function receives no support-pool or target-query observations."""
    rows=[];roles=[];logs=[]
    for fold,held in enumerate(np.array_split(np.unique(g),cfg['source_folds'])):
        ti=np.flatnonzero(~np.isin(g,held));vi=np.flatnonzero(np.isin(g,held))
        scaler=ChannelStandardizer().fit(x[ti]);tx=scaler.transform(x[ti]);vx=scaler.transform(x[vi])
        vy=y[vi];vg=g[vi];weak=half_strength(vx,vy,vg)
        for seed in cfg['selection_train_seeds']:
            model,encoder,aug,history=source_models(tx,y[ti],g[ti],seed,cfg)
            logs.append({'fold':fold,'seed':seed,**history});features={};geometries={}
            for family,enc in [('statistics',None),('meta_raw',encoder),('supervised_raw',model.encoder)]:
                sf=raw_representation(aug[0],family,columns,interval,enc)
                features[family]=[raw_representation(v,family,columns,interval,enc) for v in [vx,weak]]
                geometries[family]={method:fit_source_geometry(sf,aug[1],method,cfg['covariance_ridge']) for method in cfg['metric_candidates']}
            frozen=digest(model),digest(encoder)
            for support_seed in cfg['selection_support_seeds']:
                rng=np.random.default_rng(np.random.SeedSequence([fold,support_seed]))
                order=rng.permutation(held);support_pool=order[:5];query_groups=order[5:]
                qi=np.flatnonzero(np.isin(vg,query_groups))
                for k in cfg['selection_k']:
                    si=np.flatnonzero(np.isin(vg,support_pool[:k]))
                    if set(g[ti])&set(vg[si]) or set(vg[si])&set(vg[qi]):raise RuntimeError('Source fold record overlap')
                    identity={'fold':fold,'train_seed':seed,'support_seed':support_seed,'k':k}
                    roles.append({**identity,'training_backgrounds':sorted(set(g[ti])),
                        'support_backgrounds':sorted(set(vg[si])),'query_backgrounds':sorted(set(vg[qi]))})
                    for family in cfg['representations']:
                        for method,geometry in geometries[family].items():
                            proto=target_prototypes(features[family][0][si],vy[si],geometry)
                            for variant,ff in zip(['main','half_amplitude'],features[family]):
                                pred=geometry_scores(ff[qi],proto,geometry).argmax(axis=1)
                                rows.append({**identity,'family':family,'candidate':method,'variant':variant,
                                    'balanced_accuracy':float(accuracy_score(vy[qi],pred))})
                    for settings in cfg['transfer_candidates']:
                        adapted,history=preserve_head_adaptation(model,vx[si],vy[si],seed,cfg['target_steps'],settings,cfg['label_smoothing'])
                        for variant,xx in [('main',vx),('half_amplitude',weak)]:
                            pred=predict_process_classifier(adapted,xx[qi]).argmax(axis=1)
                            rows.append({**identity,'family':'transfer','candidate':settings['name'],'variant':variant,
                                'balanced_accuracy':float(accuracy_score(vy[qi],pred))})
                pd.DataFrame(rows).to_csv(out/'source_selection_metrics.csv',index=False)
            if frozen!=(digest(model),digest(encoder)):raise RuntimeError('Source weights changed during fold adaptation')
            print(f'SOURCE VALIDATION fold={fold+1}/3 seed={seed} complete',flush=True)
    table=pd.DataFrame(rows)
    means=table.groupby(['family','candidate'],sort=False).balanced_accuracy.mean().reset_index()
    means.to_csv(out/'source_selection_summary.csv',index=False)
    choices={family:str(part.sort_values('balanced_accuracy',ascending=False,kind='stable').iloc[0].candidate)
        for family,part in means.groupby('family',sort=False)}
    selection={'selection_data':'January-June source backgrounds only','source_original_backgrounds':len(set(g)),
        'target_query_used_in_selection':False,'fresh_source_encoders_per_fold':True,'choices':choices,
        'metric_rows':len(rows),'objective':'Equal mean accuracy over balanced 4-state main/half-amplitude source held-out tasks',
        'fold_roles':roles,'config_sha256':sha(ROOT/'02_few_shot_diagnosis/configs/rcsd_optimization_v2.json'),
        'runner_sha256':sha(__file__),'model_code_sha256':sha(ROOT/'02_few_shot_diagnosis/models/rcsd_optimization.py'),
        'warning':'Old target results have already been inspected: subsequent target evaluation is exploratory optimization.'}
    write_json(out/'source_selection_logs.json',logs);write_json(out/'selection_locked.json',selection)
    return selection


def extra_queries(x,meta,prepared,cfg,raw_path):
    preparation=json.loads((prepared/'preparation.json').read_text(encoding='utf-8'));basecfg=preparation['config']
    if sha(raw_path)!=basecfg['source_sha256']:raise ValueError('Wrong raw workbook')
    raw=pd.read_excel(raw_path,sheet_name='Sheet1').drop(columns='Timestamp').astype(float)
    blocks=pd.read_csv(prepared/'background_blocks.csv');amps=pd.read_csv(prepared/'training_amplitudes.csv')
    amplitudes=dict(zip(amps.tag,amps.amplitude));arrays=list(x);records=meta.to_dict('records')
    for name,variant in cfg['extra_stress_variants'].items():
        for number,block in blocks[blocks.split.eq('query')].iterrows():
            base=raw.iloc[int(block.first_source_row):int(block.last_source_row)+1].reset_index(drop=True)
            parameters=sample_parameters(basecfg,int(number));start=basecfg['window_start_point'];stop=start+basecfg['window_points']
            for c in range(4):
                scene,mask,response=construct_scene(base,c,parameters,variant,amplitudes,basecfg)
                if c and not (response[start:stop]>0).all():raise RuntimeError('Stress window contains preonset labels')
                values=scene.iloc[start:stop].to_numpy(dtype=np.float32).T
                row=meta[(meta.block_id==block.block_id)&(meta.variant=='main')&(meta.class_id==c)].iloc[0].to_dict()
                row.update(sample_id=len(records),window_id=f'{block.block_id}_{name}_c{c}',variant=name,
                    input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),minimum_response=float(response[start:stop].min()) if c else 0.,
                    injected_scalar_values=int(mask[start:stop].sum()))
                arrays.append(values);records.append(row)
    return np.stack(arrays),pd.DataFrame(records)


def final_evaluation(x,meta,prepared,cfg,selection,out):
    prep=json.loads((prepared/'preparation.json').read_text(encoding='utf-8'));basecfg=prep['config'];contract=prep['input_contract']
    source,tasks=support_tasks(meta[meta.variant.isin(basecfg['variants'])],basecfg)
    columns=[c['name'] for c in contract['channels']];interval=contract['interval_seconds'];classes=basecfg['class_names']
    labels=meta.class_id.to_numpy(dtype=np.int64);qi=np.flatnonzero(meta.split.eq('query'))
    variants={v:meta.index[meta.split.eq('query')&meta.variant.eq(v)].tolist() for v in meta.loc[qi,'variant'].unique()}
    for ids in variants.values():
        if len(ids)!=88 or not np.array_equal(np.bincount(labels[ids]),np.full(4,22)):raise RuntimeError('Query budget changed')
    scaler=ChannelStandardizer().fit(x[source]);sx=scaler.transform(x);g=meta.loc[source,'block_id'].to_numpy()
    metrics=[];predictions=[];resources=[];expected=[];logs=[]
    choice=selection['choices'];settings=next(s for s in cfg['transfer_candidates'] if s['name']==choice['transfer'])
    def record(method,scores,k,ss,seed,support,model=None,family=None,geometry=None,proto=None):
        identity={'method':method,'k':k,'support_seed':ss,'train_seed':seed};pred=scores.argmax(axis=1)
        for variant,ids in variants.items():
            positions=np.flatnonzero(np.isin(qi,ids));truth=labels[qi[positions]];p=pred[positions];matrix=confusion_matrix(truth,p,labels=range(4))
            recalls=np.diag(matrix)/matrix.sum(axis=1)
            metrics.append({**identity,'variant':variant,'accuracy':float(accuracy_score(truth,p)),
                'balanced_accuracy':float(recalls.mean()),'macro_f1':float(f1_score(truth,p,average='macro',zero_division=0)),
                **{f'class_{c}_recall':float(v) for c,v in enumerate(recalls)},'confusion_matrix':json.dumps(matrix.tolist()),
                'query_windows':88,'query_unique_backgrounds':22,'support_label_windows':len(support)})
        for i,p,score in zip(qi,pred,scores.max(axis=1)):
            predictions.append({**identity,'variant':meta.loc[i,'variant'],'sample_id':int(i),'window_id':meta.loc[i,'window_id'],
                'block_id':meta.loc[i,'block_id'],'true_class':int(labels[i]),'predicted_class':int(p),'model_score':float(score)})
        if seed in {42,-1} and (ss==11 or k==0):
            key=f'k{k}_{method}';path=out/'resources'/(key+'.pt');path.parent.mkdir(exist_ok=True)
            provenance={'dataset':'RCSD-1YD','semi_synthetic':True,'real_fault_incidents':0,'method':method,
                'selection_sha256':sha(out/'selection_locked.json'),'target_evaluation':'exploratory_after_v1',
                'target_support_ids':list(map(int,support)),'source_backgrounds':31,'source_original_label_windows':124,
                'source_augmentation_levels':cfg['source_augmentation_levels'] if method not in {'meta_metric_no_augmentation','transfer_no_augmentation'} else [1.]}
            if geometry is not None:
                item={'kind':'rcsd_metric_prototype','version':2,'process_contract':contract,'class_names':classes,
                    'scaler':scaler.state(),'representation_family':family,'geometry':geometry,'prototypes':np.asarray(proto).tolist(),
                    'provenance':provenance}
                if model is not None:item.update(architecture=model.architecture,encoder_state=model.state_dict())
                torch.save(item,path);operator='process_metric_prototype.py'
            else:
                kind='prototype' if method=='meta_cosine_augmented' else 'transfer' if k else 'cnn'
                save_process_model(path,model,scaler,contract,classes,kind,proto,provenance)
                operator='process_meta_learning.py' if kind=='prototype' else 'process_transfer_learning.py' if kind=='transfer' else 'process_cnn_diagnosis.py'
            resources.append({'id':key,'method':method,'k':k,'model':path.relative_to(out).as_posix(),'operator':operator,'sha256':sha(path)})
            for i,p,score in zip(qi,pred,scores.max(axis=1)):expected.append({'case':key,'window_id':meta.loc[i,'window_id'],'predicted_label':classes[p],'model_score':float(score)})

    write_json(out/'evaluation_protocol.json',{'config':cfg,'choices':choice,'query_ids_by_variant':variants,
        'selection_sha256':sha(out/'selection_locked.json'),'natural_fault_incidents':0,'target_holdout_status':'exploratory_already_seen_v1',
        'default_delivery_fixed_before_target_scoring':cfg['delivery_choice'],'prepared_v1_checksums':json.loads((prepared/'checksums.json').read_text()),
        'source_ids':source.tolist(),'tasks':tasks,'no_query_control_at_inference':True})
    # Deterministic statistics baseline is repeated only across support sets, not fictitious neural seeds.
    ax,ay,ag=augment_source_pairs(sx[source],labels[source],g,cfg['source_augmentation_levels'])
    sf=raw_representation(ax,'statistics',columns,interval);ff=raw_representation(sx,'statistics',columns,interval)
    geo=fit_source_geometry(sf,ay,choice['statistics'],cfg['covariance_ridge'])
    for task in tasks:
        support=np.asarray(task['support_ids']);proto=target_prototypes(ff[support],labels[support],geo)
        record('statistics_metric_selected',geometry_scores(ff[qi],proto,geo),task['k'],task['support_seed'],-1,support,family='statistics',geometry=geo,proto=proto)
    for seed in cfg['final_train_seeds']:
        model,encoder,aug,h=source_models(sx[source],labels[source],g,seed,cfg)
        oldmodel,oldencoder,oldaug,oldh=source_models(sx[source],labels[source],g,seed,cfg,False)
        frozen=[digest(m) for m in [model,encoder,oldmodel,oldencoder]]
        representations={};geometries={}
        for method,family,enc,train in [('meta_metric_selected','meta_raw',encoder,aug),
                ('supervised_metric_selected','supervised_raw',model.encoder,aug),
                ('meta_metric_no_augmentation','meta_raw',oldencoder,oldaug)]:
            representations[method]=raw_representation(sx,family,columns,interval,enc)
            train_features=raw_representation(train[0],family,columns,interval,enc)
            geometries[method]=fit_source_geometry(train_features,train[1],choice[family],cfg['covariance_ridge'])
        record('source_augmented',predict_process_classifier(model,sx[qi]),0,-1,seed,[],model=model)
        log={'seed':seed,'augmented_source':h,'original_source':oldh,'target':[]}
        for task in tasks:
            support=np.asarray(task['support_ids']);sy=labels[support];identity=(task['k'],task['support_seed'],seed,support)
            for method,family,enc in [('meta_metric_selected','meta_raw',encoder),('supervised_metric_selected','supervised_raw',model.encoder),
                    ('meta_metric_no_augmentation','meta_raw',oldencoder)]:
                features=representations[method];geo=geometries[method];proto=target_prototypes(features[support],sy,geo)
                record(method,geometry_scores(features[qi],proto,geo),*identity,model=enc,family=family,geometry=geo,proto=proto)
            proto=process_prototypes(encoder,sx[support],sy)
            record('meta_cosine_augmented',predict_process_prototypes(encoder,proto,sx[qi]),*identity,model=encoder,proto=proto)
            for method,initial in [('transfer_selected',model),('transfer_no_augmentation',oldmodel)]:
                adapted,history=preserve_head_adaptation(initial,sx[support],sy,seed,cfg['target_steps'],settings,cfg['label_smoothing'])
                record(method,predict_process_classifier(adapted,sx[qi]),*identity,model=adapted)
                log['target'].append({'method':method,'k':task['k'],'support_seed':task['support_seed'],'history':history})
            if frozen!=[digest(m) for m in [model,encoder,oldmodel,oldencoder]]:raise RuntimeError('Shared weights mutated')
            pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
            print(f'FINAL EXPLORATORY seed={seed} K={task["k"]} support={task["support_seed"]} complete',flush=True)
        logs.append(log);write_json(out/'training_logs.json',logs)
    table=pd.DataFrame(metrics)
    table.groupby(['variant','k','method']).agg(accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
        accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),macro_f1_mean=('macro_f1','mean'),
        background_recall=('class_0_recall','mean'),temperature_recall=('class_1_recall','mean'),
        oil_temperature_recall=('class_2_recall','mean'),vibration_axial_recall=('class_3_recall','mean'),n=('accuracy','size')).reset_index().to_csv(out/'aggregate_metrics.csv',index=False)
    pd.DataFrame(expected).to_csv(out/'expected_resource_predictions.csv',index=False);write_json(out/'resource_manifest.json',{'cases':resources})
    meta.to_csv(out/'sample_manifest.csv',index=False);np.savez_compressed(out/'all_windows.npz',x=x)
    for variant,ids in variants.items():
        pieces=[]
        for i in ids:
            part=pd.DataFrame(x[i].T,columns=columns);part.insert(0,'timestamp',pd.date_range(meta.loc[i,'start_time'],periods=20,freq='15min'))
            part.insert(0,'window_id',meta.loc[i,'window_id']);pieces.append(part)
        pd.concat(pieces,ignore_index=True).to_csv(out/f'query_{variant}.csv',index=False)
    completion={'metric_rows':len(table),'prediction_rows':len(predictions),'resources_exported':len(resources),
        'expected_metric_rows':696,'expected_prediction_rows':61248,'expected_resources':22,
        'semi_synthetic':True,'natural_fault_incidents':0,'source_backgrounds':31,'support_pool_backgrounds':11,'query_backgrounds':22,
        'target_evaluation':'exploratory','platform_registered':False,'remote_pushed':False}
    if len(table)!=696 or len(predictions)!=61248 or len(resources)!=22:raise RuntimeError('Incomplete planned comparison')
    write_json(out/'completion.json',completion);print(json.dumps(completion),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'02_few_shot_diagnosis/outputs/rcsd_optimization_v2')
    p.add_argument('--raw',type=Path,default=ROOT.parent/'公开数据集/Centrifugal compressor _2022.xlsx');args=p.parse_args()
    prepared=ROOT/'data/RCSD_semisynthetic_diagnosis_v1';out=args.output_dir
    if out.exists():raise FileExistsError('Keep completed and partial experiments; choose a new output path')
    cfg=json.loads((ROOT/'02_few_shot_diagnosis/configs/rcsd_optimization_v2.json').read_text(encoding='utf-8'))
    for name,checksum in json.loads((prepared/'checksums.json').read_text()).items():
        if sha(prepared/name)!=checksum:raise ValueError('Original prepared evidence changed')
    x=np.load(prepared/'windows.npz',allow_pickle=False)['x'];meta=pd.read_csv(prepared/'sample_manifest.csv')
    if len(x)!=len(meta) or any(hashlib.sha256(a.tobytes()).hexdigest()!=h for a,h in zip(x,meta.input_sha256)):raise ValueError('Window identity mismatch')
    out.mkdir(parents=True);write_json(out/'optimization_plan.json',cfg)
    prep=json.loads((prepared/'preparation.json').read_text());contract=prep['input_contract'];source=np.flatnonzero(meta.split.eq('source'))
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        selection=select_source_only(x[source],meta.loc[source,'class_id'].to_numpy(),meta.loc[source,'block_id'].to_numpy(),
            [c['name'] for c in contract['channels']],contract['interval_seconds'],cfg,out)
        x,meta=extra_queries(x,meta,prepared,cfg,args.raw)
        final_evaluation(x,meta,prepared,cfg,selection,out)


if __name__=='__main__':main()
