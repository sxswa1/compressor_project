"""Frozen all-method RCSD semi-synthetic diagnosis study and resource export.

Known simulated states across periods; no natural failure or new-class claim.
The final query participates only in prediction/scoring after the protocol is
written. Repeated seeds and response variants never increase incident counts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / '02_few_shot_diagnosis'))
from pipelines.rcsd_semisynthetic import sha, support_tasks
from pipelines.process_feature_selection import select_support_features
from models.process_learning import (train_process_classifier, process_prototypes,
    predict_process_classifier, predict_process_prototypes)
from models.global_record_meta import train_global_record_protonet
from common_algorithms.models.process_cnn import ChannelStandardizer, save_process_model
from common_algorithms.models.random_forest import train_random_forest
from common_algorithms.feature_extraction.process_features import process_features


def state_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared-dir', type=Path, default=ROOT / 'data/RCSD_semisynthetic_diagnosis_v1')
    p.add_argument('--output-dir', type=Path, default=ROOT / '02_few_shot_diagnosis/outputs/rcsd_diagnosis_v1')
    args = p.parse_args();prepared = args.prepared_dir;out = args.output_dir
    if out.exists(): raise FileExistsError('Preserve partial/completed studies; choose another output directory')
    preparation = json.loads((prepared / 'preparation.json').read_text());cfg = preparation['config']
    for name, digest in json.loads((prepared / 'checksums.json').read_text()).items():
        if sha(prepared / name) != digest: raise ValueError('Prepared input changed: ' + name)
    x = np.load(prepared / 'windows.npz', allow_pickle=False)['x']
    meta = pd.read_csv(prepared / 'sample_manifest.csv')
    if len(x) != len(meta) or any(hashlib.sha256(a.tobytes()).hexdigest() != b for a,b in zip(x,meta.input_sha256)):
        raise ValueError('Prepared window identities differ')
    source, tasks = support_tasks(meta, cfg);contract = preparation['input_contract']
    columns = [c['name'] for c in contract['channels']];classes = cfg['class_names']
    query_by_variant = tasks[0]['query_ids'];query_ids = np.concatenate([np.asarray(v) for v in query_by_variant.values()])
    for ids in query_by_variant.values():
        if not np.array_equal(np.bincount(meta.loc[ids, 'class_id'], minlength=4), np.full(4,22)):
            raise ValueError('Keep all 22 backgrounds and all four planned states')
    files = ['02_few_shot_diagnosis/scripts/run_rcsd_diagnosis.py',
             '02_few_shot_diagnosis/pipelines/rcsd_semisynthetic.py',
             '02_few_shot_diagnosis/models/process_learning.py', '02_few_shot_diagnosis/models/global_record_meta.py',
             'common_algorithms/models/process_cnn.py', 'common_algorithms/feature_extraction/process_features.py',
             '02_few_shot_diagnosis/pipelines/process_feature_selection.py']
    protocol = {'preparation': preparation, 'prepared_checksums': json.loads((prepared / 'checksums.json').read_text()),
        'implementation_sha256': {s: sha(ROOT/s) for s in files}, 'source_ids': source.tolist(), 'tasks': tasks,
        'query_ids_by_variant': query_by_variant, 'normalization_fit': 'source windows only, shared across baselines',
        'known_classes_in_source_and_target': True, 'natural_fault_incidents': 0,
        'query_used_for_training_or_model_selection': False, 'primary_vs_challenge_reported_separately': True,
        'source_label_windows': len(source), 'source_original_backgrounds': 31,
        'expected_metric_rows': 366, 'expected_exported_resources': 25,
        'target_labels_per_task': '4*K windows from K unique original backgrounds, not 4*K incidents',
        'rf_selection': 'support-only ANOVA top35, K1 uses all features; separate ablation',
        'source_only_comparator': 'zero target labels; K=0 once per source seed',
        'default_delivery_fixed_before_scoring': cfg['predeclared_delivery']}
    out.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    meta.to_csv(out/'sample_manifest.csv',index=False)
    scaler = ChannelStandardizer().fit(x[source]);scaled = scaler.transform(x)
    features,names = process_features(scaled,columns,contract['interval_seconds'])
    labels = meta.class_id.to_numpy(dtype=np.int64)
    metrics,predictions,logs,resources,resource_expected = [],[],[],[],[]
    torch.set_num_threads(2)
    operator = {'random_forest':'process_random_forest.py','random_forest_selected':'process_random_forest.py',
        'cnn_scratch':'process_cnn_diagnosis.py','transfer':'process_transfer_learning.py','source_cnn':'process_cnn_diagnosis.py',
        'protonet':'process_meta_learning.py','supervised_prototype':'process_meta_learning.py',
        'meta_adapted_prototype':'process_meta_learning.py','scratch_prototype':'process_meta_learning.py'}
    prototype_methods = {'protonet','supervised_prototype','meta_adapted_prototype','scratch_prototype'}

    def record(method, scores, k, support_seed, seed, support, model=None, proto=None, forest=None, selected=None, selection=None):
        predicted = scores.argmax(axis=1)
        identity = {'method':method,'k':k,'support_seed':support_seed,'train_seed':seed}
        for variant, ids in query_by_variant.items():
            pos=np.flatnonzero(np.isin(query_ids,ids));truth=labels[query_ids[pos]];pred=predicted[pos]
            recalls=recall_score(truth,pred,labels=range(4),average=None,zero_division=0)
            metrics.append({**identity,'variant':variant,'source_label_windows':len(source),
                'support_label_windows':len(support),'support_unique_backgrounds':len(set(meta.loc[support,'block_id'])),
                'query_windows':len(pos),'query_unique_backgrounds':22,'features_used':len(selected) if selected is not None else len(names) if forest is not None else None,
                'accuracy':float(accuracy_score(truth,pred)),'balanced_accuracy':float(recalls.mean()),
                'macro_f1':float(f1_score(truth,pred,labels=range(4),average='macro',zero_division=0)),
                **{f'class_{c}_recall':float(v) for c,v in enumerate(recalls)},
                'confusion_matrix':json.dumps(confusion_matrix(truth,pred,labels=range(4)).tolist())})
        for i, pred, score in zip(query_ids,predicted,scores.max(axis=1)):
            predictions.append({**identity,'variant':meta.loc[i,'variant'],'sample_id':int(i),
                'window_id':meta.loc[i,'window_id'],'block_id':meta.loc[i,'block_id'],'true_class':int(labels[i]),
                'predicted_class':int(pred),'true_label':classes[labels[i]],'predicted_label':classes[pred],'model_score':float(score)})
        canonical = seed in {42,-1} and (support_seed==11 or method=='source_cnn')
        if canonical:
            key=f'k{k}_{method}';path=out/'resources'/(key+('.joblib' if forest is not None else '.pt'))
            path.parent.mkdir(exist_ok=True)
            provenance={'dataset':'RCSD-1YD','kind':'controlled_semi_synthetic','method':method,
                'source_label_windows':len(source),'source_backgrounds':31,'target_support_ids':list(map(int,support)),
                'target_support_backgrounds':sorted(set(meta.loc[support,'block_id'])),
                'known_classes_same_across_dates':True,'real_fault_incidents':0,
                'default_delivery_chosen_before_scoring':cfg['predeclared_delivery'],
                'meta_training':method in {'protonet','meta_adapted_prototype'}}
            if forest is not None:
                item={'model':forest,'feature_columns':names,'scaler':scaler.state(),'process_contract':contract,
                      'class_names':classes,'provenance':provenance}
                if selected is not None:item['selected_feature_indices']=selected
                if selection is not None:item['feature_selection']=selection
                joblib.dump(item,path)
            else:
                kind='prototype' if method in prototype_methods else 'transfer' if method=='transfer' else 'cnn'
                save_process_model(path,model,scaler,contract,classes,kind,proto,provenance)
            resources.append({'id':key,'method':method,'k':k,'model':path.relative_to(out).as_posix(),
                              'operator':operator[method],'sha256':sha(path)})
            for i,pred,score in zip(query_ids,predicted,scores.max(axis=1)):
                resource_expected.append({'case':key,'window_id':meta.loc[i,'window_id'],
                    'predicted_label':classes[pred],'model_score':float(score)})

    with threadpool_limits(limits=2):
        for seed in cfg['train_seeds']:
            pretrained,h1=train_process_classifier(scaled[source],labels[source],seed=seed,steps=cfg['source_steps'],lr=cfg['learning_rate'])
            encoder,h2=train_global_record_protonet(scaled[source],labels[source],meta.loc[source,'block_id'].to_numpy(),
                seed=seed,episodes=cfg['meta_episodes'],support=cfg['episode_support'],query=cfg['episode_query'],ways=4,lr=cfg['learning_rate'])
            original_classifier=state_digest(pretrained);original_meta=state_digest(encoder)
            record('source_cnn',predict_process_classifier(pretrained,scaled[query_ids]),0,-1,seed,[],model=pretrained)
            log={'seed':seed,'source_classifier':h1,'episodic_meta':h2,'source_class_counts':np.bincount(labels[source]).tolist(),'target':[]}
            for task in tasks:
                support=np.asarray(task['support_ids']);sy=labels[support];ssx=scaled[support];qx=scaled[query_ids]
                scratch,h3=train_process_classifier(ssx,sy,seed=seed,head_seed=seed+100000,steps=cfg['target_steps'],lr=cfg['learning_rate'])
                transfer,h4=train_process_classifier(ssx,sy,seed=seed,head_seed=seed+100000,steps=cfg['target_steps'],lr=cfg['learning_rate'],encoder=pretrained.encoder)
                adapted,h5=train_process_classifier(ssx,sy,seed=seed,head_seed=seed+100000,steps=cfg['target_steps'],lr=cfg['learning_rate'],encoder=encoder)
                log['target'].append({'k':task['k'],'support_seed':task['support_seed'],'scratch':h3,'transfer':h4,'meta_adaptation':h5})
                kwargs={'k':task['k'],'support_seed':task['support_seed'],'seed':seed,'support':support}
                record('cnn_scratch',predict_process_classifier(scratch,qx),model=scratch,**kwargs)
                record('transfer',predict_process_classifier(transfer,qx),model=transfer,**kwargs)
                for method,enc in [('protonet',encoder),('supervised_prototype',pretrained.encoder),
                                   ('meta_adapted_prototype',adapted.encoder),('scratch_prototype',scratch.encoder)]:
                    proto=process_prototypes(enc,ssx,sy)
                    record(method,predict_process_prototypes(enc,proto,qx),model=enc,proto=proto,**kwargs)
                if seed==42:
                    for method,selected,selection in [('random_forest',None,None),
                        ('random_forest_selected',*select_support_features(features[support],sy,cfg['selected_forest_features']))]:
                        sf=features[support] if selected is None else features[support][:,selected]
                        qf=features[query_ids] if selected is None else features[query_ids][:,selected]
                        forest=train_random_forest(sf,sy,n_estimators=cfg['forest_trees'],random_state=42)
                        record(method,forest.predict_proba(qf),forest=forest,selected=selected,selection=selection,
                            **{**kwargs,'seed':-1})
                if state_digest(pretrained)!=original_classifier or state_digest(encoder)!=original_meta:
                    raise RuntimeError('Target adaptation modified a shared source encoder')
                pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
                pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
                print(f'RCSD seed={seed} K={task["k"]} support_seed={task["support_seed"]} done',flush=True)
            log['shared_source_weights_unchanged']=True;logs.append(log)
            (out/'training_logs.json').write_text(json.dumps(logs,indent=2)+'\n',encoding='utf-8')
    metrics=pd.DataFrame(metrics)
    if len(metrics)!=protocol['expected_metric_rows'] or len(resources)!=protocol['expected_exported_resources']:
        raise RuntimeError('Planned full comparison or exported resources are incomplete')
    for path,checksum in protocol['implementation_sha256'].items():
        if sha(ROOT/path)!=checksum:raise RuntimeError('Frozen implementation changed during training')
    aggregate=metrics.groupby(['variant','k','method']).agg(accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
        accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),macro_f1_mean=('macro_f1','mean'),
        background_recall=('class_0_recall','mean'),temperature_recall=('class_1_recall','mean'),
        oil_temperature_recall=('class_2_recall','mean'),vibration_axial_recall=('class_3_recall','mean'),n=('accuracy','size')).reset_index()
    aggregate.to_csv(out/'aggregate_metrics.csv',index=False)
    pd.DataFrame(resource_expected).to_csv(out/'expected_resource_predictions.csv',index=False)
    (out/'resource_manifest.json').write_text(json.dumps({'cases':resources},indent=2)+'\n',encoding='utf-8')
    for variant,ids in query_by_variant.items():
        tables=[]
        for i in ids:
            part=pd.DataFrame(x[i].T,columns=columns)
            part.insert(0,'timestamp',pd.date_range(meta.loc[i,'start_time'],periods=cfg['window_points'],freq='15min'))
            part.insert(0,'window_id',meta.loc[i,'window_id']);tables.append(part)
        pd.concat(tables,ignore_index=True).to_csv(out/f'query_{variant}.csv',index=False)
    completion={'metric_rows':len(metrics),'prediction_rows':len(predictions),'resources_exported':len(resources),
        'all_planned_conditions_completed':True,'query_backgrounds':22,'unique_backgrounds':64,
        'natural_fault_incidents':0,'state_count_including_control':4,'semi_synthetic':True,
        'platform_registration_tested':False,'remote_pushed':False,'frozen_implementation_unchanged':True}
    (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(completion,indent=2),flush=True)


if __name__=='__main__':main()
