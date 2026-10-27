"""Audit fixed seed ensembles, paired-background effects and v1 replay."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=ROOT/'02_few_shot_diagnosis/outputs/rcsd_seed_ensemble_v2';support=ROOT/'02_few_shot_diagnosis/outputs/rcsd_support_optimization_v2'
    if not (out/'completion.json').exists():raise ValueError('Finish all fixed ensemble conditions first')
    prioraudit=json.loads((support/'audit.json').read_text());x=np.load(out/'all_windows.npz',allow_pickle=False)['x'];ref=np.load(support/'all_windows.npz',allow_pickle=False)['x']
    if not np.array_equal(x,ref) or prioraudit['independently_reconstructed_windows']!=520:raise ValueError('Audited data differ')
    meta=pd.read_csv(out/'sample_manifest.csv');other=pd.read_csv(support/'sample_manifest.csv')
    if not meta.equals(other):raise ValueError('Audited labels differ')
    plan=json.loads((out/'plan_locked.json').read_text());protocol=json.loads((out/'evaluation_protocol.json').read_text());roles=json.loads((out/'source_validation_roles.json').read_text())
    if sha(__file__.replace('analyze_rcsd_seed_ensemble.py','run_rcsd_seed_ensemble.py'))!=plan['runner_sha256']:raise ValueError('Runner changed')
    for fold in roles:
        a=set(fold['training_backgrounds']);b=set(fold['support_backgrounds']);c=set(fold['query_backgrounds'])
        if a&b or a&c or b&c or max(a)>=min(b|c):raise ValueError('Source temporal/record isolation changed')
    for task in protocol['tasks']:
        part=meta.loc[task['support_ids']]
        if not part.split.eq('support_pool').all() or len(part)!=4*task['k']:raise ValueError('Support/query leakage')
    resources=json.loads((out/'resource_manifest.json').read_text())['cases'];ensembles=0
    for case in resources:
        path=out/case['model']
        if sha(path)!=case['sha256']:raise ValueError('Exported resource changed')
        item=torch.load(path,map_location='cpu',weights_only=True)
        if item['kind']=='process_seed_ensemble':
            ensembles+=1;ids=item['provenance']['same_support_ids']
            for seed,member in zip(item['train_seeds'],item['members']):
                if member['provenance']['support_ids']!=ids or member['provenance']['train_seed']!=seed:raise ValueError('Members used different labels or undeclared seeds')
                key=f"k{case['k']}_{member['provenance']['method']}_seed{seed}"
                single=torch.load(out/'resources'/(key+'.pt'),weights_only=True,map_location='cpu')
                if any(not torch.equal(v,single['state_dict'][k]) for k,v in member['state_dict'].items()):raise ValueError('Ensemble state differs from declared member')
    if ensembles!=9:raise ValueError('Missing ensemble resources')
    pred=pd.read_csv(out/'predictions.csv');pred['correct']=pred.true_class.eq(pred.predicted_class).astype(float)
    old=pd.read_csv(ROOT/'02_few_shot_diagnosis/outputs/rcsd_diagnosis_v1/predictions.csv');reproduction=[]
    for method in ['cnn_scratch','transfer','meta_adapted_prototype']:
        current=pred[pred.method.eq(method)&pred.variant.isin(['main','weak_lag_challenge'])];previous=old[old.method.eq(method)]
        matched=current.merge(previous,on=['variant','k','train_seed','support_seed','window_id'],validate='one_to_one',suffixes=('_new','_v1'))
        if len(matched)!=4752 or not matched.predicted_class_new.eq(matched.predicted_class_v1).all():raise ValueError('V1 neural baseline not reproduced')
        difference=float(np.max(np.abs(matched.model_score_new-matched.model_score_v1)))
        if difference>1e-6:raise ValueError('Original baseline score is not numerically reproduced')
        reproduction.append({'method':method,'matched_predictions':len(matched),'identical_classes':True,'max_score_difference':difference})
    pd.DataFrame(reproduction).to_csv(out/'v1_baseline_reproduction.csv',index=False)
    rng=np.random.default_rng(2026102603);effects=[];errors=[]
    for variant in pred.variant.unique():
        for k in [1,5,10]:
            for family in ['cnn_scratch','transfer','meta_adapted_prototype']:
                single=pred[(pred.variant==variant)&(pred.k==k)&(pred.method==family)]
                ens=pred[(pred.variant==variant)&(pred.k==k)&(pred.method==family+'_seed_ensemble')]
                keys=['support_seed','window_id','block_id','true_class'];average=single.groupby(keys,as_index=False).correct.mean()
                paired=ens.merge(average,on=keys,validate='one_to_one',suffixes=('_ensemble','_member_mean'))
                if len(paired)!=264:raise ValueError('Missing paired ensemble predictions')
                paired['gain']=paired.correct_ensemble-paired.correct_member_mean;bybg=paired.groupby('block_id').gain.mean().to_numpy()
                boot=bybg[rng.integers(0,22,(20000,22))].mean(axis=1);low,high=np.quantile(boot,[.025,.975])
                effects.append({'variant':variant,'k':k,'family':family,'gain_pp':100*bybg.mean(),'background_bootstrap_low_pp':100*low,
                    'background_bootstrap_high_pp':100*high,'ensemble_accuracy':ens.correct.mean(),'member_mean_accuracy':single.correct.mean(),
                    'nonoverlapping_query_backgrounds':22,'ensemble_support_draws':3,'member_train_seeds':3,
                    'uncertainty_scope':'Fixed seed ensemble; average support draws and state pairs within background before bootstrap; exploratory only'})
            for method,part in pred[(pred.variant==variant)&(pred.k==k)].groupby('method'):
                normal=part[part.true_class.eq(0)];fault=part[part.true_class.ne(0)]
                errors.append({'variant':variant,'k':k,'method':method,'background_false_positive_rate':1-normal.correct.mean(),
                    'anomaly_to_background_miss_rate':fault.predicted_class.eq(0).mean(),
                    'wrong_anomaly_type_rate':((fault.predicted_class!=0)&~fault.correct.astype(bool)).mean()})
    pd.DataFrame(effects).to_csv(out/'paired_background_gains.csv',index=False);pd.DataFrame(errors).to_csv(out/'error_summary.csv',index=False)
    audit={'independently_reconstructed_windows':520,'identical_to_independently_audited_input':True,'source_forward_role_checks':len(roles),
        'resources_verified':len(resources),'ensemble_resources':ensembles,'independent_members':3,'extra_target_label_windows':0,
        'v1_neural_predictions_reproduced':sum(p['matched_predictions'] for p in reproduction),'query_backgrounds':22,'natural_fault_incidents':0,
        'all_target_results_exploratory':True,'query_normal_reference_required':False,'raw_workbook_unchanged':prioraudit['original_workbook_unchanged']}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
