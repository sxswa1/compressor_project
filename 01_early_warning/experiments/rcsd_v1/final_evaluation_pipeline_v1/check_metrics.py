"""Analytical edge cases and independent background-bootstrap verification."""
import numpy as np
import pandas as pd
from common import ROOT,write,digest
from metrics import di_score,rul_score

def main(out=None):
    out=out or ROOT/'development_results'
    d=pd.DataFrame(dict(scene_id=['a','a','b','b'],block_id=['A','A','B','B'],step=[0,1,0,1],DI_ref=[.9,.7,.5,.2],prediction=[.9,.5,.5,.7]))
    s=di_score(d)
    np.testing.assert_allclose([s['scene_macro_mae'],s['risk_accuracy'],s['risk_underestimate_fraction'],s['risk_overestimate_fraction'],s['risk_macro_f1']],
        [.175,.5,.25,.25,5/12])
    assert s['reference_switches']==s['predicted_switches']==2
    d=d.drop(columns=['DI_ref']).assign(RUL_ref_hours=[8.,4.,0.,2.],prediction=[10.,3.,2.,2.])
    s=rul_score(d)
    np.testing.assert_allclose([s['rmse_hours'],s['scene_macro_rmse_hours'],s['short_scene_mae_hours'],s['endpoint_mae_hours']],
        [1.5,(np.sqrt(2.5)+np.sqrt(2))/2,1,2])
    assert np.isnan(rul_score(d.iloc[:1])['endpoint_mae_hours'])
    assert rul_score(d.iloc[:0])['points']==0
    # Independently form background multiplicities and dot products, not indexed
    # sums over the sampled block array used by the production summarizers.
    scenes=pd.read_csv(out/'regression_scene_metrics.csv')
    saved=pd.read_csv(out/'regression_background_bootstrap.csv')
    checked=0
    for task,primary in [('di','ridge_history'),('rul','gradient_boosting_history__residual_augmented')]:
        subset=scenes.loc[scenes.task.eq(task)&scenes.stratum.eq('main')]
        ids=sorted(subset.block_id.unique());rng=np.random.default_rng(2026102404)
        weights=np.array([np.bincount(rng.choice(len(ids),size=len(ids),replace=True),minlength=len(ids)) for _ in range(2000)])
        key='scene_macro_mae' if task=='di' else 'scene_macro_rmse_hours';distributions={}
        for family,g in subset.groupby('family'):
            perseed=[]
            for _,f in g.groupby('seed'):
                sums=np.array([f.loc[f.block_id.eq(b),key].sum() for b in ids]);counts=np.array([f.block_id.eq(b).sum() for b in ids])
                perseed.append((weights@sums)/(weights@counts))
            distributions[family]=sum(perseed)/len(perseed)
        for family,a in distributions.items():
            row=saved.loc[saved.task.eq(task)&saved.family.eq(family)].iloc[0]
            delta=a-distributions[primary]
            np.testing.assert_allclose([row.low,row.high,row.difference_vs_primary_low,row.difference_vs_primary_high],
                [*np.quantile(a,[.025,.975]),*np.quantile(delta,[.025,.975])],atol=1e-10)
            checked+=1
    events=pd.read_csv(out/'warning_scene_metrics.csv')
    events=events.loc[events.source.eq('v2')&events.stratum.eq('main')&events.endpoint.notna()].copy()
    events['family']=events.method.str.replace(r'_2026102[345]','',regex=True)
    saved=pd.read_csv(out/'warning_background_bootstrap.csv').set_index('family')
    ids=sorted(events.block_id.unique());rng=np.random.default_rng(2026102404)
    weights=np.array([np.bincount(rng.choice(len(ids),size=len(ids),replace=True),minlength=len(ids)) for _ in range(2000)])
    distributions={}
    for family,g in events.groupby('family'):
        perseed=[]
        for _,f in g.groupby('method'):
            sums=np.array([f.loc[f.block_id.eq(b),'paired_two_hour'].sum() for b in ids]);counts=np.array([f.block_id.eq(b).sum() for b in ids])
            perseed.append((weights@sums)/(weights@counts))
        distributions[family]=sum(perseed)/len(perseed)
    for family,a in distributions.items():
        row=saved.loc[family];delta=a-distributions['v3_core_limits']
        np.testing.assert_allclose([row.low,row.high,row.difference_vs_primary_low,row.difference_vs_primary_high],
            [*np.quantile(a,[.025,.975]),*np.quantile(delta,[.025,.975])],atol=1e-10)
        checked+=1
    write(out/'metric_checks.json',dict(status='PASS',analytical_DI_RUL_edge_cases=True,bootstrap_family_intervals_recomputed=checked,
        draws=2000,seed=2026102404,
        sha256={p.name:digest(p) for p in [ROOT/'metrics.py',ROOT/'summaries.py',ROOT/'check_metrics.py']}))
    print('Analytical metrics and independent bootstrap checks passed:',checked,flush=True)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'development_results')
    main(parser.parse_args().output)
