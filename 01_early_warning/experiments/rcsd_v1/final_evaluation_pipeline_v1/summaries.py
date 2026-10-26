"""Seed-family summaries, paired background bootstrap and support denominators."""
import argparse
import re
import numpy as np
import pandas as pd
from common import ROOT,read,write

def supplementary(out):
    events=pd.read_csv(out/'warning_scene_metrics.csv')
    events['family']=events.method.str.replace(r'_2026102[345]','',regex=True)
    events['seed']=events.method.str.extract(r'(2026102[345])',expand=False).fillna('20261024').astype(int)
    complete=events.loc[events.endpoint.notna()].copy()
    rows=[]
    for (source,label,family),g in complete.groupby(['source','stratum','family'],sort=False):
        values=g.groupby('seed').paired_two_hour.mean()
        rows.append(dict(source=source,stratum=label,family=family,seeds=len(values),seed_mean=values.mean(),seed_min=values.min(),seed_max=values.max()))
    pd.DataFrame(rows).to_csv(out/'warning_family_summary.csv',index=False)
    main=complete.loc[complete.source.eq('v2')&complete.stratum.eq('main')]
    ids=sorted(main.block_id.unique());rng=np.random.default_rng(2026102404)
    draws=rng.integers(0,len(ids),size=(2000,len(ids)));values={};estimates={}
    for family,g in main.groupby('family',sort=False):
        perseed=[];point=[]
        for _,s in g.groupby('seed'):
            block=s.groupby('block_id').paired_two_hour.agg(['sum','count']).reindex(ids)
            assert not block.isna().any().any()
            perseed.append(block['sum'].to_numpy()[draws].sum(axis=1)/block['count'].to_numpy()[draws].sum(axis=1))
            point.append(s.paired_two_hour.mean())
        values[family]=np.mean(perseed,axis=0);estimates[family]=float(np.mean(point))
    rows=[]
    for family,a in values.items():
        delta=a-values['v3_core_limits']
        rows.append(dict(family=family,metric='paired_2h_coverage',backgrounds=len(ids),draws=2000,bootstrap_seed=2026102404,
            estimate=estimates[family],low=float(np.quantile(a,.025)),high=float(np.quantile(a,.975)),
            difference_vs_primary_low=float(np.quantile(delta,.025)),difference_vs_primary_high=float(np.quantile(delta,.975)),
            interpretation='descriptive paired background bootstrap on one machine; seeds averaged within draws'))
    pd.DataFrame(rows).to_csv(out/'warning_background_bootstrap.csv',index=False)
    availability=pd.read_csv(out/'regression_eligibility.csv');rows=[]
    for (task,label),g in availability.groupby(['task','stratum'],sort=False):
        rows.append(dict(task=task,stratum=label,scenes=len(g),rows=int(g.rows.sum()),valid_outputs=int(g.valid_outputs.sum()),
            valid_output_fraction=float(g.valid_outputs.sum()/g.rows.sum()),label_valid_points=int(g.label_valid.sum()),
            evaluated_points=int(g.evaluated_points.sum()),
            labeled_output_fraction=float(g.evaluated_points.sum()/g.label_valid.sum()) if g.label_valid.sum() else np.nan,
            right_censored_scenes=int(g.right_censored.sum()),
            **{r:int(g[r].sum()) for r in ['insufficient_history','time_gap_history','missing_sensor_history','stopped_history']}))
    pd.DataFrame(rows).to_csv(out/'regression_availability_summary.csv',index=False)
    index=pd.read_csv(out/'scene_index.csv');sets={}
    for source,g in index.groupby('source'):
        sets[source]=set(i for r in g.itertuples() for i in range(int(r.first_source_row),int(r.last_source_row)+1))
    write(out/'source_overlap.json',dict(controlled_unique_source_points=len(sets['v2']),stress_unique_source_points=len(sets['stress']),
        shared_source_points=len(sets['v2']&sets['stress']),interpretation='Shared source backgrounds; never count the two batteries as independent datasets'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT/'development_results')
    supplementary(parser.parse_args().output)
