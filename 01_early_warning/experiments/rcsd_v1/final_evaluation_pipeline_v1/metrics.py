"""Frozen aggregation: scene/background units, no warning-gated RUL selection."""
import numpy as np
import pandas as pd
from common import risk_levels

def avg(values):
    return float(np.mean(values)) if len(values) else np.nan

def di_score(frame):
    if frame.empty:return dict(points=0,scenes=0,backgrounds=0)
    y,p=frame.DI_ref.to_numpy(),frame.prediction.to_numpy();e=p-y
    assert np.isfinite(y).all() and np.isfinite(p).all()
    a,b=risk_levels(y),risk_levels(p);cm=np.zeros((4,4),int)
    np.add.at(cm,(a-1,b-1),1)
    recall=np.divide(cm.diagonal(),cm.sum(axis=1),out=np.full(4,np.nan),where=cm.sum(axis=1)>0)
    f1=np.divide(2*cm.diagonal(),cm.sum(axis=1)+cm.sum(axis=0),out=np.zeros(4),where=cm.sum(axis=1)+cm.sum(axis=0)>0)
    details=frame[['scene_id','block_id','step']].assign(absolute=np.abs(e),reference_risk=a,predicted_risk=b)
    changes=[]
    for _,g in details.groupby('scene_id',sort=False):
        g=g.sort_values('step');cont=g.step.diff().eq(1)
        changes.append((int((g.reference_risk.diff().ne(0)&cont).sum()),int((g.predicted_risk.diff().ne(0)&cont).sum())))
    return dict(points=len(frame),scenes=int(frame.scene_id.nunique()),backgrounds=int(frame.block_id.nunique()),
        mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),
        scene_macro_mae=float(details.groupby('scene_id').absolute.mean().mean()),
        background_macro_mae=float(details.groupby('block_id').absolute.mean().mean()),
        risk_accuracy=float((a==b).mean()),risk_macro_f1=float(f1.mean()),
        risk_underestimate_fraction=float((b<a).mean()),risk_overestimate_fraction=float((b>a).mean()),
        reference_switches=sum(a for a,b in changes),predicted_switches=sum(b for a,b in changes),
        **{f'risk_recall_{i+1}':float(r) for i,r in enumerate(recall)})

def rul_score(frame):
    if frame.empty:return dict(points=0,scenes=0,backgrounds=0)
    y,p=frame.RUL_ref_hours.to_numpy(),frame.prediction.to_numpy();e=p-y
    assert np.isfinite(y).all() and np.isfinite(p).all() and (y>=0).all() and (p>=0).all()
    d=frame[['scene_id','block_id']].assign(absolute=np.abs(e),squared=e**2)
    short=y<=6;zero=y==0;long=y>6;positive=y>0
    rmse=float(np.sqrt((e**2).mean()))
    result=dict(points=len(frame),scenes=int(frame.scene_id.nunique()),backgrounds=int(frame.block_id.nunique()),
        short_points=int(short.sum()),endpoint_points=int(zero.sum()),positive_points=int(positive.sum()),
        mae_hours=float(np.abs(e).mean()),rmse_hours=rmse,
        scene_macro_mae_hours=float(d.groupby('scene_id').absolute.mean().mean()),
        scene_macro_rmse_hours=float(np.sqrt(d.groupby('scene_id').squared.mean()).mean()),
        background_macro_rmse_hours=float(np.sqrt(d.groupby('block_id').squared.mean()).mean()),
        short_scene_mae_hours=float(d.loc[short].groupby('scene_id').absolute.mean().mean()) if short.any() else np.nan,
        endpoint_mae_hours=avg(np.abs(e[zero])),long_point_mae_hours=avg(np.abs(e[long])),
        long_point_rmse_hours=float(np.sqrt((e[long]**2).mean())) if long.any() else np.nan,
        rmse_div_mean_RUL=rmse/float(y.mean()) if y.mean()>0 else np.nan,
        relative_RMSE_positive_RUL=float(np.sqrt(((e[positive]/y[positive])**2).mean())) if positive.any() else np.nan,
        mean_reference_RUL_hours=float(y.mean()))
    if 'lower_hours' in frame and frame.lower_hours.notna().all():
        d['covered']=frame.RUL_ref_hours.between(frame.lower_hours,frame.upper_hours).to_numpy()
        width=frame.upper_hours-frame.lower_hours
        result.update(point_coverage=float(d.covered.mean()),whole_scene_coverage=float(d.groupby('scene_id').covered.all().mean()),
            whole_background_coverage=float(d.groupby('block_id').covered.all().mean()),mean_width_hours=float(width.mean()),median_width_hours=float(width.median()))
    return result

def warning_summary(events):
    complete=events.loc[events.endpoint.notna()].copy();rows=[]
    for (source,stratum,method),g in complete.groupby(['source','stratum','method'],sort=False):
        rows.append(dict(source=source,stratum=stratum,method=method,scenes=len(g),backgrounds=g.block_id.nunique(),
            hits=int(g.hit.sum()),paired_hits=int(g.paired_hit.sum()),two_hour=int(g.two_hour.sum()),paired_two_hour=int(g.paired_two_hour.sum()),
            paired_two_hour_fraction=float(g.paired_two_hour.mean()),no_paired_event=int((~g.paired_hit.astype(bool)).sum()),
            paired_late_under_2h=int((g.paired_hit.astype(bool)&~g.paired_two_hour.astype(bool)).sum()),
            median_paired_lead=float(g.paired_lead_hours.dropna().median()) if g.paired_lead_hours.notna().any() else np.nan,
            valid_output_fraction=float(g.valid_points.sum()/g.rows.sum())))
    return pd.DataFrame(rows)

def bootstrap_primary(scene_scores,task,primary,seeds,draws=2000,seed=2026102404):
    """Same resampled backgrounds for all families; seeds averaged per draw."""
    metric='scene_macro_mae' if task=='di' else 'scene_macro_rmse_hours'
    subset=scene_scores.loc[scene_scores.stratum.eq('main')]
    block_ids=sorted(subset.block_id.unique());rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(block_ids),size=(draws,len(block_ids)))
    distributions={};rows=[]
    for family,g in subset.groupby('family',sort=False):
        per_seed=[]
        for _,s in g.groupby('seed'):
            block=s.groupby('block_id')[metric].agg(['sum','count']).reindex(block_ids)
            assert not block.isna().any().any()
            sums,counts=block['sum'].to_numpy(),block['count'].to_numpy()
            per_seed.append(sums[samples].sum(axis=1)/counts[samples].sum(axis=1))
        distributions[family]=np.mean(per_seed,axis=0)
    for family,values in distributions.items():
        delta=values-distributions[primary]
        point=subset.loc[subset.family.eq(family)].groupby('seed')[metric].mean().mean()
        rows.append(dict(task=task,family=family,metric=metric,backgrounds=len(block_ids),draws=draws,bootstrap_seed=seed,
            estimate=float(point),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),
            difference_vs_primary_low=float(np.quantile(delta,.025)),difference_vs_primary_high=float(np.quantile(delta,.975)),
            interpretation='descriptive paired background bootstrap on one machine; seeds averaged within draws'))
    return pd.DataFrame(rows)
