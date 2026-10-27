from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def fit_reference(raw,targets):
    context=[c for c in raw if c not in targets]
    center=raw[targets].median().to_numpy()
    scale=np.maximum((raw[targets]-center).abs().median().to_numpy()*1.4826,1e-5)
    model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(raw[context],(raw[targets]-center)/scale)
    return dict(targets=targets,context=context,center=center,scale=scale,model=model)


def features(x,reference,variant):
    valid=pd.Series(np.isfinite(x).all(axis=1),index=x.index).rolling(96,min_periods=96).sum().eq(96)
    current=x.add_suffix('__current')
    if variant=='current':return current.loc[valid]
    good=np.isfinite(x).all(axis=1);z=pd.DataFrame(np.nan,index=x.index,columns=reference['targets'])
    z.loc[good]=(x.loc[good,reference['targets']].to_numpy()-reference['center'])/reference['scale']-reference['model'].predict(x.loc[good,reference['context']])
    if variant=='residual_context':f=pd.concat([x[reference['context']].add_suffix('__context'),z.add_suffix('__residual')],axis=1)
    elif variant=='current_residual_trend':
        f=pd.concat([current,z.add_suffix('__residual'),z.rolling(8,min_periods=8).mean().add_suffix('__residual_mean8'),(z-z.shift(7)).add_suffix('__residual_change8')],axis=1)
    else:raise ValueError(variant)
    return f.loc[valid & np.isfinite(f).all(axis=1)]
