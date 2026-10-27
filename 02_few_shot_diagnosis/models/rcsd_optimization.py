"""Source-only geometry and conservative same-class transfer improvements.

These are explicit ablations for the RCSD semi-synthetic study. Metric
estimation consumes only permitted source representations; target means use
only labelled supports. No reference background is required at inference.
"""
import copy
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch import nn
import torch.nn.functional as F

from common_algorithms.feature_extraction.process_features import process_features


def augment_source_pairs(x, y, groups, levels):
    x=np.asarray(x,dtype=np.float32);y=np.asarray(y);groups=np.asarray(groups)
    if x.ndim!=3 or y.shape!=(len(x),) or groups.shape!=y.shape or not np.isfinite(x).all():
        raise ValueError('Finite aligned source windows required')
    if not levels or any(not np.isfinite(v) or v<=0 for v in levels):raise ValueError('Positive augmentation levels required')
    out=[];labels=[];records=[]
    for group in np.unique(groups):
        ids=np.flatnonzero(groups==group)
        if len(ids)!=4 or set(y[ids])!=set(range(4)):raise ValueError('One original source example per state/background required')
        control=x[ids[y[ids]==0][0]]
        for c in range(4):
            observed=x[ids[y[ids]==c][0]]
            for level in ([1.] if c==0 else levels):
                out.append(control+np.float32(level)*(observed-control));labels.append(c);records.append(group)
    return np.stack(out),np.asarray(labels,dtype=np.int64),np.asarray(records)


def raw_representation(x, family, columns, interval_seconds, encoder=None):
    x=np.asarray(x,dtype=np.float32)
    if family=='statistics':return process_features(x,columns,interval_seconds)[0]
    if family not in {'meta_raw','supervised_raw'} or encoder is None:raise ValueError('Declare a source-trained representation')
    encoder.eval();rows=[]
    with torch.no_grad():
        for start in range(0,len(x),128):rows.append(encoder(torch.from_numpy(x[start:start+128])).numpy())
    return np.concatenate(rows).astype(float) if rows else np.empty((0,encoder.architecture['embedding_dim']))


def fit_source_geometry(features, labels, method, ridge=.05):
    x=np.asarray(features,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or len(x)!=len(y) or not len(x) or not np.isfinite(x).all():raise ValueError('Finite source feature table required')
    if method not in {'euclidean','diagonal_within','shrinkage_within'} or ridge<=0:raise ValueError('Invalid geometry configuration')
    center=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale>1e-6,scale,1.)
    z=(x-center)/scale
    residual=z.copy()
    for c in np.unique(y):residual[y==c]-=z[y==c].mean(axis=0)
    dimension=z.shape[1]
    if method=='euclidean':precision=np.eye(dimension)
    elif method=='diagonal_within':precision=np.diag(1/(np.mean(residual**2,axis=0)+ridge))
    else:
        covariance=LedoitWolf(assume_centered=True).fit(residual).covariance_
        precision=np.linalg.inv(covariance+ridge*np.eye(dimension))
    precision=(precision+precision.T)/2
    if not np.isfinite(precision).all() or np.linalg.eigvalsh(precision).min()<=0:raise ValueError('Geometry is not positive definite')
    return {'method':method,'center':center.tolist(),'scale':scale.tolist(),'precision':precision.tolist(),
            'ridge':ridge,'dimension':dimension,'source_feature_rows':len(x),'query_feature_rows':0}


def target_prototypes(features, labels, geometry):
    x=np.asarray(features,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.array_equal(np.unique(y),np.arange(4)):
        raise ValueError('All four known support classes required')
    z=(x-np.asarray(geometry['center']))/np.asarray(geometry['scale'])
    return np.stack([z[y==c].mean(axis=0) for c in range(4)])


def geometry_scores(features, prototypes, geometry):
    x=np.asarray(features,dtype=float);p=np.asarray(prototypes,dtype=float)
    center=np.asarray(geometry['center']);scale=np.asarray(geometry['scale']);precision=np.asarray(geometry['precision'])
    if x.ndim!=2 or x.shape[1]!=len(center) or p.shape!=(4,len(center)) or not np.isfinite(x).all():
        raise ValueError('Representation and prototypes differ')
    delta=((x-center)/scale)[:,None,:]-p[None,:,:]
    distances=np.einsum('ncd,de,nce->nc',delta,precision,delta,optimize=True)
    logits=-np.maximum(distances,0)/max(len(center),1)
    logits-=logits.max(axis=1,keepdims=True)
    values=np.exp(logits);return values/values.sum(axis=1,keepdims=True)


def preserve_head_adaptation(source_model,x,y,seed,steps,settings,label_smoothing=.05):
    """Same known classes only. Copy the full source model and keep its head."""
    x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.int64)
    if x.ndim!=3 or y.shape!=(len(x),) or not np.array_equal(np.unique(y),np.arange(source_model.head.out_features)):
        raise ValueError('Same known class meaning and complete labelled supports required')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=copy.deepcopy(source_model)
    anchor={name:value.detach().clone() for name,value in model.encoder.named_parameters()}
    encoder_lr=settings['encoder_lr'];head_lr=settings['head_lr']
    for parameter in model.encoder.parameters():parameter.requires_grad_(encoder_lr>0)
    parameters=[{'params':model.head.parameters(),'lr':head_lr}]
    if encoder_lr>0:parameters.append({'params':model.encoder.parameters(),'lr':encoder_lr})
    optimizer=torch.optim.Adam(parameters,weight_decay=1e-4)
    pools=[np.flatnonzero(y==c) for c in range(4)];history=[];model.train()
    for step in range(steps):
        ids=np.concatenate([rng.choice(pool,16,replace=len(pool)<16) for pool in pools])
        loss=F.cross_entropy(model(torch.from_numpy(x[ids])),torch.from_numpy(y[ids]),label_smoothing=label_smoothing)
        if encoder_lr>0 and settings['anchor']:
            loss=loss+settings['anchor']*sum(((p-anchor[name])**2).sum() for name,p in model.encoder.named_parameters())
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step in {0,steps-1}:history.append({'step':step+1,'loss':float(loss.detach())})
    return model.eval(),history
