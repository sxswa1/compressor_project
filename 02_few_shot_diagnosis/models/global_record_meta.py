"""Episodic source learning with globally separated original records.

A record may contain normal and faulty windows. Its role is assigned once
per episode across every selected class. No class, record or requested shot
is silently dropped to make a difficult episode feasible.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import Bounds,LinearConstraint,milp
from common_algorithms.models.process_cnn import ProcessEncoder
from .process_learning import _labels


class GlobalRecordEpisodeSampler:
    def __init__(self,labels,record_groups):
        self.labels=np.asarray(labels)
        raw=np.asarray(record_groups)
        if self.labels.ndim!=1 or raw.shape!=self.labels.shape or not len(raw) or pd.isna(raw).any():
            raise ValueError('Aligned nonmissing original record IDs are required')
        self.groups=raw.astype(str)
        if any(not value.strip() for value in self.groups):raise ValueError('Original record IDs must be nonempty')
        if not np.issubdtype(self.labels.dtype,np.number) or not np.isfinite(self.labels).all():
            raise ValueError('Finite numeric source classes are required')
        classes=np.unique(self.labels)
        if not np.array_equal(classes,np.arange(len(classes))) or len(classes)<2:
            raise ValueError('Contiguous source classes starting at zero are required')
        self.pools={int(c):np.flatnonzero(self.labels==c) for c in classes}
        self.available={c:np.unique(self.groups[pool]) for c,pool in self.pools.items()}
        self.by_class_record={(c,g):pool[self.groups[pool]==g] for c,pool in self.pools.items() for g in self.available[c]}
        self.homogeneous=not pd.DataFrame({'group':self.groups,'label':self.labels}).groupby('group').label.nunique().gt(1).any()

    def sample(self,classes,support,query,rng):
        classes=np.asarray(classes)
        if classes.ndim!=1 or not np.issubdtype(classes.dtype,np.integer) or len(classes)<2 or len(np.unique(classes))!=len(classes) or any(int(c) not in self.pools for c in classes):
            raise ValueError('Select unique observed episode classes')
        if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in [support,query]):
            raise ValueError('Positive integer support/query record counts are required')
        if any(len(self.available[int(c)])<support+query for c in classes):
            raise ValueError('Insufficient distinct original records for the requested episode shots')
        selected=[];solver_used=not self.homogeneous
        if self.homogeneous:
            # Exact legacy RNG order for the already frozen one-class/record
            # studies: choose class groups, then one window from each group.
            for c in classes:
                c=int(c);chosen=rng.choice(self.available[c],support+query,replace=False)
                selected.append([rng.choice(self.by_class_record[(c,g)]) for g in chosen])
        else:
            active=np.unique(np.concatenate([self.available[int(c)] for c in classes]))
            presence=np.asarray([[g in self.available[int(c)] for g in active] for c in classes],dtype=float)
            upper=presence.sum(axis=1)-query
            # Binary role=1 means support, role=0 means query for every class.
            # Random source-only costs diversify feasible partitions; they are
            # not diagnostic losses and never contain target/query values.
            result=milp(c=rng.uniform(-1.,1.,len(active)),integrality=np.ones(len(active)),
                bounds=Bounds(np.zeros(len(active)),np.ones(len(active))),
                constraints=LinearConstraint(presence,np.full(len(classes),support,dtype=float),upper),
                options={'time_limit':2.,'node_limit':10000,'mip_rel_gap':0.})
            if result.status==2:raise ValueError('A globally disjoint support/query episode is infeasible for these classes and records')
            if result.status!=0 or result.x is None:
                raise RuntimeError('Bounded episode role assignment did not finish; no class or shot was dropped')
            role=np.rint(result.x).astype(int)
            if np.any(np.abs(result.x-role)>1e-6) or np.any((role<0)|(role>1)) or np.any(presence@role<support) or np.any(presence@role>upper):
                raise RuntimeError('The solver did not provide a verified integral episode partition')
            role_map=dict(zip(active,role))
            for c in classes:
                c=int(c);support_pool=[g for g in self.available[c] if role_map[g]==1]
                query_pool=[g for g in self.available[c] if role_map[g]==0]
                chosen=list(rng.choice(support_pool,support,replace=False))+list(rng.choice(query_pool,query,replace=False))
                selected.append([rng.choice(self.by_class_record[(c,g)]) for g in chosen])
        ids=np.asarray(selected,dtype=int)
        support_groups=set(self.groups[ids[:,:support]].reshape(-1));query_groups=set(self.groups[ids[:,support:]].reshape(-1))
        if support_groups&query_groups:raise RuntimeError('An original record crossed episode roles')
        for c,row in zip(classes,ids):
            if not np.all(self.labels[row]==c) or len(np.unique(self.groups[row]))!=support+query:
                raise RuntimeError('Episode classes or per-class record counts changed')
        return ids,{'support_unique_records':len(support_groups),'query_unique_records':len(query_groups),
            'record_role_intersection':0,'mixed_record_solver_used':bool(solver_used)}


def train_global_record_protonet(x,y,record_groups,seed=42,episodes=350,support=2,query=3,ways=None,lr=.002):
    """Same cosine-prototype learner, with a global record role assignment."""
    x,y,n_classes=_labels(x,y)
    if isinstance(episodes,bool) or not isinstance(episodes,int) or episodes<1 or isinstance(lr,bool) or not isinstance(lr,(int,float)) or not np.isfinite(lr) or lr<=0:
        raise ValueError('Positive episodes and a finite learning rate are required')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:
        raise ValueError('An unsigned 32-bit integer source seed is required')
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in [support,query]):
        raise ValueError('Positive integer support/query record counts are required')
    ways=n_classes if ways is None else ways
    if isinstance(ways,bool) or not isinstance(ways,int) or not 2<=ways<=n_classes:
        raise ValueError('Valid episode ways are required')
    sampler=GlobalRecordEpisodeSampler(y,record_groups)
    if any(len(groups)<support+query for groups in sampler.available.values()):
        raise ValueError('Every source class needs enough original records for support plus query')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    encoder=ProcessEncoder(x.shape[1]);optimizer=torch.optim.Adam(encoder.parameters(),lr=lr,weight_decay=1e-4)
    history=[];encoder.train()
    for episode in range(episodes):
        classes=rng.choice(n_classes,ways,replace=False)
        ids,evidence=sampler.sample(classes,support,query,rng)
        embeddings=F.normalize(encoder(torch.from_numpy(x[ids.reshape(-1)])),dim=1).reshape(ways,support+query,-1)
        prototypes=F.normalize(embeddings[:,:support].mean(dim=1),dim=1)
        logits=10*embeddings[:,support:].reshape(-1,embeddings.shape[-1])@prototypes.T
        loss=F.cross_entropy(logits,torch.arange(ways).repeat_interleave(query))
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(encoder.parameters(),5.);optimizer.step()
        if episode in {0,episodes-1}:
            history.append({'episode':episode+1,'loss':float(loss.detach()),'global_record_roles_verified':True,**evidence})
    return encoder.eval(),history
