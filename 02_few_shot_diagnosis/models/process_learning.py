"""Supervised, episodic and transfer training for multivariate process data.

No query labels or query normalization are used during fitting. The caller
defines independent source records and verifies the support/query separation.
"""
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common_algorithms.models.process_cnn import ProcessEncoder,ProcessClassifier,validate_windows,embed_process


def _labels(x,y):
    x=validate_windows(x)
    y=np.asarray(y)
    if y.ndim!=1 or len(x)!=len(y) or not len(x):
        raise ValueError("Nonempty windows and aligned labels are required")
    classes=np.unique(y)
    if len(classes)<2 or not np.array_equal(classes,np.arange(len(classes))):
        raise ValueError("Training labels must be contiguous integers starting at zero, with >=2 classes")
    return x,y.astype(np.int64),len(classes)


def train_process_classifier(x,y,seed=42,steps=200,lr=.002,batch_size=64,encoder=None,head_seed=None):
    x,y,n_classes=_labels(x,y)
    if steps<1 or lr<=0 or batch_size<1:
        raise ValueError("Invalid training configuration")
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    # An independent copy prevents a target adaptation from modifying the shared
    # source encoder or contaminating a later support-set task.
    encoder=copy.deepcopy(encoder) if encoder is not None else ProcessEncoder(x.shape[1])
    if head_seed is not None:
        # Matched-head comparisons must not depend on how many RNG draws the
        # source encoder's construction consumed before initializing the head.
        torch.manual_seed(head_seed)
    model=ProcessClassifier(encoder,n_classes)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    pools=[np.flatnonzero(y==c) for c in range(n_classes)]
    per_class=max(1,batch_size//n_classes)
    history=[]
    model.train()
    for step in range(steps):
        ids=np.concatenate([rng.choice(pool,per_class,replace=len(pool)<per_class) for pool in pools])
        logits=model(torch.from_numpy(x[ids]))
        loss=F.cross_entropy(logits,torch.from_numpy(y[ids]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step in {0,steps-1}:
            history.append({"step":step+1,"loss":float(loss.detach())})
    return model.eval(),history


def train_process_protonet(x,y,seed=42,episodes=250,support=2,query=3,ways=None,lr=.002,record_groups=None):
    x,y,n_classes=_labels(x,y)
    if episodes<1 or support<1 or query<1 or lr<=0:
        raise ValueError("Invalid episodic training configuration")
    ways=n_classes if ways is None else ways
    if ways<2 or ways>n_classes:
        raise ValueError("Invalid number of episode classes")
    pools=[np.flatnonzero(y==c) for c in range(n_classes)]
    if any(len(p)<support+query for p in pools):
        raise ValueError("Not enough distinct base records for disjoint support/query")
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    encoder=ProcessEncoder(x.shape[1])
    optimizer=torch.optim.Adam(encoder.parameters(),lr=lr,weight_decay=1e-4)
    history=[]
    encoder.train()
    for episode in range(episodes):
        classes=rng.choice(n_classes,ways,replace=False)
        if record_groups is None:
            ids=np.stack([rng.choice(pools[c],support+query,replace=False) for c in classes])
        else:
            groups=np.asarray(record_groups)
            if groups.shape!=(len(x),):
                raise ValueError('record_groups must align with source windows')
            picks=[]
            for c in classes:
                available=np.unique(groups[pools[c]])
                if len(available)<support+query:
                    raise ValueError('Insufficient distinct groups for episodic support/query')
                selected=rng.choice(available,support+query,replace=False)
                picks.append([rng.choice(pools[c][groups[pools[c]]==g]) for g in selected])
            ids=np.asarray(picks)
        e=F.normalize(encoder(torch.from_numpy(x[ids.reshape(-1)])),dim=1)
        e=e.reshape(ways,support+query,-1)
        proto=F.normalize(e[:,:support].mean(dim=1),dim=1)
        logits=10*e[:,support:].reshape(-1,e.shape[-1])@proto.T
        targets=torch.arange(ways).repeat_interleave(query)
        loss=F.cross_entropy(logits,targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(),5.)
        optimizer.step()
        if episode in {0,episodes-1}:
            history.append({"episode":episode+1,"loss":float(loss.detach())})
    return encoder.eval(),history


def train_multitask_source(x,label_tasks,record_groups,seed=42,steps=200,episodes=250,lr=.002):
    """Compare supervised multihead pretraining and multitask ProtoNet.

    Both consume the same source records and component-label annotations.
    Each meta episode selects a source component task; support/query use
    distinct complete condition groups. Target component labels are excluded.
    """
    x=validate_windows(x)
    tasks=[]
    for name,labels in label_tasks.items():
        _,y,n=_labels(x,labels)
        tasks.append((name,y,n))
    if not tasks:
        raise ValueError('At least one source label task is required')
    groups=np.asarray(record_groups)
    if groups.shape!=(len(x),):
        raise ValueError('Source condition groups must align with windows')
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    supervised=ProcessEncoder(x.shape[1])
    heads=nn.ModuleList([nn.Linear(supervised.architecture['embedding_dim'],n) for _,_,n in tasks])
    optimizer=torch.optim.Adam(list(supervised.parameters())+list(heads.parameters()),lr=lr,weight_decay=1e-4)
    history=[]
    for step in range(steps):
        ids=rng.choice(len(x),min(64,len(x)),replace=False)
        emb=supervised(torch.from_numpy(x[ids]))
        loss=sum(F.cross_entropy(head(emb),torch.from_numpy(y[ids])) for head,(_,y,_) in zip(heads,tasks))/len(tasks)
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(list(supervised.parameters())+list(heads.parameters()),5.)
        optimizer.step()
        if step in {0,steps-1}:
            history.append({'step':step+1,'multitask_supervised_loss':float(loss.detach())})
    torch.manual_seed(seed)
    meta=ProcessEncoder(x.shape[1])
    optimizer=torch.optim.Adam(meta.parameters(),lr=lr,weight_decay=1e-4)
    meta_history=[]
    for episode in range(episodes):
        name,y,n=tasks[int(rng.integers(len(tasks)))]
        picks=[]
        for c in range(n):
            pool=np.flatnonzero(y==c)
            available=np.unique(groups[pool])
            if len(available)<5:
                raise ValueError('Source task lacks five distinct groups per class')
            chosen=rng.choice(available,5,replace=False)
            picks.append([rng.choice(pool[groups[pool]==g]) for g in chosen])
        ids=np.asarray(picks)
        e=F.normalize(meta(torch.from_numpy(x[ids.reshape(-1)])),dim=1).reshape(n,5,-1)
        proto=F.normalize(e[:,:2].mean(dim=1),dim=1)
        logits=10*e[:,2:].reshape(-1,e.shape[-1])@proto.T
        loss=F.cross_entropy(logits,torch.arange(n).repeat_interleave(3))
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(meta.parameters(),5.)
        optimizer.step()
        if episode in {0,episodes-1}:
            meta_history.append({'episode':episode+1,'component_task':name,'loss':float(loss.detach())})
    return supervised.eval(),meta.eval(),history,meta_history


def process_prototypes(encoder,support_x,support_y):
    x,y,n_classes=_labels(support_x,support_y)
    e=embed_process(encoder,x)
    return F.normalize(torch.stack([e[torch.from_numpy(y==c)].mean(dim=0) for c in range(n_classes)]),dim=1)


def predict_process_classifier(model,x):
    a=validate_windows(x,model.encoder.architecture['channels'])
    model.eval()
    with torch.no_grad():
        logits=torch.cat([model(torch.from_numpy(a[i:i+128])) for i in range(0,len(a),128)]) if len(a) else torch.empty((0,model.head.out_features))
    return logits.softmax(dim=1).numpy()


def predict_process_prototypes(encoder,prototypes,x):
    return (10*embed_process(encoder,x)@prototypes.T).softmax(dim=1).numpy()
