"""RUL-specific internal stopping: hour-scale scene RMSE, no DI clipping."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from common_algorithms.models.lstm import LSTMRegressor


def select_epochs(X,y,V,vkeys,params,policy):
    model=LSTMRegressor(**params,epochs=policy['max_epochs'])
    X,V=model._input(X),model._input(V)
    y=np.asarray(y,dtype=np.float32)
    truth=vkeys.RUL_ref_hours.to_numpy(dtype=float)
    assert y.shape==(len(X),) and np.isfinite(y).all() and (y>=0).all()
    assert len(truth)==len(V) and np.isfinite(truth).all() and (truth>=0).all()
    assert X.shape[1:]==V.shape[1:]
    model.window_,model.n_features_in_=X.shape[1:]
    model.mean_=X.mean(axis=(0,1),dtype=np.float64).astype(np.float32)
    model.scale_=X.std(axis=(0,1),dtype=np.float64).astype(np.float32)
    model.scale_[model.scale_<1e-6]=1.0
    model.target_mean_=float(y.mean(dtype=np.float64))
    model.target_scale_=max(float(y.std(dtype=np.float64)),1e-6)
    xt=torch.from_numpy(np.ascontiguousarray((X-model.mean_)/model.scale_))
    yt=torch.from_numpy((y-model.target_mean_)/model.target_scale_)
    ids,_=pd.factorize(vkeys.scene_id,sort=True)
    counts=np.bincount(ids)
    model.loss_history_=[]
    trace,best,stale,chosen,state=[],float('inf'),0,None,None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(params['random_state'])
        model.model_=model._new_network()
        optimizer=torch.optim.Adam(model.model_.parameters(),lr=params['learning_rate'])
        for epoch in range(1,policy['max_epochs']+1):
            model.model_.train()
            total=0.0
            for index in torch.randperm(len(X)).split(params['batch_size']):
                optimizer.zero_grad()
                loss=nn.functional.mse_loss(model.model_(xt[index]),yt[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.model_.parameters(),1.0)
                optimizer.step()
                total+=float(loss.detach())*len(index)
            model.loss_history_.append(total/len(X))
            pred=np.maximum(0,model.predict(V))
            metric=float(np.sqrt(np.bincount(ids,weights=(pred-truth)**2)/counts).mean())
            if not np.isfinite(metric):
                raise FloatingPointError('nonfinite selection metric')
            improved=metric<best-policy['min_delta_hours']
            if improved:
                best,stale,chosen=metric,0,epoch
                state={k:v.detach().clone() for k,v in model.model_.state_dict().items()}
            else:
                stale+=1
            stopped=epoch>=policy['minimum_epochs'] and stale>=policy['patience']
            trace.append(dict(epoch=epoch,normalized_train_mse=model.loss_history_[-1],
                inner_scene_rmse_hours=metric,accepted_improvement=improved,
                selected_epoch_so_far=chosen,stale_epochs=stale,stopped_by_patience=stopped))
            if stopped:
                break
    model.model_.load_state_dict(state)
    model.model_.eval()
    model.params['epochs']=chosen
    model.loss_history_=model.loss_history_[:chosen]
    return model,trace
