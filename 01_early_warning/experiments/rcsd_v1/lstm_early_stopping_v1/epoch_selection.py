"""Research-only epoch selection; accepts inner fit/validation data, never outer data.

Uses the existing shared network, normalization and Adam training recipe.
The public LSTM implementation and its saved six-epoch results remain unchanged.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from common_algorithms.models.lstm import LSTMRegressor


def select_epochs(X, y, validation, validation_keys, params, policy):
    model = LSTMRegressor(**params, epochs=policy['max_epochs'])
    X = model._input(X)
    validation = model._input(validation)
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError('finite inner-fit target required')
    if validation.shape[1:] != X.shape[1:] or len(validation_keys) != len(validation):
        raise ValueError('inner validation dimensions do not match')
    truth = validation_keys.DI_ref.to_numpy(dtype=float)
    if not np.isfinite(truth).all():
        raise ValueError('finite inner-validation target required')
    model.window_, model.n_features_in_ = X.shape[1:]
    model.mean_ = X.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    model.scale_ = X.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    model.scale_[model.scale_ < 1e-6] = 1.0
    model.target_mean_ = float(y.mean(dtype=np.float64))
    model.target_scale_ = max(float(y.std(dtype=np.float64)), 1e-6)
    x_tensor = torch.from_numpy(np.ascontiguousarray((X-model.mean_)/model.scale_))
    y_tensor = torch.from_numpy((y-model.target_mean_)/model.target_scale_)
    scene_ids, unique = pd.factorize(validation_keys.scene_id, sort=True)
    counts = np.bincount(scene_ids)
    model.loss_history_ = []
    trace, best, stale, selected_epoch, selected_state = [], float('inf'), 0, None, None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(params['random_state'])
        model.model_ = model._new_network()
        optimizer = torch.optim.Adam(model.model_.parameters(), lr=params['learning_rate'])
        for epoch in range(1, policy['max_epochs']+1):
            model.model_.train()
            order = torch.randperm(len(X))
            total = 0.0
            for indices in order.split(params['batch_size']):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model.model_(x_tensor[indices]), y_tensor[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite inner-fit loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.model_.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            model.loss_history_.append(total / len(X))
            pred = np.clip(model.predict(validation), 0, 1)
            metric = float((np.bincount(scene_ids, weights=np.abs(pred-truth))/counts).mean())
            if not np.isfinite(metric):
                raise FloatingPointError('nonfinite selection metric')
            improved = metric < best-policy['min_delta']
            if improved:
                best, stale, selected_epoch = metric, 0, epoch
                selected_state = {k:v.detach().clone() for k,v in model.model_.state_dict().items()}
            else:
                stale += 1
            stopped = epoch >= policy['minimum_epochs'] and stale >= policy['patience']
            trace.append(dict(epoch=epoch,normalized_train_mse=model.loss_history_[-1],
                              inner_scene_mae=metric,accepted_improvement=improved,
                              selected_epoch_so_far=selected_epoch,stale_epochs=stale,
                              stopped_by_patience=stopped))
            if stopped:
                break
    model.model_.load_state_dict(selected_state)
    model.model_.eval()
    model.params['epochs'] = selected_epoch
    model.loss_history_ = model.loss_history_[:selected_epoch]
    return model, trace
