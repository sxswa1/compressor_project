"""CPU, unidirectional LSTM scalar regression with fit-only normalization.

X shape: [samples, time, features]. y shape: [samples]. Windows must be
constructed causally by the caller; this module does not infer labels or split data.
There is no hidden state carried between samples/calls, nor implicit early stopping.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _Network(nn.Module):
    def __init__(self, features, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, X):
        sequence, _ = self.lstm(X)
        return self.head(sequence[:, -1, :]).squeeze(-1)


class LSTMRegressor:
    def __init__(self, hidden_size=24, num_layers=1, epochs=6, batch_size=128,
                 learning_rate=0.001, random_state=20261023):
        for name, value in dict(hidden_size=hidden_size, num_layers=num_layers,
                                epochs=epochs, batch_size=batch_size).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        self.params = dict(hidden_size=hidden_size, num_layers=num_layers, epochs=epochs,
                           batch_size=batch_size, learning_rate=learning_rate,
                           random_state=random_state)

    @staticmethod
    def _input(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError('X must be a nonempty finite [samples, time, features] array')
        return X

    def _new_network(self):
        return _Network(self.n_features_in_, self.params['hidden_size'], self.params['num_layers'])

    def fit(self, X, y):
        X = self._input(X)
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError('y must be a finite scalar target for each sample')
        self.window_, self.n_features_in_ = X.shape[1:]
        self.mean_ = X.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.scale_ = X.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.scale_[self.scale_ < 1e-6] = 1.0
        self.target_mean_ = float(y.mean(dtype=np.float64))
        self.target_scale_ = max(float(y.std(dtype=np.float64)), 1e-6)
        x_tensor = torch.from_numpy(np.ascontiguousarray((X-self.mean_)/self.scale_))
        y_tensor = torch.from_numpy((y-self.target_mean_)/self.target_scale_)
        self.loss_history_ = []
        # Do not consume/change the caller's torch random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.params['random_state'])
            self.model_ = self._new_network()
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.params['learning_rate'])
            self.model_.train()
            for _ in range(self.params['epochs']):
                order = torch.randperm(len(X))
                total = 0.0
                for indices in order.split(self.params['batch_size']):
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(self.model_(x_tensor[indices]), y_tensor[indices])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('non-finite training loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(indices)
                self.loss_history_.append(total / len(X))
        self.model_.eval()
        return self

    def predict(self, X):
        if not hasattr(self, 'model_'):
            raise RuntimeError('fit or load a model before prediction')
        X = self._input(X)
        if X.shape[1:] != (self.window_, self.n_features_in_):
            raise ValueError('window or feature dimension differs from training')
        self.model_.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(X), self.params['batch_size']):
                batch = np.ascontiguousarray((X[start:start+self.params['batch_size']]-self.mean_)/self.scale_)
                predictions.append(self.model_(torch.from_numpy(batch)).numpy())
        return np.concatenate(predictions) * self.target_scale_ + self.target_mean_

    def save(self, path):
        if not hasattr(self, 'model_'):
            raise RuntimeError('cannot save an unfitted model')
        torch.save(dict(format='lstm_regressor_v1', params=self.params, window=self.window_,
                        features=self.n_features_in_, mean=torch.from_numpy(self.mean_),
                        scale=torch.from_numpy(self.scale_), target_mean=self.target_mean_,
                        target_scale=self.target_scale_, loss_history=self.loss_history_,
                        state_dict=self.model_.state_dict()), Path(path))

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != 'lstm_regressor_v1':
            raise ValueError('unsupported LSTM artifact')
        result = cls(**payload['params'])
        result.window_, result.n_features_in_ = payload['window'], payload['features']
        result.mean_, result.scale_ = payload['mean'].numpy(), payload['scale'].numpy()
        result.target_mean_, result.target_scale_ = payload['target_mean'], payload['target_scale']
        result.loss_history_ = payload['loss_history']
        with torch.random.fork_rng(devices=[]):
            result.model_ = result._new_network()
        result.model_.load_state_dict(payload['state_dict'])
        result.model_.eval()
        return result
