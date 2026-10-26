"""Tabular gradient-boosting regression; no compressor-specific fields or rules.

Uses sklearn HistGradientBoostingRegressor, not ExtraTrees or random forest.
Early stopping is disabled: callers must supply chronological validation themselves.
"""
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GradientBoostingRegressor:
    def __init__(self, max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                 min_samples_leaf=20, l2_regularization=1.0, random_state=20261023):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                           l2_regularization=l2_regularization, random_state=random_state)

    @staticmethod
    def _input(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError('X must be a nonempty finite [samples, features] array')
        return X

    def fit(self, X, y):
        X = self._input(X)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError('y must be a finite scalar target for each sample')
        self.model_ = HistGradientBoostingRegressor(early_stopping=False, **self.params)
        self.model_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, 'model_'):
            raise RuntimeError('fit or load a model before prediction')
        X = self._input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('feature dimension differs from training')
        return self.model_.predict(X)

    def save(self, path):
        if not hasattr(self, 'model_'):
            raise RuntimeError('cannot save an unfitted model')
        with Path(path).open('wb') as stream:
            pickle.dump(dict(format='gradient_boosting_v1', params=self.params,
                             features=self.n_features_in_, model=self.model_), stream, protocol=5)

    @classmethod
    def load(cls, path):
        """Load a trusted local pickle, in a compatible sklearn environment."""
        with Path(path).open('rb') as stream:
            payload = pickle.load(stream)
        if payload.get('format') != 'gradient_boosting_v1':
            raise ValueError('unsupported gradient boosting artifact')
        result = cls(**payload['params'])
        result.model_, result.n_features_in_ = payload['model'], payload['features']
        return result
