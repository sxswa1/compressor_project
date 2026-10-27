"""Label-free scene inference. Invalid windows remain visible as NaN outputs."""
import pickle
import numpy as np
import pandas as pd
from data import ROOT, read, raw_windows, features, tabular_features, risk_levels
from common_algorithms.models.lstm import LSTMRegressor
from common_algorithms.models.gradient_boosting import GradientBoostingRegressor

def load_model(path, name):
    if name=='lstm_window':
        return LSTMRegressor.load(path)
    if name=='gradient_boosting_history':
        return GradientBoostingRegressor.load(path)
    with open(path,'rb') as f:
        return pickle.load(f)

def predict_array(model, X):
    if isinstance(model,dict) and model.get('format')=='constant_v1':
        return np.full(len(X),model['value'],dtype=float)
    return model.predict(X)

def predict_scene(task, name, seed, observations, timestamps, variant='raw_original', root=ROOT):
    if task not in ('di','rul'):
        raise ValueError('task must be di or rul')
    meta = read(root/'results/preprocessing.json')
    sensors = meta['sensors']
    asset = f'{name}__{seed}' if task=='di' else f'{name}__{variant}__{seed}'
    ext = '.pt' if name=='lstm_window' else '.pkl'
    model = load_model(root/'models'/task/(asset+ext),name)
    X, ends = raw_windows(observations,timestamps,sensors)
    if task=='rul' and variant.startswith('residual_'):
        with (root/'models/rul/reference.pkl').open('rb') as f:
            reference = pickle.load(f)
        F,rends = features(observations,timestamps,sensors,reference,meta)
        np.testing.assert_array_equal(ends,rends)
    else:
        F = X if name=='lstm_window' else (X[:,-1] if name.endswith('_current') else tabular_features(X))
    result = pd.DataFrame(dict(step=observations.index,valid=False,raw_prediction=np.nan,prediction=np.nan))
    if task=='di':
        result['risk_level'] = np.nan
    if len(ends):
        raw = predict_array(model,F)
        assert np.isfinite(raw).all()
        pred = np.clip(raw,0,1) if task=='di' else np.maximum(raw,0)
        result.loc[ends,['valid','raw_prediction','prediction']] = pd.DataFrame(
            dict(valid=True,raw_prediction=raw,prediction=pred),index=ends)
        if task=='di':
            result.loc[ends,'risk_level'] = risk_levels(pred)
    radii_path = root/'models/rul/interval_radii.json'
    if task=='rul' and radii_path.exists():
        radius = read(radii_path).get(asset)
        if radius:
            q = float('inf') if radius['radius_hours'] is None else radius['radius_hours']
            result['lower_hours'] = np.maximum(0,result.prediction-q)
            result['upper_hours'] = result.prediction+q
    return result
