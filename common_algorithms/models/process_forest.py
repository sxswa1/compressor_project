"""Raw multivariate-window inference for a trained process random forest.

Requires numpy, pandas, joblib and scikit-learn; no torch dependency. Training
reuses common_algorithms.models.random_forest.train_random_forest. Features
reuse the canonical process_features implementation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from common_algorithms.feature_extraction.process_features import process_features
from common_algorithms.data_processing.process_clock import normalize_process_time


def execute(df: pd.DataFrame,value_columns,model_path,input_contract,
            window_id_column="window_id",time_column="timestamp",minimum_model_score=None,**kwargs):
    """out_1 prediction/window; out_2 model info. Inference never fits.

    Declare exact channel names, units, quantities, aggregation, cadence and
    length using the resource's multivariate_process contract. Missing values,
    incomplete/discontinuous windows and incompatible resources raise. Scores
    are uncalibrated model outputs. minimum_model_score optionally returns
    unresolved below an independently chosen validation threshold; it does not
    guarantee unknown-fault detection. Default: closed-set classification.
    Only load trusted joblib resources.
    """
    if minimum_model_score is not None and (isinstance(minimum_model_score,bool) or not np.isfinite(minimum_model_score) or not 0<=minimum_model_score<=1):
        raise ValueError('minimum_model_score must be None or a finite number in [0,1]')
    if not isinstance(input_contract,dict) or input_contract.get('signal_type')!='multivariate_process':
        raise ValueError("An explicit multivariate_process contract is required")
    values=[value_columns] if isinstance(value_columns,str) else list(value_columns)
    if values!=[c['name'] for c in input_contract.get('channels',[])]:
        raise ValueError("Channel order differs from contract")
    if not isinstance(df,pd.DataFrame) or not df.columns.is_unique or any(c not in df for c in values+[window_id_column,time_column]):
        raise ValueError("Missing or duplicate input columns")
    if set(values)&{window_id_column,time_column} or window_id_column==time_column or df[window_id_column].isna().any():
        raise ValueError("Distinct signal/time/window columns with nonmissing window IDs required")
    if not model_path:
        raise ValueError("model_path must be supplied")
    resource=joblib.load(Path(model_path))
    if resource.get('process_contract')!=input_contract or not isinstance(resource.get('model'),RandomForestClassifier):
        raise ValueError("Process contract or model kind differs from this operator")
    mean=np.asarray(resource['scaler']['mean'],dtype=np.float32)
    scale=np.asarray(resource['scaler']['scale'],dtype=np.float32)
    if mean.shape!=(len(values),) or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError("Invalid resource scaler")
    arrays,records=[],[]
    for key,part in df.groupby(window_id_column,sort=False):
        part=part.copy()
        part[time_column]=normalize_process_time(part[time_column],input_contract.get('source_timezone'))
        part=part.sort_values(time_column)
        delta=part[time_column].diff().dt.total_seconds().iloc[1:].to_numpy()
        if len(part)!=input_contract['window_points'] or part[time_column].isna().any() or not np.allclose(delta,input_contract['interval_seconds'],rtol=0,atol=1e-6):
            raise ValueError("Incomplete, duplicate or discontinuous process window")
        arrays.append(part[values].to_numpy(dtype=np.float32).T)
        records.append({window_id_column:key,'start_time':part[time_column].iloc[0],'end_time':part[time_column].iloc[-1]})
    a=np.stack(arrays) if arrays else np.empty((0,len(values),input_contract['window_points']),dtype=np.float32)
    scaled=(a-mean[None,:,None])/scale[None,:,None]
    features,names=process_features(scaled,values,input_contract['interval_seconds'])
    if names!=resource['feature_columns']:
        raise ValueError("Process feature schema differs from resource")
    selected=resource.get('selected_feature_indices')
    if selected is not None:
        selected=np.asarray(selected)
        if selected.ndim!=1 or not np.issubdtype(selected.dtype,np.integer) or not len(selected) or len(np.unique(selected))!=len(selected) or selected.min()<0 or selected.max()>=len(names):
            raise ValueError('Invalid saved feature selection')
        features=features[:,selected]
    if features.shape[1]!=resource['model'].n_features_in_:
        raise ValueError('Saved forest and feature selection dimensions disagree')
    scores=resource['model'].predict_proba(features) if len(a) else np.empty((0,len(resource['class_names'])))
    pred=resource['model'].classes_[scores.argmax(axis=1)].astype(int) if len(a) else np.empty(0,dtype=int)
    result_df=pd.DataFrame(records,columns=[window_id_column,'start_time','end_time'])
    result_df['predicted_label']=[resource['class_names'][i] for i in pred]
    result_df['model_score']=scores.max(axis=1) if len(a) else pd.Series(dtype=float)
    accepted=np.ones(len(result_df),dtype=bool) if minimum_model_score is None else result_df.model_score.to_numpy()>=minimum_model_score
    result_df['diagnosis_status']=np.where(accepted,'classified','below_configured_score_threshold')
    result_df.loc[~accepted,'predicted_label']='unresolved'
    info_df=pd.DataFrame([{'kind':'process_random_forest','n_windows':len(a),'scores_calibrated':False,
        'minimum_model_score':minimum_model_score}])
    return {'out_1':result_df,'out_2':info_df}
