"""Strict standalone inference for source-fitted RCSD metric prototypes.

Source geometry and labelled target prototypes are already fitted. Inference
accepts one complete process window; it does not need a paired normal window.
Dependencies: numpy, pandas, torch. Scores are uncalibrated similarities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common_algorithms.data_processing.process_clock import normalize_process_time
from common_algorithms.feature_extraction.process_features import process_features
from common_algorithms.models.process_cnn import validate_contract,validate_windows,ChannelStandardizer,ProcessEncoder


def predict_metric_resource(resource,x,input_contract):
    contract=validate_contract(input_contract)
    if resource.get('kind')!='rcsd_metric_prototype' or resource.get('version')!=2 or resource.get('process_contract')!=contract:
        raise ValueError('Metric resource kind/schema/units/aggregation/cadence/window differs')
    x=validate_windows(x,len(contract['channels']),contract['window_points'])
    classes=resource.get('class_names')
    if not isinstance(classes,list) or len(classes)!=4 or len(set(classes))!=4 or any(not isinstance(c,str) or not c for c in classes):
        raise ValueError('Exactly four distinct known state names required')
    geometry=resource['geometry'];center=np.asarray(geometry['center'],dtype=float);scale=np.asarray(geometry['scale'],dtype=float)
    precision=np.asarray(geometry['precision'],dtype=float);proto=np.asarray(resource['prototypes'],dtype=float);dim=len(center)
    if center.ndim!=1 or not dim or scale.shape!=center.shape or precision.shape!=(dim,dim) or proto.shape!=(4,dim):
        raise ValueError('Invalid saved geometry dimensions')
    if not all(np.isfinite(v).all() for v in [center,scale,precision,proto]) or (scale<=0).any():
        raise ValueError('Saved geometry must be finite with positive scales')
    if not np.allclose(precision,precision.T,rtol=0,atol=1e-8) or np.linalg.eigvalsh(precision).min()<=0:
        raise ValueError('Saved metric must be symmetric positive definite')
    family=resource.get('representation_family')
    if family not in {'statistics','meta_raw','supervised_raw'}:raise ValueError('Unsupported representation family')
    a=ChannelStandardizer.from_state(resource['scaler']).transform(x)
    if family=='statistics':
        expected_dim=7*len(contract['channels'])
        features=process_features(a,[c['name'] for c in contract['channels']],contract['interval_seconds'])[0]
    else:
        encoder=ProcessEncoder(**resource['architecture']);encoder.load_state_dict(resource['encoder_state']);encoder.eval()
        if encoder.architecture['channels']!=len(contract['channels']):raise ValueError('Saved encoder channels differ')
        expected_dim=encoder.architecture['embedding_dim'];rows=[]
        with torch.no_grad():
            for begin in range(0,len(a),128):rows.append(encoder(torch.from_numpy(a[begin:begin+128])).numpy())
        features=np.concatenate(rows).astype(float) if rows else np.empty((0,expected_dim))
    if dim!=expected_dim:raise ValueError('Saved geometry dimension differs from representation')
    delta=((features-center)/scale)[:,None,:]-proto[None,:,:]
    distances=np.einsum('ncd,de,nce->nc',delta,precision,delta,optimize=True)
    logits=-np.maximum(distances,0)/dim
    logits-=logits.max(axis=1,keepdims=True);scores=np.exp(logits);scores/=scores.sum(axis=1,keepdims=True)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite metric output')
    return scores.argmax(axis=1) if len(a) else np.empty(0,dtype=int),scores


def execute(df:pd.DataFrame,value_columns,model_path,input_contract,window_id_column='window_id',
            time_column='timestamp',minimum_model_score=None,**kwargs):
    """Return out_1 diagnosis/window and out_2 resource metadata; never train.

    Required channels, units, aggregation, 900-second cadence and 20-point
    windows must match this resource's saved contract. A threshold can mark
    unresolved outputs, but must be independently validated by the caller.
    """
    if minimum_model_score is not None and (isinstance(minimum_model_score,bool) or not np.isfinite(minimum_model_score) or not 0<=minimum_model_score<=1):
        raise ValueError('minimum_model_score must be finite in [0,1]')
    contract=validate_contract(input_contract);values=[value_columns] if isinstance(value_columns,str) else list(value_columns)
    if values!=[c['name'] for c in contract['channels']]:raise ValueError('Signal columns/order differ from contract')
    if not isinstance(df,pd.DataFrame) or not df.columns.is_unique or any(c not in df for c in values+[window_id_column,time_column]):
        raise ValueError('Missing or duplicate required columns')
    if set(values)&{window_id_column,time_column} or window_id_column==time_column or df[window_id_column].isna().any():
        raise ValueError('Distinct signal/time/nonmissing window identifiers required')
    if not model_path:raise ValueError('Explicit trusted model path required')
    resource=torch.load(Path(model_path),map_location='cpu',weights_only=True)
    arrays=[];records=[]
    for key,part in df.groupby(window_id_column,sort=False):
        part=part.copy();part[time_column]=normalize_process_time(part[time_column],contract.get('source_timezone'));part=part.sort_values(time_column)
        delta=part[time_column].diff().dt.total_seconds().iloc[1:].to_numpy()
        if len(part)!=contract['window_points'] or part[time_column].isna().any() or not np.allclose(delta,contract['interval_seconds'],rtol=0,atol=1e-6):
            raise ValueError('Incomplete, duplicate or discontinuous process window')
        arrays.append(part[values].to_numpy(dtype=np.float32).T)
        records.append({window_id_column:key,'start_time':part[time_column].iloc[0],'end_time':part[time_column].iloc[-1]})
    x=np.stack(arrays) if arrays else np.empty((0,len(values),contract['window_points']),dtype=np.float32)
    pred,scores=predict_metric_resource(resource,x,contract)
    result=pd.DataFrame(records,columns=[window_id_column,'start_time','end_time'])
    result['predicted_label']=[resource['class_names'][c] for c in pred];result['model_score']=scores.max(axis=1) if len(scores) else pd.Series(dtype=float)
    accepted=np.ones(len(result),dtype=bool) if minimum_model_score is None else result.model_score.to_numpy()>=minimum_model_score
    result['diagnosis_status']=np.where(accepted,'classified','below_configured_score_threshold');result.loc[~accepted,'predicted_label']='unresolved'
    info=pd.DataFrame([{'kind':resource['kind'],'model_path':str(model_path),'n_windows':len(x),'signal_type':'multivariate_process',
        'scores_calibrated':False,'minimum_model_score':minimum_model_score,'requires_query_normal_reference':False}])
    return {'out_1':result,'out_2':info}
