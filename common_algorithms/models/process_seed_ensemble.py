"""Equal-weight independent-seed ensemble, fixed before target scoring.

Uses the same labelled support windows in every member. Additional seeds add
computation, not labelled observations or independent failures. Dependencies:
numpy, pandas, torch. Similarity scores are not calibrated probabilities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common_algorithms.data_processing.process_clock import normalize_process_time
from common_algorithms.models.process_cnn import validate_contract,validate_windows,predict_process_resource


def predict_seed_ensemble(resource,x,input_contract):
    contract=validate_contract(input_contract)
    if resource.get('kind')!='process_seed_ensemble' or resource.get('format_version')!=1 or resource.get('contract')!=contract:
        raise ValueError('Ensemble kind or physical contract differs')
    members=resource.get('members');names=resource.get('class_names');seeds=resource.get('train_seeds')
    if not isinstance(members,list) or len(members)!=3 or not isinstance(seeds,list) or len(set(seeds))!=3 or len(seeds)!=3:
        raise ValueError('Exactly three declared independent training seeds required')
    if not isinstance(names,list) or len(names)<2 or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
        raise ValueError('At least two distinct known class names required')
    if any(m.get('class_names')!=names or m.get('contract')!=contract for m in members):raise ValueError('Ensemble member mapping/schema differs')
    if len({m.get('kind') for m in members})!=1:raise ValueError('All ensemble members must use the same method')
    a=validate_windows(x,len(contract['channels']),contract['window_points'])
    scores=np.mean([predict_process_resource(m,a,contract)[1].astype(float) for m in members],axis=0)
    if scores.shape!=(len(a),len(names)) or not np.isfinite(scores).all():raise ValueError('Invalid ensemble member scores')
    return scores.argmax(axis=1) if len(a) else np.empty(0,dtype=int),scores


def execute(df:pd.DataFrame,value_columns,model_path,input_contract,window_id_column='window_id',time_column='timestamp',minimum_model_score=None,**kwargs):
    """out_1 prediction/window; out_2 metadata. No fitting or query adaptation."""
    if minimum_model_score is not None and (isinstance(minimum_model_score,bool) or not np.isfinite(minimum_model_score) or not 0<=minimum_model_score<=1):
        raise ValueError('minimum_model_score must be finite in [0,1]')
    contract=validate_contract(input_contract);values=[value_columns] if isinstance(value_columns,str) else list(value_columns)
    if values!=[c['name'] for c in contract['channels']]:raise ValueError('Signal columns/order differ')
    if not isinstance(df,pd.DataFrame) or not df.columns.is_unique or any(c not in df for c in values+[window_id_column,time_column]):raise ValueError('Missing or duplicate required columns')
    if set(values)&{window_id_column,time_column} or window_id_column==time_column or df[window_id_column].isna().any():raise ValueError('Distinct signal/time/nonmissing window identifiers required')
    if not model_path:raise ValueError('An explicit trusted resource path is required')
    resource=torch.load(Path(model_path),map_location='cpu',weights_only=True);arrays=[];records=[]
    for key,part in df.groupby(window_id_column,sort=False):
        part=part.copy();part[time_column]=normalize_process_time(part[time_column],contract.get('source_timezone'));part=part.sort_values(time_column)
        delta=part[time_column].diff().dt.total_seconds().iloc[1:].to_numpy()
        if len(part)!=contract['window_points'] or part[time_column].isna().any() or not np.allclose(delta,contract['interval_seconds'],rtol=0,atol=1e-6):raise ValueError('Incomplete, duplicate or discontinuous process window')
        arrays.append(part[values].to_numpy(dtype=np.float32).T);records.append({window_id_column:key,'start_time':part[time_column].iloc[0],'end_time':part[time_column].iloc[-1]})
    x=np.stack(arrays) if arrays else np.empty((0,len(values),contract['window_points']),dtype=np.float32)
    pred,scores=predict_seed_ensemble(resource,x,contract)
    result=pd.DataFrame(records,columns=[window_id_column,'start_time','end_time']);result['predicted_label']=[resource['class_names'][i] for i in pred]
    result['model_score']=scores.max(axis=1) if len(scores) else pd.Series(dtype=float)
    accepted=np.ones(len(result),dtype=bool) if minimum_model_score is None else result.model_score.to_numpy()>=minimum_model_score
    result['diagnosis_status']=np.where(accepted,'classified','below_configured_score_threshold');result.loc[~accepted,'predicted_label']='unresolved'
    return {'out_1':result,'out_2':pd.DataFrame([{'kind':resource['kind'],'n_windows':len(x),'member_count':3,'scores_calibrated':False,
        'same_support_labels_in_all_members':True,'minimum_model_score':minimum_model_score}])}
