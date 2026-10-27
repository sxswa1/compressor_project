"""Multichannel temporal CNN and strictly typed process-model inference.

Independent of a particular dataset. Input channels carry explicit names,
units, physical quantities, aggregation, cadence and window length. Acoustic
resources are incompatible. Dependencies: numpy, pandas, torch.
"""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from common_algorithms.data_processing.process_clock import normalize_process_time


def validate_contract(contract):
    if not isinstance(contract,dict) or contract.get("signal_type")!="multivariate_process":
        raise ValueError("An explicit multivariate_process input contract is required")
    channels=contract.get("channels")
    if not isinstance(channels,list) or not channels or any(not isinstance(c,dict) for c in channels):
        raise ValueError("Contract needs a nonempty channel schema")
    for c in channels:
        if any(not isinstance(c.get(k),str) or not c[k].strip() for k in ["name","unit","quantity","aggregation"]):
            raise ValueError("Every channel needs name, unit, quantity and aggregation")
    if len({c['name'] for c in channels})!=len(channels):
        raise ValueError("Duplicate contract channel names")
    step=contract.get("interval_seconds")
    length=contract.get("window_points")
    if isinstance(step,bool) or not isinstance(step,(int,float)) or not np.isfinite(step) or step<=0:
        raise ValueError("Contract interval_seconds must be positive")
    if isinstance(length,bool) or not isinstance(length,int) or length<2:
        raise ValueError("Contract window_points must be an integer >=2")
    return copy.deepcopy(contract)


def validate_windows(x,channels=None,points=None):
    a=np.asarray(x,dtype=np.float32)
    if a.ndim!=3 or a.shape[1]<1 or a.shape[2]<2 or not np.isfinite(a).all():
        raise ValueError("Expected finite N x channels x time process windows")
    if channels is not None and a.shape[1]!=channels:
        raise ValueError("Channel count differs from model")
    if points is not None and a.shape[2]!=points:
        raise ValueError("Window length differs from model")
    return a


class ChannelStandardizer:
    """Fit on permitted training/support data only, never query/test windows."""
    def fit(self,x):
        a=validate_windows(x)
        if len(a)==0:
            raise ValueError("Cannot fit standardizer on empty data")
        self.mean=a.astype(np.float64).mean(axis=(0,2)).astype(np.float32)
        scale=a.astype(np.float64).std(axis=(0,2)).astype(np.float32)
        self.scale=np.where(scale>1e-8,scale,1.).astype(np.float32)
        return self

    def transform(self,x):
        if not hasattr(self,"mean"):
            raise ValueError("Fit the standardizer before transform")
        a=validate_windows(x,len(self.mean))
        return ((a-self.mean[None,:,None])/self.scale[None,:,None]).astype(np.float32)

    def state(self):
        if not hasattr(self,"mean"):
            raise ValueError("Standardizer is not fitted")
        return {"mean":self.mean.tolist(),"scale":self.scale.tolist()}

    @classmethod
    def from_state(cls,state):
        obj=cls()
        obj.mean=np.asarray(state['mean'],dtype=np.float32)
        obj.scale=np.asarray(state['scale'],dtype=np.float32)
        if obj.mean.ndim!=1 or len(obj.mean)<1 or obj.scale.shape!=obj.mean.shape or not np.isfinite(obj.mean).all() or not np.isfinite(obj.scale).all() or np.any(obj.scale<=0):
            raise ValueError("Invalid saved standardizer")
        return obj


class ProcessEncoder(nn.Module):
    def __init__(self,channels,hidden=32,embedding_dim=32):
        super().__init__()
        if channels<1 or hidden<4 or hidden%4 or embedding_dim<2:
            raise ValueError("Invalid process encoder dimensions")
        self.architecture={"channels":int(channels),"hidden":int(hidden),"embedding_dim":int(embedding_dim)}
        self.temporal=nn.Sequential(nn.Conv1d(channels,hidden,5,padding=2),nn.GroupNorm(4,hidden),nn.ReLU(),
            nn.Conv1d(hidden,hidden,3,padding=1),nn.GroupNorm(4,hidden),nn.ReLU())
        self.project=nn.Linear(hidden+channels*2,embedding_dim)

    def forward(self,x):
        z=self.temporal(x).mean(dim=-1)
        # Absolute levels and slow trends matter for process signals. No
        # per-window normalization that would erase a persistent temperature rise.
        stats=torch.cat([x.mean(-1),x[:,:,-1]-x[:,:,0]],dim=1)
        return self.project(torch.cat([z,stats],dim=1))


class ProcessClassifier(nn.Module):
    def __init__(self,encoder,n_classes):
        super().__init__()
        self.encoder=encoder
        self.head=nn.Linear(encoder.architecture['embedding_dim'],n_classes)

    def forward(self,x):
        return self.head(self.encoder(x))


def embed_process(encoder,x,batch_size=128):
    a=validate_windows(x,encoder.architecture['channels'])
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError("batch_size must be positive")
    encoder.eval()
    result=[]
    with torch.no_grad():
        for begin in range(0,len(a),batch_size):
            result.append(F.normalize(encoder(torch.from_numpy(a[begin:begin+batch_size])),dim=1))
    return torch.cat(result) if result else torch.empty((0,encoder.architecture['embedding_dim']))


def save_process_model(path,model,scaler,contract,class_names,kind="cnn",prototypes=None,provenance=None):
    contract=validate_contract(contract)
    if kind not in {"cnn","transfer","prototype"} or not class_names or len(set(class_names))!=len(class_names):
        raise ValueError("Invalid model kind or class names")
    encoder=model.encoder if isinstance(model,ProcessClassifier) else model
    if encoder.architecture['channels']!=len(contract['channels']) or len(scaler.mean)!=len(contract['channels']):
        raise ValueError("Encoder/scaler channels do not match contract")
    if kind=="prototype":
        if prototypes is None or tuple(prototypes.shape)!=(len(class_names),encoder.architecture['embedding_dim']) or not torch.isfinite(prototypes).all():
            raise ValueError("A finite prototype for every class is required")
        state=encoder.state_dict()
    else:
        if not isinstance(model,ProcessClassifier) or model.head.out_features!=len(class_names):
            raise ValueError("Classifier head does not match class_names")
        state=model.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Cannot save nonfinite model parameters')
    resource={"format_version":1,"kind":kind,"architecture":encoder.architecture,
        "contract":contract,"class_names":list(class_names),"scaler":scaler.state(),
        "state_dict":state,"provenance":provenance or {}}
    if prototypes is not None:
        resource['prototypes']=prototypes.detach().cpu()
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(resource,path)


def predict_process_resource(resource,x,input_contract,expected_kind=None):
    """Return class indices and uncalibrated normalized model scores."""
    actual=validate_contract(input_contract)
    if resource.get('format_version')!=1 or actual!=resource.get('contract'):
        raise ValueError("Process input schema/units/aggregation/cadence/window differs from training")
    kind=resource.get('kind')
    if kind not in {"cnn","transfer","prototype"} or (expected_kind is not None and kind!=expected_kind):
        raise ValueError("Model resource kind is incompatible with this operator")
    a=validate_windows(x,len(actual['channels']),actual['window_points'])
    scaler=ChannelStandardizer.from_state(resource['scaler'])
    a=scaler.transform(a)
    encoder=ProcessEncoder(**resource['architecture'])
    names=resource['class_names']
    if not names or len(set(names))!=len(names):
        raise ValueError("Invalid saved class mapping")
    if kind=="prototype":
        encoder.load_state_dict(resource['state_dict'])
        prototypes=resource['prototypes']
        if tuple(prototypes.shape)!=(len(names),encoder.architecture['embedding_dim']) or not torch.isfinite(prototypes).all():
            raise ValueError("Invalid saved prototypes")
        logits=10*embed_process(encoder,a)@F.normalize(prototypes,dim=1).T
    else:
        model=ProcessClassifier(encoder,len(names))
        model.load_state_dict(resource['state_dict'])
        model.eval()
        with torch.no_grad():
            logits=torch.cat([model(torch.from_numpy(a[b:b+128])) for b in range(0,len(a),128)]) if len(a) else torch.empty((0,len(names)))
    scores=logits.softmax(dim=1).numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Model produced nonfinite scores; verify the resource and input scale')
    return scores.argmax(axis=1) if len(a) else np.empty(0,dtype=int),scores


def execute(df: pd.DataFrame,value_columns,model_path,input_contract,
            window_id_column="window_id",time_column="timestamp",expected_kind=None,
            minimum_model_score=None,**kwargs):
    """out_1 prediction/window; out_2 model metadata. Inference never trains.

    Every window must have the saved number of complete samples in the saved
    channel order, with unique regular timestamps. Model scores are not
    calibrated fault probabilities. minimum_model_score optionally returns
    unresolved below a caller-supplied threshold. Calibrate that threshold on
    independent validation; it is not guaranteed unknown-fault detection. The
    default is closed-set classification. The deployment supplies a trusted resource.
    Dependencies: numpy, pandas, torch. No private project imports are needed in
    generated standalone operators.
    """
    if minimum_model_score is not None and (isinstance(minimum_model_score,bool) or not np.isfinite(minimum_model_score) or not 0<=minimum_model_score<=1):
        raise ValueError('minimum_model_score must be None or a finite number in [0,1]')
    contract=validate_contract(input_contract)
    values=[value_columns] if isinstance(value_columns,str) else list(value_columns)
    if values!=[c['name'] for c in contract['channels']]:
        raise ValueError("value_columns/order must match the channel schema")
    if not isinstance(df,pd.DataFrame) or not df.columns.is_unique or any(c not in df for c in values+[window_id_column,time_column]):
        raise ValueError("Missing or duplicate required table columns")
    if set(values)&{window_id_column,time_column} or window_id_column==time_column or df[window_id_column].isna().any():
        raise ValueError("Distinct signal, time and nonmissing window ID columns required")
    if not model_path:
        raise ValueError("An explicit model_path is required")
    resource=torch.load(Path(model_path),map_location='cpu',weights_only=True)
    arrays,records=[],[]
    for key,part in df.groupby(window_id_column,sort=False):
        part=part.copy()
        part[time_column]=normalize_process_time(part[time_column],contract.get('source_timezone'))
        part=part.sort_values(time_column)
        delta=part[time_column].diff().dt.total_seconds().iloc[1:].to_numpy()
        if len(part)!=contract['window_points'] or part[time_column].isna().any() or not np.allclose(delta,contract['interval_seconds'],rtol=0,atol=1e-6):
            raise ValueError("Incomplete, duplicate or discontinuous process window")
        arrays.append(part[values].to_numpy(dtype=np.float32).T)
        records.append({window_id_column:key,"start_time":part[time_column].iloc[0],"end_time":part[time_column].iloc[-1]})
    x=np.stack(arrays) if arrays else np.empty((0,len(values),contract['window_points']),dtype=np.float32)
    pred,scores=predict_process_resource(resource,x,contract,expected_kind)
    result_df=pd.DataFrame(records,columns=[window_id_column,'start_time','end_time'])
    result_df['predicted_label']=[resource['class_names'][i] for i in pred]
    result_df['model_score']=scores.max(axis=1) if len(scores) else pd.Series(dtype=float)
    accepted=np.ones(len(result_df),dtype=bool) if minimum_model_score is None else result_df.model_score.to_numpy()>=minimum_model_score
    result_df['diagnosis_status']=np.where(accepted,'classified','below_configured_score_threshold')
    result_df.loc[~accepted,'predicted_label']='unresolved'
    info_df=pd.DataFrame([{"kind":resource['kind'],"model_path":str(model_path),"n_windows":len(x),
        "signal_type":"multivariate_process","scores_calibrated":False,
        "minimum_model_score":minimum_model_score}])
    return {"out_1":result_df,"out_2":info_df}
