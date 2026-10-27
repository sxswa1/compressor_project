"""New-class support adaptation preserves source weights and original counts."""
from pathlib import Path
import importlib,hashlib
import numpy as np
import pandas as pd
import torch
import pytest
from common_algorithms.models.process_cnn import ChannelStandardizer,ProcessEncoder,ProcessClassifier,save_process_model
adapt=importlib.import_module('02_few_shot_diagnosis.pipelines.process_adaptation').adapt_process_resource


def fixture(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(7)
    contract={'signal_type':'multivariate_process','interval_seconds':360,'window_points':4,'channels':[
        {'name':'T','unit':'degC','quantity':'temperature','aggregation':'test'},
        {'name':'P','unit':'bar','quantity':'pressure','aggregation':'test'}]}
    x=np.arange(32,dtype=np.float32).reshape(4,2,4)
    scaler=ChannelStandardizer().fit(x);model=ProcessClassifier(ProcessEncoder(2),2)
    source=tmp_path/'source.pt';save_process_model(source,model,scaler,contract,['old_a','old_b'])
    frames=[]
    for i in range(4):
        f=pd.DataFrame(x[i].T,columns=['T','P'])
        f['window_id']=i;f['timestamp']=pd.date_range('2026-01-01',periods=4,freq='360s')
        f['fault']='new_a' if i<2 else 'new_b'
        f['origin']='same_source_record' if i<2 else f'other_record_{i}'
        frames.append(f)
    return pd.concat(frames,ignore_index=True),source,contract


def test_new_prototypes_preserve_source_and_count_original_records(tmp_path):
    df,source,contract=fixture(tmp_path)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    output=tmp_path/'new_prototypes.pt'
    report=adapt(df,source,output,contract,'fault','origin','physics_simulation')
    resource=torch.load(output,weights_only=True)
    assert resource['kind']=='prototype' and resource['class_names']==['new_a','new_b']
    assert report['query_records_used']==0 and not report['normalization_refitted']
    assert report['counts']['new_a']=={'support_windows':2,'caller_provided_origin_records':1}
    assert report['counts']['new_b']['caller_provided_origin_records']==2
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=adapt(df,source,tmp_path/'one_per_window.pt',contract,'fault','window_id','physics_simulation')
    assert report['counts']['new_a']['caller_provided_origin_records']==2


def test_fine_tuning_updates_copy_without_changing_original_resource(tmp_path):
    df,source,contract=fixture(tmp_path)
    before=torch.load(source,weights_only=True);digest=hashlib.sha256(source.read_bytes()).hexdigest()
    output=tmp_path/'new_transfer.pt'
    adapt(df,source,output,contract,'fault','origin','physics_simulation',method='fine_tune',steps=2)
    after=torch.load(output,weights_only=True)
    assert after['kind']=='transfer' and after['class_names']==['new_a','new_b']
    assert any(not torch.equal(v,after['state_dict'][k]) for k,v in before['state_dict'].items() if k.startswith('encoder.'))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest


def test_fine_tuning_rate_is_forwarded_without_changing_source(tmp_path):
    df,source,contract=fixture(tmp_path);original=hashlib.sha256(source.read_bytes()).hexdigest()
    first=tmp_path/'first.pt';second=tmp_path/'second.pt'
    adapt(df,source,first,contract,'fault','origin','physics_simulation',method='fine_tune',steps=2,learning_rate=.002)
    report=adapt(df,source,second,contract,'fault','origin','physics_simulation',method='fine_tune',steps=2,learning_rate=.0001)
    a=torch.load(first,weights_only=True);b=torch.load(second,weights_only=True)
    assert report['learning_rate']==.0001
    assert any(not torch.equal(value,b['state_dict'][name]) for name,value in a['state_dict'].items())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original
    with pytest.raises(ValueError,match='learning_rate'):
        adapt(df,source,tmp_path/'invalid.pt',contract,'fault','origin','physics_simulation',learning_rate=float('nan'))


def test_invalid_support_is_rejected_instead_of_becoming_a_smaller_dataset(tmp_path):
    df,source,contract=fixture(tmp_path)
    with pytest.raises(ValueError,match='complete'):
        adapt(df.iloc[1:],source,tmp_path/'missing.pt',contract,'fault','origin','physics_simulation')
    bad=df.copy();bad.loc[0,'T']=float('nan')
    with pytest.raises(ValueError,match='missing values'):
        adapt(bad,source,tmp_path/'nan.pt',contract,'fault','origin','physics_simulation')
    bad_contract={**contract,'channels':[dict(c) for c in contract['channels']]}
    bad_contract['channels'][0]['unit']='K'
    with pytest.raises(ValueError,match='schema differs'):
        adapt(df,source,tmp_path/'wrong_units.pt',bad_contract,'fault','origin','physics_simulation')
