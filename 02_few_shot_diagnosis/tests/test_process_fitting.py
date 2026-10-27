"""New-schema fitting uses support evidence and interoperates with operators."""
import importlib
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from common_algorithms.models.process_forest import execute as forest_execute
from common_algorithms.models.process_cnn import execute as cnn_execute

fit=importlib.import_module('02_few_shot_diagnosis.pipelines.process_fitting').fit_process_resource
prepare=importlib.import_module('02_few_shot_diagnosis.pipelines.process_support').labelled_support_windows


def support_table():
    contract={'signal_type':'multivariate_process','interval_seconds':360,'window_points':6,'channels':[
        {'name':'temperature','unit':'degC','quantity':'temperature','aggregation':'test_native'},
        {'name':'pressure','unit':'bar','quantity':'pressure','aggregation':'test_native'}]}
    parts=[]
    for i in range(6):
        part=pd.DataFrame({'temperature':20+i*3+np.arange(6)*.2,'pressure':2+i*.4+np.arange(6)*.01})
        part['timestamp']=pd.date_range('2026-01-01',periods=6,freq='360s')
        part['window_id']=f'window_{i}';part['fault']='A' if i<3 else 'B'
        part['original_record']='same_record_A' if i<3 else f'B_{i}'
        parts.append(part)
    return pd.concat(parts,ignore_index=True),contract


def test_new_forest_records_budget_and_uses_support_scaler(tmp_path):
    df,contract=support_table();before=df.copy(deep=True);path=tmp_path/'baseline.joblib'
    report=fit(df,path,contract,'fault','original_record','physics_simulation',n_estimators=9,class_order=['B','A'])
    resource=joblib.load(path)
    assert resource['class_names']==['B','A']
    assert report['counts']['A']=={'support_windows':3,'caller_provided_origin_records':1}
    assert report['query_records_used']==0 and report['normalization_fit']=='labelled_support_only'
    assert not report['source_pretraining_used']
    assert np.allclose(resource['scaler']['mean'],df[['temperature','pressure']].mean().to_numpy(),atol=1e-5)
    output=forest_execute(df,['temperature','pressure'],path,contract)['out_1']
    assert len(output)==6 and set(output.predicted_label)<=set(['A','B'])
    pd.testing.assert_frame_equal(df,before)
    digest=path.read_bytes()
    with pytest.raises(FileExistsError):
        fit(df,path,contract,'fault','original_record','physics_simulation')
    assert path.read_bytes()==digest


def test_feature_selection_and_single_example_fallback_interoperate(tmp_path):
    df,contract=support_table();path=tmp_path/'selected.joblib'
    report=fit(df,path,contract,'fault','original_record','physics_simulation',n_estimators=7,max_features=3)
    resource=joblib.load(path)
    assert len(resource['selected_feature_indices'])==3
    assert report['feature_selection']['method']=='support_only_anova'
    assert len(forest_execute(df,['temperature','pressure'],path,contract)['out_1'])==6
    sparse=df[df.window_id.isin(['window_0','window_3'])]
    path=tmp_path/'k1.joblib'
    report=fit(sparse,path,contract,'fault','original_record','physics_simulation',n_estimators=5,max_features=3)
    assert report['feature_selection']['reason']=='less_than_two_support_records_per_class'
    assert len(joblib.load(path)['selected_feature_indices'])==14


def test_new_cnn_uses_declared_schema_and_is_loadable(tmp_path):
    torch.set_num_threads(1);df,contract=support_table();path=tmp_path/'cnn.pt'
    report=fit(df,path,contract,'fault','original_record','physics_simulation',method='cnn',steps=2)
    resource=torch.load(path,weights_only=True)
    assert resource['kind']=='cnn' and resource['contract']==contract
    assert report['training_steps']==2 and not report['source_pretraining_used']
    output=cnn_execute(df,['temperature','pressure'],path,contract)['out_1']
    assert len(output)==6 and output.model_score.between(0,1).all()


def test_cnn_learning_rate_changes_fitting_and_is_recorded(tmp_path):
    torch.set_num_threads(1);df,contract=support_table()
    first=tmp_path/'first.pt';second=tmp_path/'second.pt'
    fit(df,first,contract,'fault','original_record','physics_simulation',method='cnn',steps=2,learning_rate=.002)
    report=fit(df,second,contract,'fault','original_record','physics_simulation',method='cnn',steps=2,learning_rate=.0001)
    a=torch.load(first,weights_only=True);b=torch.load(second,weights_only=True)
    assert report['learning_rate']==.0001 and b['provenance']['learning_rate']==.0001
    assert any(not torch.equal(value,b['state_dict'][name]) for name,value in a['state_dict'].items())
    for invalid in [float('nan'),0,True]:
        with pytest.raises(ValueError,match='learning_rate'):
            fit(df,tmp_path/'invalid.pt',contract,'fault','original_record','physics_simulation',method='cnn',learning_rate=invalid)


@pytest.mark.parametrize('kind',['blank_origin','gap','missing_value','incomplete','mixed_label'])
def test_bad_support_is_rejected_before_a_resource_is_written(tmp_path,kind):
    df,contract=support_table()
    if kind=='blank_origin':df.loc[:5,'original_record']='  '
    if kind=='gap':df.loc[2,'timestamp']+=pd.Timedelta(seconds=1)
    if kind=='missing_value':df.loc[0,'temperature']=np.nan
    if kind=='incomplete':df=df.iloc[1:]
    if kind=='mixed_label':df.loc[0,'fault']='B'
    path=tmp_path/'bad.joblib'
    with pytest.raises(ValueError):
        fit(df,path,contract,'fault','original_record','physics_simulation')
    assert not path.exists() and not path.with_suffix('.joblib.metadata.json').exists()


def test_metadata_and_resource_options_are_explicit(tmp_path):
    df,contract=support_table()
    x,_,_,counts,_,_=prepare(df,contract,'fault','window_id')
    assert x.flags.c_contiguous
    assert counts['A']['caller_provided_origin_records']==3
    for options in [{'method':'cnn','max_features':2},{'seed':-1},{'max_features':True},{'max_features':15},
                    {'class_order':['A','C']},{'support_source_kind':'unlabelled_field'}]:
        params={'support_source_kind':'physics_simulation',**options}
        with pytest.raises(ValueError):
            fit(df,tmp_path/('bad.pt' if options.get('method')=='cnn' else 'bad.joblib'),contract,
                'fault','original_record',**params)
