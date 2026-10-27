"""Metric-resource deployment behavior, including empty and malformed input."""
import copy
from importlib import import_module
import numpy as np
import pandas as pd
import pytest
import torch
from common_algorithms.models.process_cnn import ChannelStandardizer,ProcessEncoder
from common_algorithms.models.process_metric_prototype import execute,predict_metric_resource

opt=import_module('02_few_shot_diagnosis.models.rcsd_optimization')


@pytest.fixture(params=['statistics','meta_raw','supervised_raw'])
def resource(request,tmp_path):
    torch.set_num_threads(2);torch.manual_seed(7);rng=np.random.default_rng(7)
    x=rng.normal(size=(16,2,5)).astype(np.float32);y=np.tile(np.arange(4),4);x+=y[:,None,None]
    contract={'signal_type':'multivariate_process','interval_seconds':900.,'window_points':5,
        'channels':[{'name':'p'+str(i),'quantity':'temperature','unit':'degC','aggregation':'maximum'} for i in range(2)]}
    scaler=ChannelStandardizer().fit(x);family=request.param;encoder=None if family=='statistics' else ProcessEncoder(2)
    features=opt.raw_representation(scaler.transform(x),family,['p0','p1'],900.,encoder)
    geometry=opt.fit_source_geometry(features,y,'shrinkage_within');prototypes=opt.target_prototypes(features,y,geometry)
    item={'kind':'rcsd_metric_prototype','version':2,'process_contract':contract,'class_names':['n','t','o','v'],
        'representation_family':family,'scaler':scaler.state(),'geometry':geometry,'prototypes':prototypes.tolist()}
    if encoder is not None:item.update(architecture=encoder.architecture,encoder_state=encoder.state_dict())
    path=tmp_path/'resource.pt';torch.save(item,path)
    frame=pd.DataFrame(x[0].T,columns=['p0','p1']);frame.insert(0,'timestamp',pd.date_range('2022-01-01',periods=5,freq='15min'));frame.insert(0,'window_id','a')
    return item,path,x,frame,contract,features,prototypes,geometry


def test_metric_inference_matches_training_math_and_handles_empty(resource):
    item,path,x,frame,contract,features,prototypes,geometry=resource
    pred,scores=predict_metric_resource(item,x,contract)
    np.testing.assert_allclose(scores,opt.geometry_scores(features,prototypes,geometry),rtol=1e-6,atol=1e-8)
    result=execute(frame,['p0','p1'],path,contract)
    assert set(result)=={'out_1','out_2'} and len(result['out_1'])==1
    assert result['out_1'].predicted_label.iloc[0]==item['class_names'][pred[0]]
    empty=execute(frame.iloc[:0],['p0','p1'],path,contract)
    assert empty['out_1'].empty and 'predicted_label' in empty['out_1']
    assert not bool(empty['out_2'].requires_query_normal_reference.iloc[0])


def test_metric_rejects_changed_units_clock_and_resource_kind(resource):
    item,path,x,frame,contract,*_=resource
    wrong=copy.deepcopy(contract);wrong['channels'][0]['unit']='K'
    with pytest.raises(ValueError):execute(frame,['p0','p1'],path,wrong)
    badclock=frame.copy();badclock.loc[1,'timestamp']=badclock.loc[0,'timestamp']
    with pytest.raises(ValueError):execute(badclock,['p0','p1'],path,contract)
    wrong=copy.deepcopy(item);wrong['kind']='prototype'
    with pytest.raises(ValueError):predict_metric_resource(wrong,x,contract)


def test_metric_rejects_nonpositive_geometry_and_nonfinite_input(resource):
    item,path,x,frame,contract,*_=resource
    bad=copy.deepcopy(item);bad['geometry']['precision'][0][0]=-99999
    with pytest.raises(ValueError):predict_metric_resource(bad,x,contract)
    badx=x.copy();badx[0,0,0]=np.nan
    with pytest.raises(ValueError):predict_metric_resource(item,badx,contract)


def test_metric_threshold_only_marks_outputs_unresolved(resource):
    item,path,x,frame,contract,*_=resource
    result=execute(frame,['p0','p1'],path,contract,minimum_model_score=1.)['out_1']
    assert result.predicted_label.tolist()==['unresolved']
    assert result.diagnosis_status.tolist()==['below_configured_score_threshold']
    with pytest.raises(ValueError):execute(frame,['p0','p1'],path,contract,minimum_model_score=float('nan'))
