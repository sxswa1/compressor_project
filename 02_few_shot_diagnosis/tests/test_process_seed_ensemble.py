"""Independent seed aggregation math and strict standalone contracts."""
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from common_algorithms.models.process_cnn import ChannelStandardizer,ProcessEncoder,ProcessClassifier,save_process_model,predict_process_resource
from common_algorithms.models.process_seed_ensemble import execute,predict_seed_ensemble


@pytest.fixture(params=['cnn','prototype','transfer'])
def ensemble(request,tmp_path):
    torch.set_num_threads(2);rng=np.random.default_rng(1);x=rng.normal(size=(8,2,5)).astype(np.float32);scaler=ChannelStandardizer().fit(x)
    contract={'signal_type':'multivariate_process','interval_seconds':900.,'window_points':5,
        'channels':[{'name':'p'+str(i),'quantity':'temperature','unit':'degC','aggregation':'maximum'} for i in range(2)]}
    members=[]
    for seed in [42,43,44]:
        torch.manual_seed(seed);encoder=ProcessEncoder(2);model=ProcessClassifier(encoder,4);proto=torch.nn.functional.normalize(torch.randn(4,32),dim=1) if request.param=='prototype' else None
        path=tmp_path/(str(seed)+'.pt');save_process_model(path,encoder if proto is not None else model,scaler,contract,['n','t','o','v'],request.param,proto,{'train_seed':seed})
        members.append(torch.load(path,weights_only=True))
    item={'kind':'process_seed_ensemble','format_version':1,'contract':contract,'class_names':['n','t','o','v'],'members':members,'train_seeds':[42,43,44]}
    path=tmp_path/'ensemble.pt';torch.save(item,path)
    frame=pd.DataFrame(x[0].T,columns=['p0','p1']);frame.insert(0,'timestamp',pd.date_range('2022-01-01',periods=5,freq='15min'));frame.insert(0,'window_id','a')
    return item,path,x,frame,contract


def test_ensemble_uses_equal_probability_average_and_handles_empty(ensemble):
    item,path,x,frame,contract=ensemble
    reference=np.mean([predict_process_resource(m,x,contract)[1].astype(float) for m in item['members']],axis=0)
    pred,scores=predict_seed_ensemble(item,x,contract);np.testing.assert_array_equal(scores,reference)
    np.testing.assert_array_equal(pred,reference.argmax(axis=1))
    result=execute(frame,['p0','p1'],path,contract)
    assert result['out_1'].predicted_label.tolist()==[item['class_names'][pred[0]]]
    assert bool(result['out_2'].same_support_labels_in_all_members.iloc[0])
    empty=execute(frame.iloc[:0],['p0','p1'],path,contract)
    assert empty['out_1'].empty and 'predicted_label' in empty['out_1']


def test_ensemble_rejects_wrong_units_and_incompatible_member(ensemble):
    item,path,x,frame,contract=ensemble;wrong=copy.deepcopy(contract);wrong['channels'][0]['unit']='K'
    with pytest.raises(ValueError):execute(frame,['p0','p1'],path,wrong)
    malformed=copy.deepcopy(item);malformed['members'][1]['class_names'][0]='other'
    with pytest.raises(ValueError):predict_seed_ensemble(malformed,x,contract)


def test_ensemble_rejects_missing_members_duplicate_seeds_and_clock(ensemble):
    item,path,x,frame,contract=ensemble;bad=copy.deepcopy(item);bad['members']=bad['members'][:2]
    with pytest.raises(ValueError):predict_seed_ensemble(bad,x,contract)
    bad=copy.deepcopy(item);bad['train_seeds']=[42,42,44]
    with pytest.raises(ValueError):predict_seed_ensemble(bad,x,contract)
    broken=frame.copy();broken.loc[1,'timestamp']=broken.loc[0,'timestamp']
    with pytest.raises(ValueError):execute(broken,['p0','p1'],path,contract)


def test_ensemble_threshold_is_explicit_and_scores_are_uncalibrated(ensemble):
    item,path,x,frame,contract=ensemble
    out=execute(frame,['p0','p1'],path,contract,minimum_model_score=1.)
    assert out['out_1'].predicted_label.tolist()==['unresolved'] and not bool(out['out_2'].scores_calibrated.iloc[0])
    with pytest.raises(ValueError):execute(frame,['p0','p1'],path,contract,minimum_model_score=-.1)


def test_ensemble_class_mapping_is_not_hardcoded_to_rcsd_four_states(tmp_path):
    # API compatibility with eight synthetic names is not real eight-fault validation.
    torch.set_num_threads(2);x=np.ones((1,2,5),dtype=np.float32);scaler=ChannelStandardizer().fit(x)
    contract={'signal_type':'multivariate_process','interval_seconds':900.,'window_points':5,
        'channels':[{'name':str(i),'quantity':'temperature','unit':'degC','aggregation':'maximum'} for i in range(2)]}
    names=['synthetic_name_'+str(i) for i in range(8)];members=[]
    for seed in [42,43,44]:
        torch.manual_seed(seed);path=tmp_path/(str(seed)+'.pt')
        save_process_model(path,ProcessClassifier(ProcessEncoder(2),8),scaler,contract,names,'cnn')
        members.append(torch.load(path,weights_only=True))
    resource={'kind':'process_seed_ensemble','format_version':1,'contract':contract,'class_names':names,'members':members,'train_seeds':[42,43,44]}
    pred,scores=predict_seed_ensemble(resource,x,contract)
    assert scores.shape==(1,8) and 0<=pred[0]<8
    np.testing.assert_allclose(scores.sum(axis=1),1.,atol=1e-7)
