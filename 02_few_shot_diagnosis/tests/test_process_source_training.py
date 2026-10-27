"""Source training rejects record leakage and feeds the existing target adapter."""
import importlib
import numpy as np
import pandas as pd
import pytest
import torch
from common_algorithms.models.process_cnn import execute,ChannelStandardizer

train=importlib.import_module('02_few_shot_diagnosis.pipelines.process_source_training').train_process_source
adapt=importlib.import_module('02_few_shot_diagnosis.pipelines.process_adaptation').adapt_process_resource
learning=importlib.import_module('02_few_shot_diagnosis.models.process_learning')
prepare=importlib.import_module('02_few_shot_diagnosis.pipelines.process_support').labelled_support_windows


def source_table():
    contract={'signal_type':'multivariate_process','interval_seconds':360,'window_points':6,'channels':[
        {'name':'T','unit':'degC','quantity':'temperature','aggregation':'test_native'},
        {'name':'P','unit':'bar','quantity':'pressure','aggregation':'test_native'}]}
    frames=[]
    for i in range(8):
        part=pd.DataFrame({'T':20+i*2+np.arange(6)*.1,'P':2+i*.2+np.arange(6)*.01})
        part['timestamp']=pd.date_range('2026-01-01',periods=6,freq='360s')
        part['window_id']=f'w{i}';part['record']=f'{i:05d}';part['label']='old_a' if i<4 else 'old_b'
        frames.append(part)
    return pd.concat(frames,ignore_index=True),contract


def test_episodic_source_reuses_learner_and_adapts_to_new_classes(tmp_path):
    torch.set_num_threads(1);df,contract=source_table();before=df.copy(deep=True)
    path=tmp_path/'source.pt'
    report=train(df,path,contract,'label','record','physics_simulation',episodes=2,episode_support=1,episode_query=1)
    resource=torch.load(path,weights_only=True)
    x,y,_,_,_,_=prepare(df,contract,'label','record')
    scale=ChannelStandardizer().fit(x)
    encoder,_=learning.train_process_protonet(scale.transform(x),y,seed=42,episodes=2,
        support=1,query=1,ways=2,record_groups=np.asarray([f'{i:05d}' for i in range(8)]))
    assert all(torch.equal(v,resource['state_dict'][name]) for name,v in encoder.state_dict().items())
    assert resource['provenance']['source_training_episodes']==2
    assert report['episode_global_record_separation'] and report['normalization_fit']=='labelled_source_only'
    assert report['counts']['old_a']=={'source_windows':4,'caller_provided_origin_records':4}
    target=df.copy();target['label']=target.label.map({'old_a':'new_a','old_b':'new_b'});target[['T','P']]+=10
    target['record']='target_'+target.record
    adapted=tmp_path/'new.pt';digest=path.read_bytes()
    adapt(target,path,adapted,contract,'label','record','physics_simulation')
    new=torch.load(adapted,weights_only=True)
    assert new['scaler']==resource['scaler'] and new['class_names']==['new_a','new_b']
    assert new['provenance']['source_training_provenance']['method']=='protonet'
    assert len(execute(target,['T','P'],adapted,contract)['out_1'])==8
    assert path.read_bytes()==digest
    pd.testing.assert_frame_equal(df,before)


def test_mixed_class_origins_are_rejected_globally_before_training(tmp_path):
    df,contract=source_table()
    df.loc[df.window_id.eq('w4'),'record']='00000'
    path=tmp_path/'leaky.pt'
    with pytest.raises(ValueError,match='one class per original record'):
        train(df,path,contract,'label','record','physics_simulation',episodes=1,episode_support=1,episode_query=1)
    assert not path.exists() and not path.with_suffix('.pt.metadata.json').exists()


def test_multiple_windows_do_not_become_extra_episode_records(tmp_path):
    df,contract=source_table();df['record']=df.label
    with pytest.raises(ValueError,match='distinct original source records'):
        train(df,tmp_path/'bad.pt',contract,'label','record','physics_simulation',episodes=1,episode_support=1,episode_query=1)


def test_supervised_source_is_compatible_and_existing_weights_preserved(tmp_path):
    torch.set_num_threads(1);df,contract=source_table();path=tmp_path/'cnn.pt'
    report=train(df,path,contract,'label','record','physics_simulation',method='cnn',steps=2)
    resource=torch.load(path,weights_only=True)
    assert resource['kind']=='cnn' and report['source_training_episodes']==0
    assert report['source_training_steps']==2 and report['final_query_records_used']==0
    assert len(execute(df,['T','P'],path,contract)['out_1'])==8
    with pytest.raises(FileExistsError):
        train(df,path,contract,'label','record','physics_simulation',method='cnn',steps=2)


@pytest.mark.parametrize('options',[{'episode_ways':3},{'seed':True},{'learning_rate':float('nan')},
    {'episodes':False},{'episode_query':0},{'source_kind':'unlabelled_field'}])
def test_invalid_source_training_options_fail_before_writing(tmp_path,options):
    df,contract=source_table();path=tmp_path/'invalid.pt'
    kwargs={'source_kind':'physics_simulation','episodes':1,'episode_support':1,'episode_query':1,**options}
    with pytest.raises(ValueError):train(df,path,contract,'label','record',**kwargs)
    assert not path.exists()
