"""One mixed record cannot silently enter two classes' opposite episode roles."""
import importlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
module=importlib.import_module('02_few_shot_diagnosis.models.global_record_meta')
legacy=importlib.import_module('02_few_shot_diagnosis.models.process_learning')
source=importlib.import_module('02_few_shot_diagnosis.pipelines.process_source_training')


def mixed():
    return np.array([0,1,0,1,0,2,0,2]),np.array(['A','A','B','B','C','C','D','D'])


def test_mixed_normal_fault_records_are_globally_disjoint_in_every_episode():
    labels,groups=mixed();sampler=module.GlobalRecordEpisodeSampler(labels,groups)
    first=np.random.default_rng(42);second=np.random.default_rng(42)
    for _ in range(40):
        ids,evidence=sampler.sample(np.arange(3),1,1,first)
        replay,_=sampler.sample(np.arange(3),1,1,second)
        np.testing.assert_array_equal(ids,replay)
        assert not set(groups[ids[:,0]])&set(groups[ids[:,1]])
        assert evidence['record_role_intersection']==0 and evidence['mixed_record_solver_used']
        assert np.array_equal(labels[ids],np.repeat(np.arange(3)[:,None],2,axis=1))


def test_odd_cycle_is_infeasible_for_three_classes_but_valid_subsets_work():
    labels=np.repeat(np.arange(3),2);groups=np.array(['A','B','B','C','C','A'])
    sampler=module.GlobalRecordEpisodeSampler(labels,groups)
    with pytest.raises(ValueError,match='globally disjoint.*infeasible'):
        sampler.sample(np.arange(3),1,1,np.random.default_rng(1))
    ids,_=sampler.sample(np.array([0,1]),1,1,np.random.default_rng(1))
    assert not set(groups[ids[:,0]])&set(groups[ids[:,1]])


def test_homogeneous_global_training_preserves_frozen_legacy_weights():
    torch.set_num_threads(1);rng=np.random.default_rng(3)
    x=rng.normal(size=(12,2,6)).astype(np.float32);y=np.repeat(np.arange(3),4)
    groups=np.array([f'{i:03d}' for i in range(12)])
    old,_=legacy.train_process_protonet(x,y,seed=7,episodes=3,support=1,query=1,ways=3,record_groups=groups)
    new,history=module.train_global_record_protonet(x,y,groups,seed=7,episodes=3,support=1,query=1,ways=3)
    assert all(torch.equal(value,new.state_dict()[name]) for name,value in old.state_dict().items())
    assert all(r['global_record_roles_verified'] and not r['mixed_record_solver_used'] for r in history)


def test_bounded_solver_failure_does_not_drop_a_class_or_accept_a_partial_episode(monkeypatch):
    y,g=mixed();sampler=module.GlobalRecordEpisodeSampler(y,g)
    monkeypatch.setattr(module,'milp',lambda **kwargs:SimpleNamespace(status=1,x=None))
    with pytest.raises(RuntimeError,match='no class or shot was dropped'):
        sampler.sample(np.arange(3),1,1,np.random.default_rng(42))


def test_explicit_global_source_mode_accepts_mixed_records_and_records_provenance(tmp_path):
    torch.set_num_threads(1);labels,groups=mixed();rng=np.random.default_rng(10)
    contract={'signal_type':'multivariate_process','interval_seconds':360,'window_points':4,'channels':[
        {'name':'T','unit':'degC','quantity':'temperature','aggregation':'test'}]}
    parts=[]
    for i,(label,group) in enumerate(zip(labels,groups)):
        parts.append(pd.DataFrame({'T':rng.normal(size=4),'window_id':f'w{i}','origin':group,'label':str(label),
            'timestamp':pd.date_range('2026-01-01',periods=4,freq='360s')}))
    table=pd.concat(parts,ignore_index=True)
    path=tmp_path/'global.pt'
    report=source.train_process_source(table,path,contract,'label','origin','physics_simulation',episodes=2,
        episode_support=1,episode_query=1,episode_grouping='global_records')
    assert report['episode_grouping']=='global_records' and report['episode_global_record_separation']
    assert report['counts']['0']['caller_provided_origin_records']==4
    assert report['counts']['1']['caller_provided_origin_records']==2
    assert torch.load(path,weights_only=True)['provenance']['training_history'][-1]['mixed_record_solver_used']


@pytest.mark.parametrize('kind',['missing','blank','unaligned','too_few_records'])
def test_invalid_source_groups_are_not_renamed_or_replaced(kind):
    y,g=mixed();g=g.astype(object)
    if kind=='missing':g[0]=None
    if kind=='blank':g[0]=' '
    if kind=='unaligned':g=g[:-1]
    if kind=='too_few_records':g[:4]='A'
    with pytest.raises(ValueError):
        sampler=module.GlobalRecordEpisodeSampler(y,g)
        sampler.sample(np.arange(3),1,1,np.random.default_rng(42))
