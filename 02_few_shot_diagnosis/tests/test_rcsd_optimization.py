"""New source-only math, background roles and conservative transfer contracts."""
import copy
from importlib import import_module
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from common_algorithms.models.process_cnn import ProcessEncoder,ProcessClassifier
from common_algorithms.models.process_metric_prototype import predict_metric_resource

opt=import_module('02_few_shot_diagnosis.models.rcsd_optimization')


def pairs():
    x=np.zeros((12,2,5),dtype=np.float32);y=np.tile(np.arange(4),3);g=np.repeat(['a','b','c'],4)
    for i in range(12):x[i]=i//4+y[i]
    return x,y,g


def test_source_augmentation_keeps_original_backgrounds_and_strengths():
    x,y,g=pairs();before=x.copy();ax,ay,ag=opt.augment_source_pairs(x,y,g,[.35,.65,1.])
    np.testing.assert_array_equal(x,before)
    assert ax.shape==(30,2,5) and set(ag)==set(g)
    for record in set(g):
        control=x[(g==record)&(y==0)][0]
        for c in range(1,4):np.testing.assert_allclose(ax[(ag==record)&(ay==c),0,0]-control[0,0],np.array([.35,.65,1.])*c,atol=5e-7)


def test_source_augmentation_rejects_dropped_state():
    x,y,g=pairs()
    with pytest.raises(ValueError):opt.augment_source_pairs(x[:-1],y[:-1],g[:-1],[.5,1.])


@pytest.mark.parametrize('method',['euclidean','diagonal_within','shrinkage_within'])
def test_source_metric_is_positive_definite_and_classifies_own_prototypes(method):
    rng=np.random.default_rng(2);y=np.repeat(np.arange(4),8);x=rng.normal(size=(32,5))*.03+y[:,None]*3
    geo=opt.fit_source_geometry(x,y,method);before=copy.deepcopy(geo);proto=opt.target_prototypes(x,y,geo)
    scores=opt.geometry_scores(x,proto,geo)
    assert (scores.argmax(axis=1)==y).all() and np.linalg.eigvalsh(geo['precision']).min()>0
    opt.geometry_scores(x*100,proto,geo)
    assert geo==before and geo['query_feature_rows']==0
    np.testing.assert_allclose(scores.sum(axis=1),1.)


def test_transfer_head_only_preserves_source_and_encoder():
    torch.set_num_threads(2);torch.manual_seed(2);model=ProcessClassifier(ProcessEncoder(2),4)
    x,y,g=pairs();source=copy.deepcopy(model.state_dict())
    adapted,history=opt.preserve_head_adaptation(model,x,y,42,3,{'encoder_lr':0.,'head_lr':.002,'anchor':0.})
    assert all(torch.equal(v,source[k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,source['encoder.'+k]) for k,v in adapted.encoder.state_dict().items())
    assert not torch.equal(adapted.head.weight,source['head.weight']) and len(history)==2


def test_transfer_refuses_missing_known_state():
    model=ProcessClassifier(ProcessEncoder(2),4);x,y,g=pairs();take=y!=3
    with pytest.raises(ValueError):opt.preserve_head_adaptation(model,x[take],y[take],42,2,{'encoder_lr':0.,'head_lr':.002,'anchor':0.})


def test_half_strength_uses_only_matching_background_and_keeps_control():
    # Script imports intentionally configure their own CLI path. Isolate them
    # so another project's top-level `models` package cannot be shadowed.
    code="""import importlib,numpy as np
r=importlib.import_module('02_few_shot_diagnosis.scripts.run_rcsd_optimization')
x=np.arange(8,dtype=np.float32).reshape(8,1,1).repeat(3,axis=2)
y=np.tile(np.arange(4),2);g=np.repeat(['a','b'],4);weak=r.half_strength(x,y,g)
np.testing.assert_allclose(weak[:,0,0],[0,.5,1,1.5,4,4.5,5,5.5])
"""
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],check=True,capture_output=True)


def test_metric_resource_rejects_other_operator_kind():
    contract={'signal_type':'multivariate_process','interval_seconds':900.,'window_points':5,
        'channels':[{'name':str(c),'quantity':'temperature','unit':'degC','aggregation':'maximum'} for c in range(2)]}
    with pytest.raises(ValueError,match='kind/schema'):predict_metric_resource({'kind':'prototype'},np.zeros((0,2,5)),contract)
