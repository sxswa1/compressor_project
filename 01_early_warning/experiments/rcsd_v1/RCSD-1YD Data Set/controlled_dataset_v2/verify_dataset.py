"""Reconstruct every scene independently and check split/label contracts."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from load_dataset import load_scenes

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
SOURCE=ROOT.parent/'dataset_v1/data'

def main():
    for folder in [SOURCE,DATA]:
        for name,digest in json.loads((folder/'checksums.json').read_text()).items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,(folder,name)
    manifest=json.loads((DATA/'scene_manifest.json').read_text())
    raw=pd.read_csv(SOURCE/'raw_observations.csv.gz').set_index('source_row_id')
    meta=pd.read_csv(SOURCE/'raw_metadata.csv').set_index('source_row_id')
    amps=pd.read_csv(DATA/'training_amplitudes.csv').set_index('tag')
    cfg=json.loads((DATA/'config_snapshot.json').read_text())
    original=json.loads((SOURCE/'config_snapshot.json').read_text())
    normal=raw.loc[meta['split'].eq('train') & meta.baseline_candidate]
    floors={'temperature':.1,'vibration':.1,'axial':.001,'pressure':.01,'differential_pressure':.01,'speed':1.}
    for tag,a in amps.iterrows():
        scale=max(float(np.median(np.abs(normal[tag]-np.median(normal[tag]))))*1.4826,floors[a['kind']])
        assert np.isclose(scale,a.robust_scale)
        if pd.notna(a.amplitude):
            assert np.isclose(a.amplitude,np.clip(scale*6,*original['amplitude_caps'][a['kind']]))
    source_sets={};maximum_error=0.;stats=[]
    for split in ['train','validation']:
        obs,lab,lin,mask=[pd.read_csv(DATA/f'{split}_{name}.csv.gz').set_index(['scene_id','step']) for name in ['observations','labels','lineage','injection_mask']]
        assert obs.index.is_unique and all(obs.index.equals(f.index) for f in [lab,lin,mask])
        assert list(obs)==list(raw) and np.isfinite(obs.to_numpy()).all()
        source_sets[split]=set(lin.source_row_id)
        assert meta.loc[sorted(source_sets[split]),'split'].eq(split).all()
        for tag in obs:
            assert obs[tag].between(*original['bounds'][amps.loc[tag,'kind']]).all()
        for m in [m for m in manifest if m['split']==split]:
            sid=m['scene_id'];x=obs.loc[sid];y=lab.loc[sid];l=lin.loc[sid];n=len(x)
            assert n==m['n'] and list(x.index)==list(range(n))
            assert np.array_equal(l.source_row_id,np.arange(m['first_source_row'],m['last_source_row']+1))
            assert np.array_equal(l.source_timestamp,meta.loc[l.source_row_id,'source_timestamp'])
            expected=raw.loc[l.source_row_id].to_numpy().copy();expected_mask=np.zeros_like(expected,dtype=int)
            if m['benchmark']=='replay':
                assert y[['DI_ref','risk_ref','RUL_ref_hours']].isna().all().all()
                assert not y.di_label_valid.any() and not y.rul_label_valid.any()
            else:
                step=np.arange(n)
                d=np.clip((np.maximum(step-m['onset'],0)/(m['nominal_hours']*4))**m['curve_exponent'],0,1)
                response=np.zeros(n)
                for t in range(n):
                    if t>=m['lag_points']:response[t]=d[t-m['lag_points']]**m['response_power']
                for tag,sign in cfg['groups'][m['group']].items():
                    amplitude=sign*amps.loc[tag,'amplitude']*m['amplitude_multiplier']
                    assert np.isclose(amplitude,m['channels'][tag])
                    j=list(obs).index(tag);expected[:,j]+=amplitude*response;expected_mask[:,j]=response>0
                assert np.allclose(y.DI_ref,1-d) and (np.diff(y.DI_ref)<=1e-12).all()
                risk=1+(y.DI_ref<.85).astype(int)+(y.DI_ref<.65).astype(int)+(y.DI_ref<.4).astype(int)
                assert np.array_equal(y.risk_ref,risk) and y.di_label_valid.all()
                assert np.array_equal(y.rul_label_valid,(d>0)&m['observed'])
                assert y.event_observed.eq(m['observed']).all() and y.right_censored.eq(m['censored']).all()
                if m['observed']:
                    assert y.DI_ref.iloc[-1]==0 and y.RUL_ref_hours.iloc[-1]==0
                    assert np.allclose(y.RUL_ref_hours,(m['onset']+m['nominal_hours']*4-step)/4)
                else:
                    assert y.RUL_ref_hours.isna().all() and y.DI_ref.iloc[-1]>0
                if m['benchmark']=='main':
                    assert (m['amplitude_multiplier'],m['response_power'],m['lag_points'])==(1,1,0)
                else:
                    parent=m['main_scene_id']
                    pd.testing.assert_frame_equal(y,lab.loc[parent])
                    pd.testing.assert_frame_equal(l,lin.loc[parent])
            error=float(np.max(np.abs(expected-x.to_numpy())))
            maximum_error=max(maximum_error,error)
            assert error<1e-9,(sid,error)
            assert np.array_equal(mask.loc[sid],expected_mask)
        loaded,_,_=load_scenes(split)
        assert set(loaded.index.get_level_values(0))=={m['scene_id'] for m in manifest if m['split']==split and m['benchmark']=='main'}
        stats.append(dict(split=split,rows=len(obs),scenes=obs.index.get_level_values(0).nunique(),source_points=len(source_sets[split])))
    assert not source_sets['train'] & source_sets['validation']
    assert not list(DATA.glob('test_*'))
    assert not json.loads((DATA/'rejected_scenes.json').read_text())
    counts=pd.DataFrame(manifest).groupby(['split','benchmark','observed','censored']).size().reset_index(name='scenes')
    result=dict(passed=True,scenes_checked=len(manifest),maximum_reconstruction_error=maximum_error,
                source_v1_checksums_verified=True,train_validation_source_overlap=0,test_built=False,test_scored=False,
                splits=stats,label_counts=counts.to_dict('records'))
    checks=ROOT/'checks';checks.mkdir(exist_ok=True)
    (checks/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
