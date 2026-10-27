"""Data preparation and integrity records for the portable RCSD experiment."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def training_fingerprints(root=ROOT):
    paths = [p for p in root.rglob('*.py') if '__pycache__' not in p.parts]
    paths += list(root.glob('*/plan.json')) + list(root.glob('*/config.json'))
    paths += list((root/'RCSD-1YD Data Set').glob('*/config.json'))
    paths += [root/'early_warning_v2/stress_test/config.json',root/'final_evaluation_protocol_v1/protocol.json']
    data = root/'RCSD-1YD Data Set/controlled_dataset_v2/data'
    paths += [data/f'train_{name}.csv.gz' for name in ['observations','labels','lineage']]
    paths += [data/'scene_manifest.json', data/'model_input_columns.json',data/'training_amplitudes.csv']
    paths += list((root/'rul_balanced_validation_v1/auxiliary_data').glob('*'))
    paths += [REPO/'common_algorithms/models/lstm.py',REPO/'common_algorithms/models/gradient_boosting.py',
              REPO/'01_early_warning/pipelines/baseline_data.py']
    return {Path(os.path.relpath(p,root)).as_posix():digest(p) for p in sorted(set(paths)) if p.is_file()}


def seal_assets(root=ROOT):
    import pandas as pd
    registry=pd.read_csv(root/'final_model_fit_v1/results/model_registry.csv')
    assert len(registry)==34
    paths=[root/str(p).replace('\\','/') for p in registry.model_path]
    paths += [root/'final_model_fit_v1/models/rul/reference.pkl',
              root/'final_model_fit_v1/models/rul/interval_radii.json',
              root/'final_model_fit_v1/results/preprocessing.json',
              root/'final_model_fit_v1/results/model_registry.csv',
              root/'warning_lstm_budget_v1/results/bundle.pkl']
    hashes=training_fingerprints(root)
    hashes.update({p.relative_to(root).as_posix():digest(p) for p in paths})
    write(root/'final_evaluation_pipeline_v1/release_manifest.json',dict(
        purpose='Reproduction asset hashes; historical scores are stored separately',sha256=hashes))
    verify_assets(root)


def verify_assets(root=ROOT):
    path=root/'final_evaluation_pipeline_v1/release_manifest.json'
    manifest=read(path)
    for name,expected in manifest['sha256'].items():
        if digest(root/name)!=expected:
            raise ValueError(f'Changed input or model: {name}; use a separate experiment version')


def build_auxiliary(root=ROOT):
    import numpy as np
    import pandas as pd
    data=root/'RCSD-1YD Data Set/controlled_dataset_v2/data'
    cfg=read(data.parent/'config.json')
    bounds=read(root/'RCSD-1YD Data Set/dataset_v1/data/config_snapshot.json')['bounds']
    obs=pd.read_csv(data/'train_observations.csv.gz').set_index(['scene_id','step'])
    manifests=pd.DataFrame(read(data/'scene_manifest.json')).set_index('scene_id')
    amps=pd.read_csv(data/'training_amplitudes.csv').set_index('tag')
    xs=[];ys=[];records=[]
    for block in [f'B{i:04d}' for i in range(10,20)]:
        base=obs.loc[block+'_replay']
        for group,channels in cfg['groups'].items():
            for hours in cfg['duration_hours']:
                for exponent in cfg['damage_exponents']:
                    sid=f'AUX_{block}_{group}_h{hours}_p{exponent:g}'
                    onset=96+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%49
                    endpoint=onset+hours*4;n=endpoint+1;steps=np.arange(n)
                    damage=np.clip((steps-onset)*.25/hours,0,1)**exponent
                    x=base.iloc[:n].copy()
                    for channel,sign in channels.items():
                        x[channel]+=sign*amps.loc[channel,'amplitude']*damage
                    for channel in x:
                        assert x[channel].between(*bounds[amps.loc[channel,'kind']]).all()
                    x.index=pd.MultiIndex.from_arrays([[sid]*n,steps],names=['scene_id','step'])
                    xs.append(x)
                    ys.append(pd.DataFrame(dict(RUL_ref_hours=(endpoint-steps)*.25,rul_label_valid=steps>onset),index=x.index))
                    records.append(dict(scene_id=sid,block_id=block,group=group,nominal_hours=hours,
                        curve_exponent=exponent,onset=onset,endpoint=endpoint,n=n,
                        first_source_row=int(manifests.loc[block+'_replay','first_source_row']),
                        label_provenance='auxiliary_complete_synthetic_not_recovered_censoring'))
    out=root/'rul_balanced_validation_v1/auxiliary_data'
    out.mkdir(parents=True,exist_ok=True)
    pd.concat(xs).to_csv(out/'observations.csv.gz')
    pd.concat(ys).to_csv(out/'labels.csv.gz')
    pd.DataFrame(records).set_index('scene_id').to_csv(out/'manifest.csv')
    print('Built 360 auxiliary trajectories',flush=True)
