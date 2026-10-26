"""Final-fit data access: only the frozen training partition and existing augmentation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REPO = WORK.parents[2]
for p in [REPO, REPO/'01_early_warning',
          WORK/'di_context_v1', WORK/'rul_background_augmentation_ablation_v1']:
    sys.path.insert(0, str(p))
from pipelines.baseline_data import windows, tabular_features, risk_levels
from ablation import load_inputs, build_fold, variant_data, features

DATA = WORK/'RCSD-1YD Data Set/controlled_dataset_v2/data'
AUX = WORK/'rul_balanced_validation_v1/auxiliary_data'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def raw_windows(x, timestamps, sensors):
    values = x[sensors].to_numpy(dtype=np.float32).copy()
    values[~x['75SI865R.pv'].between(6050,8500).to_numpy()] = np.nan
    return windows(values, timestamps, 96, 15)

def labeled_windows(inputs, task, blocks, stride=None):
    manifest = inputs['manifest']
    selected = manifest.loc[manifest.split.eq('train') & manifest.benchmark.eq('main') & manifest.block_id.isin(blocks)]
    batches, keysets = [], []
    target = 'DI_ref' if task == 'di' else 'RUL_ref_hours'
    for sid, meta in selected.iterrows():
        x, y, source = (inputs[k].loc[sid] for k in ['observations','labels','lineage'])
        assert source.split.eq('train').all() and source.block_id.eq(meta.block_id).all()
        X, ends = raw_windows(x, source.source_timestamp, inputs['sensors'])
        eligible = y['di_label_valid' if task=='di' else 'rul_label_valid'] & np.isfinite(y[target])
        if task == 'rul':
            eligible &= y.event_observed & ~y.right_censored & y[target].ge(0)
        ends, X = ends[eligible.iloc[ends]], X[eligible.iloc[ends]]
        if not len(ends):
            continue
        if stride:
            take = np.arange(0,len(ends),stride)
            if task == 'rul':
                take = np.unique(np.r_[take,len(ends)-1])
            ends, X = ends[take], X[take]
        keys = pd.DataFrame(dict(scene_id=sid, block_id=meta.block_id, step=x.index[ends],
            source_timestamp=source.iloc[ends].source_timestamp.to_numpy()))
        keys[target] = y.iloc[ends][target].to_numpy(dtype=float)
        batches.append(X); keysets.append(keys)
    return np.concatenate(batches), pd.concat(keysets, ignore_index=True)

def blocks(a,b):
    return [f'B{i:04d}' for i in range(a,b)]

def rul_config():
    cfg = read(WORK/'rul_background_augmentation_ablation_v1/plan.json')
    return dict(cfg, fold_block_ranges=[[20,31]], reference_background_counts=[10])

def audit_auxiliary(inputs):
    cfg = read(DATA.parent/'config.json')
    amps = pd.read_csv(DATA/'training_amplitudes.csv').set_index('tag')
    manifest = inputs['aux_manifest']
    assert len(manifest)==360 and set(manifest.block_id)==set(blocks(10,20))
    assert manifest.groupby('block_id').size().eq(36).all()
    count = 0
    for sid,m in manifest.iterrows():
        x,y = inputs['aux_observations'].loc[sid],inputs['aux_labels'].loc[sid]
        onset = 96+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%49
        endpoint = onset+int(m.nominal_hours)*4
        assert onset==m.onset and endpoint==m.endpoint and len(x)==endpoint+1
        steps = np.arange(endpoint+1)
        np.testing.assert_array_equal(x.index,steps)
        np.testing.assert_array_equal(y.RUL_ref_hours,(endpoint-steps)*.25)
        np.testing.assert_array_equal(y.rul_label_valid,steps>onset)
        damage = np.clip((steps-onset)*.25/m.nominal_hours,0,1)**m.curve_exponent
        expected = inputs['observations'].loc[m.block_id+'_replay'].iloc[:len(x)][inputs['sensors']].copy()
        for tag,sign in cfg['groups'][m.group].items():
            expected[tag] += sign*amps.loc[tag,'amplitude']*damage
        np.testing.assert_allclose(x[inputs['sensors']],expected,rtol=1e-10,atol=1e-9)
        assert m.label_provenance=='auxiliary_complete_synthetic_not_recovered_censoring'
        count += len(x)
    return dict(scenes=360,rows=count,reconstructed=True,source_blocks=blocks(10,20))
