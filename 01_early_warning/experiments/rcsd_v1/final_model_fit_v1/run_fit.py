"""Fit the predeclared final assets; no validation/test observations are opened."""
import gc
import math
import pickle
import platform
import time
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from data import (ROOT,WORK,DATA,AUX,read,write,digest,import_file,load_inputs,
    labeled_windows,blocks,rul_config,build_fold,variant_data,tabular_features,audit_auxiliary)
from inference import load_model,predict_array
from common_algorithms.models.lstm import LSTMRegressor
from common_algorithms.models.gradient_boosting import GradientBoostingRegressor

OUT = ROOT/'results'

def sources():
    from portable import training_fingerprints
    return training_fingerprints(WORK)


def save_keys(name,keys):
    assert not keys.duplicated(['scene_id','step']).any()
    keys.to_csv(OUT/f'{name}_keys.csv.gz',index=False)

def save_model(model,path,name):
    path.parent.mkdir(parents=True,exist_ok=True)
    if name in ['lstm_window','gradient_boosting_history']:
        model.save(path)
    else:
        with path.open('wb') as f:
            pickle.dump(model,f,protocol=5)

def fit_one(task,name,seed,variant,X,keys,inner=None):
    asset = f'{name}__{seed}' if task=='di' else f'{name}__{variant}__{seed}'
    ext = '.pt' if name=='lstm_window' else '.pkl'
    path = ROOT/'models'/task/(asset+ext)
    done = OUT/f'{task}__{asset}.json'
    if done.exists():
        record = read(done)
        assert record['model_sha256']==digest(path)
        print(f'resume verified {task}/{asset}',flush=True)
        return record
    cfg = read(WORK/('paper_baseline_comparison_v1/plan.json' if task=='di' else 'rul_matched_baselines_v1/plan.json'))
    y = keys['DI_ref' if task=='di' else 'RUL_ref_hours'].to_numpy()
    start = time.monotonic()
    record = dict(task=task,asset=asset,name=name,variant=variant,seed=seed,
        fit_points=len(keys),fit_scenes=int(keys.scene_id.nunique()),fit_backgrounds=int(keys.block_id.nunique()))
    print(f'start {task}/{asset}: {len(X)} fit samples',flush=True)
    if name in ['constant','median']:
        model = dict(format='constant_v1',value=float(np.median(y)),statistic='training_target_median')
    elif name.startswith('ridge_'):
        model = make_pipeline(StandardScaler(),Ridge(alpha=cfg['ridge_alpha'])).fit(X,y)
    elif name.startswith('trees_'):
        model = ExtraTreesRegressor(**cfg['trees'],random_state=seed,n_jobs=1).fit(X,y)
    elif name=='gradient_boosting_history':
        model = GradientBoostingRegressor(**cfg['gradient_boosting'],random_state=seed).fit(X,y)
    elif name=='lstm_window':
        IX,ik,V,vk = inner
        policy_root = 'lstm_early_stopping_v1' if task=='di' else 'rul_matched_baselines_v1'
        policy = read(WORK/policy_root/'plan.json')
        params = dict(policy['lstm'],random_state=seed)
        selector = import_file(task+'_epoch_selector',WORK/policy_root/'epoch_selection.py')
        selected,trace = selector.select_epochs(IX,ik[yname(task)].to_numpy(),V,vk,params,policy['stopping'])
        selected.save(OUT/f'{task}__{asset}__inner.pt')
        pd.DataFrame(trace).to_csv(OUT/f'{task}__{asset}__selection.csv',index=False)
        innerpred = vk.copy();innerpred['raw_prediction'] = selected.predict(V)
        innerpred.to_csv(OUT/f'{task}__{asset}__inner_predictions.csv.gz',index=False)
        epochs = selected.params['epochs']
        record.update(selected_epochs=int(epochs),selection_ran_epochs=len(trace),
            hit_budget=not trace[-1]['stopped_by_patience'],inner_fit_points=len(ik),inner_validation_points=len(vk))
        del selected
        print(f'{task}/{asset}: selected {epochs}, ran {len(trace)}; refitting',flush=True)
        model = LSTMRegressor(**params,epochs=epochs).fit(X,y)
        write(OUT/f'{task}__{asset}__refit_loss.json',model.loss_history_)
    else:
        raise ValueError(name)
    save_model(model,path,name)
    predictions = predict_array(model,X)
    restored = load_model(path,name)
    np.testing.assert_allclose(predict_array(restored,X),predictions,rtol=1e-7,atol=1e-7)
    assert np.isfinite(predictions).all()
    replay = keys.copy();replay['raw_prediction'] = predictions
    replay['prediction'] = np.clip(predictions,0,1) if task=='di' else np.maximum(predictions,0)
    replay.to_csv(OUT/f'{task}__{asset}__fit_replay.csv.gz',index=False)
    record.update(model_path=str(path.relative_to(WORK)),model_sha256=digest(path),elapsed_seconds=time.monotonic()-start)
    write(done,record)
    print(f'done {task}/{asset}: {record["elapsed_seconds"]:.1f}s',flush=True)
    return record

def yname(task):
    return 'DI_ref' if task=='di' else 'RUL_ref_hours'

def calibrate(tables,records):
    radii,diagnostics = {},[]
    for row in records:
        if row['task']!='rul' or row['variant']!='residual_augmented':
            continue
        model = load_model(WORK/row['model_path'],row['name'])
        frame = tables['evaluation']['keys'].copy()
        raw = predict_array(model,tables['evaluation']['residual'])
        pred = np.maximum(raw,0)
        frame['raw_prediction'],frame['prediction'] = raw,pred
        frame['absolute_error'] = np.abs(pred-frame.RUL_ref_hours)
        scores = frame.groupby('block_id').absolute_error.max().sort_index()
        assert set(scores.index)==set(blocks(20,31))
        n = len(scores);rank = math.ceil((n+1)*.9)
        radius = float(np.sort(scores)[rank-1]) if rank<=n else None
        q = float('inf') if radius is None else radius
        frame['lower_hours'],frame['upper_hours'] = np.maximum(0,pred-q),pred+q
        frame['covered'] = frame.RUL_ref_hours.between(frame.lower_hours,frame.upper_hours)
        frame['width_hours'] = frame.upper_hours-frame.lower_hours
        frame.to_csv(OUT/f'rul__{row["asset"]}__calibration.csv.gz',index=False)
        scores.rename('max_absolute_error_hours').to_csv(OUT/f'rul__{row["asset"]}__background_scores.csv')
        info = dict(radius_hours=radius,background_count=n,order_statistic_rank=rank,nominal_coverage=.9,
            interpretation='calibration diagnostics only; no independent performance or field coverage guarantee')
        radii[row['asset']] = info
        diagnostics.append(dict(asset=row['asset'],**info,points=len(frame),scenes=int(frame.scene_id.nunique()),
            point_coverage=float(frame.covered.mean()),
            whole_scene_coverage=float(frame.groupby('scene_id').covered.all().mean()),
            whole_background_coverage=float(frame.groupby('block_id').covered.all().mean()),
            mean_width_hours=float(frame.width_hours.mean()),median_width_hours=float(frame.width_hours.median())))
    assert len(radii)==4
    write(ROOT/'models/rul/interval_radii.json',radii)
    pd.DataFrame(diagnostics).to_csv(OUT/'interval_calibration_diagnostics.csv',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    provenance = dict(sha256=sources(),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        sklearn=sklearn.__version__,torch=torch.__version__,development_validation_read=False,final_test_read=False)
    p = OUT/'provenance.json'
    if p.exists():
        assert read(p)==provenance,'Source/runtime changed during a resumable fit'
    else:
        write(p,provenance)
    protocol = read(WORK/'final_evaluation_protocol_v1/protocol.json')
    write(OUT/'frozen_protocol.json',protocol)
    inputs = load_inputs(DATA,AUX)
    write(OUT/'auxiliary_reconstruction.json',audit_auxiliary(inputs))
    print('360 existing auxiliary scenes reconstructed successfully',flush=True)
    write(OUT/'preprocessing.json',dict(sensors=inputs['sensors'],history_points=96,sampling_minutes=15,
        running_speed_range=[6050,8500],risk_thresholds=[.85,.65,.4],
        feature_order=['current','last24_mean','current_minus_first_of_last24','last96_mean','current_minus_first_of_last96'],
        invalid_windows='NaN output; gaps/missing/stopped rows reset 96-point history',
        label_columns_are_not_features=True,interval_lower_bound=0))
    records=[]
    with threadpool_limits(limits=2):
        X,keys = labeled_windows(inputs,'di',blocks(0,31),4)
        V,vk = labeled_windows(inputs,'di',blocks(24,31))
        mask = keys.block_id.isin(blocks(0,24)).to_numpy()
        inner = X[mask],keys.loc[mask].reset_index(drop=True),V,vk
        save_keys('di_fit',keys);save_keys('di_inner_fit',inner[1]);save_keys('di_inner_validation',vk)
        F = tabular_features(X)
        cfg = read(WORK/'paper_baseline_comparison_v1/plan.json')
        for name in cfg['candidates']:
            seeds = protocol['seeds'] if name.startswith('trees_') or name=='lstm_window' else [protocol['reference_seed']]
            matrix = X if name=='lstm_window' else (X[:,-1] if name.endswith('_current') else F)
            for seed in seeds:
                records.append(fit_one('di',name,seed,'raw',matrix,keys,inner))
        del X,V,F,inner,matrix;gc.collect()
        cfg = rul_config()
        tables,reference,refkeys,split = build_fold(inputs,cfg,0)
        write(OUT/'rul_split.json',split)
        for role,table in tables.items():
            save_keys('rul_'+role,table['keys'])
        save_keys('rul_reference',refkeys)
        (ROOT/'models/rul').mkdir(parents=True,exist_ok=True)
        with (ROOT/'models/rul/reference.pkl').open('wb') as f:
            pickle.dump(reference,f,protocol=5)
        X,keys = labeled_windows(inputs,'rul',blocks(10,20),4)
        pd.testing.assert_frame_equal(keys,tables['original_fit']['keys'])
        np.testing.assert_array_equal(tabular_features(X),tables['original_fit']['raw'])
        V,vk = labeled_windows(inputs,'rul',blocks(17,20))
        mask = keys.block_id.isin(blocks(10,17)).to_numpy()
        inner = X[mask],keys.loc[mask].reset_index(drop=True),V,vk
        save_keys('rul_inner_fit',inner[1]);save_keys('rul_inner_validation',vk)
        rawcfg = read(WORK/'rul_matched_baselines_v1/plan.json')
        for name in rawcfg['candidates']:
            matrix = X if name=='lstm_window' else tables['original_fit']['raw']
            seeds = protocol['seeds'] if name in ['trees_history','lstm_window'] else [protocol['reference_seed']]
            for seed in seeds:
                records.append(fit_one('rul',name,seed,'raw_original',matrix,keys,inner))
        del X,V,inner,matrix;gc.collect()
        for variant in protocol['rul']['ablation_variants'][1:]:
            F,keys,_,_ = variant_data(tables,variant)
            for name in protocol['rul']['ablation_models']:
                seeds = protocol['seeds'] if name=='trees_history' else [protocol['reference_seed']]
                for seed in seeds:
                    records.append(fit_one('rul',name,seed,variant,F,keys))
        calibrate(tables,records)
    assert len(records)==34
    pd.DataFrame(records).to_csv(OUT/'model_registry.csv',index=False)
    plan = pd.read_csv(WORK/'final_evaluation_protocol_v1/results/model_asset_plan.csv').dropna(subset=['target'])
    assert len(plan)==36 and all((WORK/p).is_file() for p in plan.target)
    assert sources()==provenance['sha256']
    write(OUT/'fit_status.json',dict(status='FITTED_CALIBRATED_PENDING_INDEPENDENT_VERIFICATION',
        di_models=13,rul_point_models=21,reference_models=1,interval_models=4,
        final_scoring_ready=False,final_test_read=False))
    print('All 34 point models fitted; reference and four interval calibrations saved.',flush=True)

if __name__=='__main__':
    main()
