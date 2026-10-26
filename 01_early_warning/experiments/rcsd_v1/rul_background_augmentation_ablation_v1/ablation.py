"""Fold-local reference fitting and paired raw/residual features, no oracle inputs."""
import json
import numpy as np
import pandas as pd
from pipelines.baseline_data import windows,tabular_features
from context_model import fit_reference


def load_inputs(data_dir,aux_dir):
    tables={}
    for kind in ['observations','labels','lineage']:
        tables[kind]=pd.read_csv(data_dir/f'train_{kind}.csv.gz').set_index(['scene_id','step'])
        assert not tables[kind].index.has_duplicates
    assert tables['observations'].index.equals(tables['labels'].index)
    assert tables['observations'].index.equals(tables['lineage'].index)
    tables['manifest']=pd.DataFrame(json.loads((data_dir/'scene_manifest.json').read_text(encoding='utf-8'))).set_index('scene_id')
    tables['sensors']=json.loads((data_dir/'model_input_columns.json').read_text(encoding='utf-8'))
    tables['aux_observations']=pd.read_csv(aux_dir/'observations.csv.gz').set_index(['scene_id','step'])
    tables['aux_labels']=pd.read_csv(aux_dir/'labels.csv.gz').set_index(['scene_id','step'])
    tables['aux_manifest']=pd.read_csv(aux_dir/'manifest.csv').set_index('scene_id')
    assert tables['aux_observations'].index.equals(tables['aux_labels'].index)
    assert not tables['aux_observations'].index.has_duplicates
    return tables


def features(x,timestamps,sensors,reference,cfg):
    values=x[sensors].to_numpy(dtype=float).copy()
    running=x['75SI865R.pv'].between(6050,8500).to_numpy()
    good=np.isfinite(values).all(axis=1)&running
    values[~good]=np.nan
    if reference is not None and good.any():
        positions=[sensors.index(t) for t in reference['targets']]
        prediction=reference['model'].predict(x.loc[good,reference['context']])*reference['scale']+reference['center']
        values[np.ix_(good,positions)]-=prediction
    X,ends=windows(values,timestamps,cfg['history_points'],cfg['sampling_minutes'])
    return tabular_features(X),ends


def build_fold(inputs,cfg,fold):
    obs,labels,lineage,manifest=(inputs[k] for k in ['observations','labels','lineage','manifest'])
    ax,ay,am=(inputs[k] for k in ['aux_observations','aux_labels','aux_manifest'])
    sensors=inputs['sensors']
    main=manifest.loc[manifest.split.eq('train')&manifest.benchmark.eq('main')]
    blocks=main.groupby('block_id').first_source_row.min().sort_values().index.tolist()
    end,stop=cfg['fold_block_ranges'][fold];n=cfg['reference_background_counts'][fold]
    assert 0<n<end<stop<=len(blocks)
    reference_blocks,rul_fit_blocks,eval_blocks=blocks[:n],blocks[n:end],blocks[end:stop]
    limits=[]
    for ids in [reference_blocks,rul_fit_blocks,eval_blocks]:
        replay=manifest.loc[[b+'_replay' for b in ids]]
        assert replay.split.eq('train').all()
        limits.append((int(replay.first_source_row.min()),int(replay.last_source_row.max())))
    assert limits[0][1]<limits[1][0] and limits[1][1]<limits[2][0]
    normal,refkeys=[],[]
    for block in reference_blocks:
        sid=block+'_replay';x=obs.loc[sid];source=lineage.loc[sid]
        good=np.isfinite(x[sensors]).all(axis=1)&x['75SI865R.pv'].between(6050,8500)
        normal.append(x.loc[good,sensors])
        refkeys.append(pd.DataFrame(dict(scene_id=sid,block_id=block,step=x.index[good],source_timestamp=source.loc[good,'source_timestamp'].to_numpy())))
    normal=pd.concat(normal,ignore_index=True)
    reference=fit_reference(normal,cfg['reference_targets'])
    assert not set(reference['context'])&set(reference['targets'])
    reference_keys=pd.concat(refkeys,ignore_index=True)
    selected_main=main.loc[main.block_id.isin(rul_fit_blocks+eval_blocks)]
    selected_aux=am.loc[am.block_id.isin(rul_fit_blocks)]
    assert len(selected_aux)==36*len(rul_fit_blocks)
    frames={role:{name:[] for name in ['raw','residual','keys']} for role in ['original_fit','auxiliary_fit','evaluation']}
    for auxiliary,selected in [(False,selected_main),(True,selected_aux)]:
        for sid,meta in selected.iterrows():
            block=meta.block_id
            x=ax.loc[sid] if auxiliary else obs.loc[sid]
            lab=ay.loc[sid] if auxiliary else labels.loc[sid]
            source=lineage.loc[block+'_replay'].loc[x.index] if auxiliary else lineage.loc[sid]
            assert source.block_id.eq(block).all() and source.split.eq('train').all()
            raw,ends=features(x,source.source_timestamp,sensors,None,cfg)
            residual,rends=features(x,source.source_timestamp,sensors,reference,cfg)
            np.testing.assert_array_equal(ends,rends)
            valid=lab.rul_label_valid.eq(True)&np.isfinite(lab.RUL_ref_hours)&lab.RUL_ref_hours.ge(0)
            if not auxiliary:
                assert not (valid&(~lab.event_observed|lab.right_censored)).any()
            keep=valid.iloc[ends].to_numpy()
            raw,residual,ends=raw[keep],residual[keep],ends[keep]
            if not len(ends):
                continue
            role='auxiliary_fit' if auxiliary else ('original_fit' if block in rul_fit_blocks else 'evaluation')
            if role!='evaluation':
                take=np.unique(np.r_[np.arange(0,len(ends),cfg['training_stride']),len(ends)-1])
                raw,residual,ends=raw[take],residual[take],ends[take]
            keys=pd.DataFrame(dict(scene_id=sid,block_id=block,step=x.index[ends],
                source_timestamp=source.iloc[ends].source_timestamp.to_numpy(),RUL_ref_hours=lab.iloc[ends].RUL_ref_hours.to_numpy(dtype=float)))
            frames[role]['raw'].append(raw);frames[role]['residual'].append(residual);frames[role]['keys'].append(keys)
    tables={role:dict(raw=np.concatenate(v['raw']),residual=np.concatenate(v['residual']),keys=pd.concat(v['keys'],ignore_index=True)) for role,v in frames.items()}
    split=dict(fold=fold,reference_blocks=reference_blocks,rul_fit_blocks=rul_fit_blocks,eval_blocks=eval_blocks,
               reference_source_range=limits[0],rul_fit_source_range=limits[1],eval_source_range=limits[2],
               auxiliary_scenes=selected_aux.index.tolist())
    return tables,reference,reference_keys,split


def variant_data(tables,variant):
    channel='residual' if variant.startswith('residual_') else 'raw'
    sources=['original_fit','auxiliary_fit'] if variant.endswith('_augmented') else ['original_fit']
    X=np.concatenate([tables[role][channel] for role in sources])
    keys=pd.concat([tables[role]['keys'] for role in sources],ignore_index=True)
    return X,keys,tables['evaluation'][channel],tables['evaluation']['keys']
