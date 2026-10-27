from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data';SOURCE=ROOT.parent/'dataset_v1/data'


def kind(c):
    return 'temperature' if 'TI' in c else 'vibration' if 'XI' in c else 'axial' if 'ZI' in c else 'speed' if 'SI' in c else 'differential_pressure' if 'PDI' in c else 'pressure'


def risk(di):return np.where(di>=.85,1,np.where(di>=.65,2,np.where(di>=.4,3,4)))


def main():
    OUT.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'config.json').read_text());original=json.loads((SOURCE/'config_snapshot.json').read_text())
    blocks=pd.read_csv(SOURCE/'source_blocks.csv');meta=pd.read_csv(SOURCE/'raw_metadata.csv').set_index('source_row_id')
    raw=pd.read_csv(SOURCE/'raw_observations.csv.gz').set_index('source_row_id');columns=list(raw)
    # Train-only global amplitudes, fixed across backgrounds and speeds in the controlled main experiment.
    candidates=meta['split'].eq('train') & meta.baseline_candidate
    normal=raw.loc[candidates];scale_rows=[];amplitudes={}
    for c in columns:
        k=kind(c);center=float(normal[c].median());s=float((normal[c]-center).abs().median()*1.4826)
        floor={'temperature':.1,'vibration':.1,'axial':.001,'pressure':.01,'differential_pressure':.01,'speed':1.}[k]
        s=max(s,floor);amp=float(np.clip(s*cfg['amplitude_scale_multiplier'],*original['amplitude_caps'][k])) if k in original['amplitude_caps'] else None
        scale_rows.append(dict(tag=c,kind=k,training_median=center,robust_scale=s,amplitude=amp,training_points=len(normal)))
        if amp is not None:amplitudes[c]=amp
    pd.DataFrame(scale_rows).to_csv(OUT/'training_amplitudes.csv',index=False)
    manifests=[];rejects=[];counts=[]
    for split in cfg['splits_built']:
        rng=np.random.default_rng(cfg['seeds'][split]);observations=[];labels=[];lineages=[];masks=[]
        for b in blocks[blocks['split'].eq(split)].itertuples():
            srcids=np.arange(b.first_source_row,b.last_source_row+1);base=raw.loc[srcids].reset_index(drop=True)
            assert len(base)==480
            specs=[dict(scene_id=f'{b.block_id}_replay',benchmark='replay',group='none',curve_exponent=None,onset=None,nominal_hours=None,
                        observed=False,n=len(base),censored=False,main_scene_id=None,variant='replay',amplitude_multiplier=0,response_power=1.,lag_points=0)]
            for group in cfg['groups']:
                for exponent in cfg['damage_exponents']:
                    onset=int(rng.integers(cfg['onset_range_inclusive'][0],cfg['onset_range_inclusive'][1]+1));duration=int(rng.choice(cfg['duration_hours']))
                    endpoint=onset+duration*4;assert endpoint<len(base)
                    censored=bool(rng.random()<cfg['censor_probability'])
                    n=onset+int(duration*4*rng.uniform(*cfg['censor_fraction_range']))+1 if censored else endpoint+1
                    sid=f'{b.block_id}_{group}_p{exponent:g}_main'
                    main=dict(scene_id=sid,benchmark='main',group=group,curve_exponent=exponent,onset=onset,nominal_hours=duration,
                              observed=not censored,n=n,censored=censored,main_scene_id=sid,variant='main',amplitude_multiplier=1.,response_power=1.,lag_points=0)
                    specs.append(main)
                    if split=='validation' and exponent==1.:
                        for robustness in cfg['robustness']:
                            specs.append(dict(main,scene_id=sid.replace('_main','_'+robustness['name']),benchmark='robustness',variant=robustness['name'],
                                              amplitude_multiplier=robustness['amplitude_multiplier'],response_power=robustness['response_power'],lag_points=robustness['lag_points']))
            for spec in specs:
                n=spec['n'];sid=spec['scene_id'];x=base.iloc[:n].copy();is_replay=spec['benchmark']=='replay'
                d=np.zeros(n) if is_replay else np.minimum(1,np.maximum((np.arange(n)-spec['onset'])*.25/spec['nominal_hours'],0)**spec['curve_exponent'])
                mask=pd.DataFrame(np.zeros((n,len(columns)),dtype=np.uint8),columns=columns)
                channels={};lag=spec['lag_points'];response=np.r_[np.zeros(lag),d][:n]**spec['response_power']
                if not is_replay:
                    for c,sign in cfg['groups'][spec['group']].items():
                        amp=sign*amplitudes[c]*spec['amplitude_multiplier'];x[c]+=amp*response
                        mask[c]=(response>0).astype(np.uint8);channels[c]=amp
                invalid=[c for c in columns if not x[c].between(*original['bounds'][kind(c)]).all()]
                if invalid:
                    rejects.append(dict(spec,block_id=b.block_id,invalid_channels=invalid));continue
                keys=pd.DataFrame(dict(scene_id=sid,step=np.arange(n)))
                observations.append(pd.concat([keys,x],axis=1));masks.append(pd.concat([keys,mask],axis=1))
                di=1-d if not is_replay else np.full(n,np.nan)
                rul=(n-1-np.arange(n))*.25 if spec['observed'] else np.full(n,np.nan)
                labels.append(keys.assign(DI_ref=di,risk_ref=pd.array(risk(di) if not is_replay else [pd.NA]*n,dtype='Int64'),
                                           di_label_valid=not is_replay,RUL_ref_hours=rul,rul_label_valid=(d>0)&spec['observed'],
                                           degradation_active=d>0,event_observed=spec['observed'],right_censored=spec['censored'],
                                           label_provenance='candidate_only' if is_replay else 'controlled_synthetic_assumption'))
                source=srcids[:n]
                lineages.append(keys.assign(source_row_id=source,source_timestamp=meta.loc[source,'source_timestamp'].to_numpy(),block_id=b.block_id,split=split))
                manifests.append(dict(spec,split=split,block_id=b.block_id,first_source_row=int(source[0]),last_source_row=int(source[-1]),
                                      channels=channels,seed=cfg['seeds'][split],failure_step=n-1 if spec['observed'] else None))
        for name,frames in [('observations',observations),('labels',labels),('lineage',lineages),('injection_mask',masks)]:
            pd.concat(frames,ignore_index=True).to_csv(OUT/f'{split}_{name}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        counts.append(dict(split=split,scenes=len(observations),rows=sum(len(g) for g in observations)))
    (OUT/'scene_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    (OUT/'rejected_scenes.json').write_text(json.dumps(rejects,indent=2),encoding='utf-8')
    (OUT/'model_input_columns.json').write_text(json.dumps(columns,indent=2))
    (OUT/'config_snapshot.json').write_text(json.dumps(cfg,indent=2))
    blocks.to_csv(OUT/'source_block_plan.csv',index=False)
    summary=dict(counts=counts,rejected=len(rejects),test_built=False,test_scored=False,training_candidates=len(normal),
                 source_data_checksums_sha256=hashlib.sha256((SOURCE/'checksums.json').read_bytes()).hexdigest())
    (OUT/'build_summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'build_provenance.json').write_text(json.dumps(dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
        sources={str(p.relative_to(ROOT.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'config.json',SOURCE/'checksums.json',SOURCE/'config_snapshot.json']}),indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='checksums.json'}
    (OUT/'checksums.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summary,indent=2));print(pd.DataFrame(manifests).groupby(['split','benchmark','variant']).size().to_string())

if __name__=='__main__':main()
