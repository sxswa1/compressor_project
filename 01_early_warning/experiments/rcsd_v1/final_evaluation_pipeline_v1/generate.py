"""Same controlled/stress recipes, frozen amplitudes, explicit inclusion ledger."""
import argparse
import numpy as np
import pandas as pd
from common import ROOT,WORK,D1,D2,ST,PROTOCOL,read,write,assert_release

def category(c):
    return 'temperature' if 'TI' in c else 'vibration' if 'XI' in c else 'axial' if 'ZI' in c else 'speed' if 'SI' in c else 'differential_pressure' if 'PDI' in c else 'pressure'

def load_raw(split):
    if split not in ['train','validation','test']:raise ValueError(split)
    if split=='test':assert_release()
    meta=pd.read_csv(D1/'raw_metadata.csv').set_index('source_row_id')
    stop={'train':'2022-07-01','validation':'2022-09-01','test':'2023-01-01'}[split]
    allowed=meta.loc[pd.to_datetime(meta.source_timestamp).lt(stop)]
    assert np.array_equal(allowed.index,np.arange(len(allowed)))
    raw=pd.read_csv(D1/'raw_observations.csv.gz',nrows=len(allowed)).set_index('source_row_id')
    assert raw.index.equals(allowed.index)
    return raw,allowed

def controlled(raw,meta,split,ledger):
    cfg=read(D2.parent/'config.json');bounds=read(D1/'config_snapshot.json')['bounds']
    seed=read(PROTOCOL)['dataset']['test_seed'] if split=='test' else cfg['seeds'][split]
    rng=np.random.default_rng(seed)
    amps=pd.read_csv(D2/'training_amplitudes.csv',float_precision='round_trip').set_index('tag').amplitude
    blocks=pd.read_csv(D1/'source_blocks.csv')
    for b in blocks.loc[blocks.split.eq(split)].itertuples():
        ids=np.arange(b.first_source_row,b.last_source_row+1)
        assert meta.loc[ids,'split'].eq(split).all()
        base=raw.loc[ids].reset_index(drop=True);assert len(base)==480
        specs=[dict(scene_id=f'{b.block_id}_replay',benchmark='replay',group='none',curve_exponent=None,onset=None,nominal_hours=None,
            observed=False,n=len(base),censored=False,main_scene_id=None,variant='replay',amplitude_multiplier=0,response_power=1.,lag_points=0)]
        for group in cfg['groups']:
            for exponent in cfg['damage_exponents']:
                onset=int(rng.integers(cfg['onset_range_inclusive'][0],cfg['onset_range_inclusive'][1]+1))
                duration=int(rng.choice(cfg['duration_hours']));endpoint=onset+duration*4
                assert endpoint<len(base)
                censored=bool(rng.random()<cfg['censor_probability'])
                n=onset+int(duration*4*rng.uniform(*cfg['censor_fraction_range']))+1 if censored else endpoint+1
                sid=f'{b.block_id}_{group}_p{exponent:g}_main'
                main=dict(scene_id=sid,benchmark='main',group=group,curve_exponent=exponent,onset=onset,nominal_hours=duration,
                    observed=not censored,n=n,censored=censored,main_scene_id=sid,variant='main',amplitude_multiplier=1.,response_power=1.,lag_points=0)
                specs.append(main)
                if split in ['validation','test'] and exponent==1.:
                    for r in cfg['robustness']:
                        specs.append(dict(main,scene_id=sid.replace('_main','_'+r['name']),benchmark='robustness',variant=r['name'],
                            amplitude_multiplier=r['amplitude_multiplier'],response_power=r['response_power'],lag_points=r['lag_points']))
        for spec in specs:
            n=spec['n'];sid=spec['scene_id'];replay=spec['benchmark']=='replay'
            x=base.iloc[:n].copy();steps=np.arange(n)
            d=np.zeros(n) if replay else np.minimum(1,np.maximum((steps-spec['onset'])*.25/spec['nominal_hours'],0)**spec['curve_exponent'])
            mask=pd.DataFrame(np.zeros(x.shape,dtype=np.uint8),columns=x.columns)
            response=np.r_[np.zeros(spec['lag_points']),d][:n]**spec['response_power'];channels={}
            if not replay:
                for c,sign in cfg['groups'][spec['group']].items():
                    a=sign*amps[c]*spec['amplitude_multiplier'];x[c]+=a*response
                    mask[c]=(response>0).astype(np.uint8);channels[c]=float(a)
            invalid=[c for c in x if not x[c].between(*bounds[category(c)]).all()]
            ledger.append(dict(source='v2',scene_id=sid,block_id=b.block_id,split=split,status='rejected' if invalid else 'included',
                invalid_channels=invalid,planned_rows=n,benchmark=spec['benchmark'],variant=spec['variant']))
            if invalid:continue
            di=1-d if not replay else np.full(n,np.nan)
            risk=np.where(di>=.85,1,np.where(di>=.65,2,np.where(di>=.4,3,4)))
            y=pd.DataFrame(dict(DI_ref=di,risk_ref=pd.array(risk if not replay else [pd.NA]*n,dtype='Int64'),
                di_label_valid=not replay,RUL_ref_hours=(n-1-steps)*.25 if spec['observed'] else np.full(n,np.nan),
                rul_label_valid=(d>0)&spec['observed'],degradation_active=d>0,event_observed=spec['observed'],right_censored=spec['censored'],
                label_provenance='candidate_only' if replay else 'controlled_synthetic_assumption'))
            lineage=pd.DataFrame(dict(source_row_id=ids[:n],source_timestamp=meta.loc[ids[:n],'source_timestamp'].to_numpy(),block_id=b.block_id,split=split))
            m=dict(spec,split=split,block_id=b.block_id,first_source_row=int(ids[0]),last_source_row=int(ids[n-1]),channels=channels,
                seed=seed,failure_step=n-1 if spec['observed'] else None)
            yield m,x,y,lineage,mask

def stress_blocks(meta,split):
    cfg=read(WORK/'early_warning_v2/stress_test/config.json')
    subset=meta.loc[meta.split.eq(split)&meta.baseline_candidate];result=[]
    for seg,g in subset.groupby('segment_id'):
        for start in range(0,len(g)-cfg['block_points']+1,cfg['block_points']):
            ids=g.index[start:start+cfg['block_points']].to_numpy()
            assert np.all(np.diff(ids)==1)
            assert pd.to_datetime(g.loc[ids,'source_timestamp']).diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
            result.append(dict(block_id=f'{"TSB" if split=="test" else "SB"}{len(result):02d}',segment_id=int(seg),first_row=int(ids[0]),last_row=int(ids[-1])))
    return result

def stress(raw,meta,split,ledger):
    cfg=read(WORK/'early_warning_v2/stress_test/config.json');dcfg=read(D1/'config_snapshot.json')
    scales=pd.read_csv(D1/'training_scales.csv').set_index(['speed_band','tag'])
    groups=read(D2.parent/'config.json')['groups'];counter=0
    for block in stress_blocks(meta,split):
        ids=np.arange(block['first_row'],block['last_row']+1);base=raw.loc[ids].reset_index(drop=True)
        assert meta.loc[ids,'split'].eq(split).all()
        band=int(np.searchsorted(dcfg['speed_band_edges'],base['75SI865R.pv'].median(),side='right')-1)
        specs=[('control','none',None,None)]
        for group in groups:
            for amp in cfg['amplitude_multipliers']:
                specs.extend(('ramp_plateau',group,amp,d) for d in cfg['ramp_hours'])
                specs.append(('abnormal_start',group,amp,None))
        for channel in cfg['spike_channels']:
            for amp in cfg['spike_scale_multipliers']:
                specs.extend(('spike',channel,amp,d) for d in cfg['spike_points'])
        for kind,group,strength,duration in specs:
            sid=f'{"TST" if split=="test" else "ST"}{counter:04d}';counter+=1
            onset=cfg['prelude_points'];endpoint=None;d=None
            if kind=='ramp_plateau':
                endpoint=onset+int(duration*4);n=endpoint+1+int(cfg['plateau_hours']*4)
                d=np.clip((np.arange(n)-onset)/(duration*4),0,1)
            elif kind=='abnormal_start':n=cfg['startup_points'];onset=0;d=np.ones(n)
            elif kind=='spike':n=cfg['spike_scene_points']
            else:n=cfg['block_points'];onset=None
            assert n<=len(base)
            x=base.iloc[:n].copy();mask=np.zeros(x.shape,dtype=np.uint8);amplitudes={}
            if kind in ['ramp_plateau','abnormal_start']:
                for c,sign in groups[group].items():
                    a=sign*strength*float(np.clip(cfg['training_scale_multiplier']*scales.loc[(band,c),'scale'],*dcfg['amplitude_caps'][category(c)]))
                    x[c]+=a*d;amplitudes[c]=a;mask[:,x.columns.get_loc(c)]=d>0
            elif kind=='spike':
                a=float(strength*scales.loc[(band,group),'scale'])*(-1 if 'PI' in group else 1)
                x.loc[onset:onset+duration-1,group]+=a;amplitudes[group]=a;mask[onset:onset+duration,x.columns.get_loc(group)]=1
            invalid=[c for c in amplitudes if not x[c].between(*dcfg['bounds'][category(c)]).all()]
            ledger.append(dict(source='stress',scene_id=sid,block_id=block['block_id'],split=split,status='rejected' if invalid else 'included',
                invalid_channels=invalid,planned_rows=n,benchmark=kind,variant=kind))
            if invalid:continue
            np.testing.assert_array_equal(x.to_numpy()[mask==0],base.iloc[:n].to_numpy()[mask==0])
            y=pd.DataFrame(dict(source_row_id=ids[:n],constructed_change=mask.any(axis=1),d_assumed=d if d is not None else np.full(n,np.nan)))
            m=dict(scene_id=sid,**block,kind=kind,group=group,strength=strength,duration=duration,onset_step=onset,reference_endpoint_step=endpoint,
                rows=n,speed_band=band,source_start=str(meta.loc[ids[0],'source_timestamp']),source_end=str(meta.loc[ids[n-1],'source_timestamp']),
                amplitudes=amplitudes,interpretation='experimental offset or change; not measured failure or RUL truth')
            lineage=pd.DataFrame(dict(source_row_id=ids[:n],source_timestamp=meta.loc[ids[:n],'source_timestamp'].to_numpy(),block_id=block['block_id'],split=split))
            yield m,x,y,lineage,pd.DataFrame(mask,columns=x.columns)

def keyed(sid,frame):
    result=frame.copy();result.insert(0,'step',np.arange(len(frame)));result.insert(0,'scene_id',sid);return result

def save_scenes(iterator,out,prefix):
    import gzip
    out.mkdir(parents=True,exist_ok=True);names=['observations','labels','lineage','injection_mask']
    handles={name:gzip.open(out/f'{prefix}_{name}.csv.gz','wt',encoding='utf-8',newline='') for name in names}
    manifests=[]
    try:
        for m,*frames in iterator:
            for name,frame in zip(names,frames):keyed(m['scene_id'],frame).to_csv(handles[name],index=False,header=not manifests)
            manifests.append(m)
    finally:
        for h in handles.values():h.close()
    write(out/f'{prefix}_manifest.json',manifests)
    return len(manifests)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reserved',action='store_true',required=True)
    parser.parse_args();assert_release()
    out=WORK/read(PROTOCOL)['dataset']['output_directory']
    if out.exists():raise FileExistsError('Reserved output already exists; preserve original outputs and version any bug-driven rerun')
    raw,meta=load_raw('test');ledger=[]
    counts=dict(controlled=save_scenes(controlled(raw,meta,'test',ledger),out,'controlled'),
        stress=save_scenes(stress(raw,meta,'test',ledger),out,'stress'))
    write(out/'inclusion_ledger.json',ledger)
    assert len(ledger)==550+294
    write(out/'build_status.json',dict(status='GENERATED_NOT_SCORED',split='test',planned=len(ledger),counts=counts,
        rejected=sum(r['status']=='rejected' for r in ledger),release_manifest_sha256=__import__('common').digest(ROOT/'release_manifest.json')))
    write(out/'checksums.json',{p.name:__import__('common').digest(p) for p in out.iterdir() if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':main()
