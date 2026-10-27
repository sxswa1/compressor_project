"""Reconstruct every prepared window independently from the original RCSD table."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT.parent/'公开数据集/Centrifugal compressor _2022.xlsx')
    p.add_argument('--prepared-dir',type=Path,default=ROOT/'data/RCSD_semisynthetic_diagnosis_v1')
    p.add_argument('--output',type=Path,default=ROOT/'02_few_shot_diagnosis/outputs/rcsd_preparation_audit_v1.json')
    args=p.parse_args();prep=json.loads((args.prepared_dir/'preparation.json').read_text());cfg=prep['config']
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=cfg['source_sha256']:raise ValueError('Raw source differs')
    raw=pd.read_excel(args.source,sheet_name='Sheet1');times=pd.to_datetime(raw.pop('Timestamp'));raw=raw.astype(float)
    meta=pd.read_csv(args.prepared_dir/'sample_manifest.csv')
    x=np.load(args.prepared_dir/'windows.npz',allow_pickle=False)['x']
    amp=pd.read_csv(args.prepared_dir/'training_amplitudes.csv').set_index('tag').amplitude
    blocks=pd.read_csv(args.prepared_dir/'background_blocks.csv').set_index('block_id')
    if len(x)!=344 or len(meta)!=344 or len(blocks)!=64:raise ValueError('Planned background/window counts differ')
    if not meta.groupby('block_id')['split'].nunique().eq(1).all():raise ValueError('A background crosses roles')
    if not meta.groupby(['block_id','variant']).class_id.nunique().eq(4).all():raise ValueError('Missing classes')
    reconstructed=0;changed_cells=0
    for row in meta.itertuples():
        block=blocks.loc[row.block_id];start=int(row.first_source_row);end=int(row.last_source_row)
        if start!=int(block.first_source_row)+cfg['window_start_point'] or end-start+1!=cfg['window_points']:
            raise ValueError('Window selection changed')
        source=raw.iloc[start:end+1].copy();expected=source.copy()
        indices=np.arange(cfg['window_start_point'],cfg['window_start_point']+cfg['window_points'])
        variant=cfg['variants'][row.variant]
        progress=np.maximum(0.,np.minimum(1.,(indices-variant['lag_points']-row.onset)*.25/row.duration_hours))
        response=progress**(row.exponent*variant['response_power'])
        for tag,sign in cfg['groups'][row.class_id].items():
            expected[tag]=expected[tag]+sign*amp.loc[tag]*variant['amplitude_multiplier']*response
        numeric=expected.to_numpy(dtype=np.float32).T
        if not np.array_equal(x[row.sample_id],numeric):raise ValueError('Reconstruction differs at sample '+str(row.sample_id))
        if row.class_id and not (response>0).all():raise ValueError('Anomaly label on an unmodified prefix')
        if not row.class_id and not np.array_equal(numeric,source.to_numpy(dtype=np.float32).T):raise ValueError('Background was modified')
        if str(times.iloc[start])!=row.start_time or str(times.iloc[end])!=row.end_time:
            raise ValueError('Original time provenance differs')
        if not times.iloc[start:end+1].diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():raise ValueError('Time gap')
        changed_cells+=int(np.count_nonzero(expected.to_numpy()!=source.to_numpy()));reconstructed+=1
    duplicate_groups=meta.groupby('input_sha256').agg(n=('sample_id','size'),blocks=('block_id','nunique'),roles=('split','nunique'))
    if duplicate_groups.roles.gt(1).any() or duplicate_groups.blocks.gt(1).any():raise ValueError('Numeric content duplicates across distinct source backgrounds/roles')
    result={'scope':'all prepared windows reconstructed from original Excel without calling construction functions',
        'windows_reconstructed':reconstructed,'source_backgrounds':64,'all_labels_and_times_match':True,
        'non_injected_values_preserved':True,'changed_scalar_cells_across_prepared_windows':changed_cells,
        'content_duplicates_crossing_background_or_split':0,'same_background_repeated_controls':int(duplicate_groups[duplicate_groups.n.gt(1)]['n'].sum()),
        'repeated_controls_are_not_new_independent_events':True,'raw_source_unchanged':hashlib.sha256(args.source.read_bytes()).hexdigest()==cfg['source_sha256'],
        'natural_fault_incidents':0,'field_accuracy_verified':False,'raw_xlsx_included_in_delivery':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
