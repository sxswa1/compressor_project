from pathlib import Path
import json
import pandas as pd
DATA=Path(__file__).resolve().parent/'data'


def load_scenes(split='train',benchmark='main'):
    if split not in ['train','validation']:raise ValueError('Only train and development validation have been built')
    if benchmark not in ['main','robustness','replay']:raise ValueError('Choose one explicitly named benchmark')
    manifest=json.loads((DATA/'scene_manifest.json').read_text());ids={m['scene_id'] for m in manifest if m['split']==split and m['benchmark']==benchmark}
    if not ids:raise ValueError('No scenes in this split/benchmark')
    columns=json.loads((DATA/'model_input_columns.json').read_text())
    frames=[]
    for name in ['observations','labels','lineage']:
        f=pd.read_csv(DATA/f'{split}_{name}.csv.gz').set_index(['scene_id','step'])
        frames.append(f.loc[f.index.get_level_values(0).isin(ids)])
    if not frames[0].index.equals(frames[1].index) or not frames[0].index.equals(frames[2].index):raise ValueError('Mismatched keys')
    return frames[0][columns],frames[1],frames[2]
