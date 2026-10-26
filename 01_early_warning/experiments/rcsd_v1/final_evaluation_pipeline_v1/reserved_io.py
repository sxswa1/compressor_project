"""Read materialized scene files. Used only after the reserved release gate."""
import pandas as pd
from common import WORK,PROTOCOL,read,digest,assert_release

def scenes_from_directory(directory,prefix):
    manifests=read(directory/f'{prefix}_manifest.json')
    if not manifests:return
    names=['observations','labels','lineage','injection_mask']
    tables={name:pd.read_csv(directory/f'{prefix}_{name}.csv.gz').set_index(['scene_id','step']) for name in names}
    assert all(tables[name].index.equals(tables['observations'].index) for name in names)
    assert not tables['observations'].index.has_duplicates
    assert list(tables['observations'].index.get_level_values('scene_id').unique())==[m['scene_id'] for m in manifests]
    for m in manifests:
        frames=[tables[name].loc[m['scene_id']].reset_index(drop=True) for name in names]
        assert all(len(f)==m.get('n',m.get('rows')) for f in frames)
        yield m,*frames

def reserved_scenes(prefix):
    assert_release()
    directory=WORK/read(PROTOCOL)['dataset']['output_directory']
    status=read(directory/'build_status.json')
    assert status['status']=='GENERATED_NOT_SCORED' and status['split']=='test'
    for path,h in read(directory/'checksums.json').items():assert digest(directory/path)==h,path
    yield from scenes_from_directory(directory,prefix)
