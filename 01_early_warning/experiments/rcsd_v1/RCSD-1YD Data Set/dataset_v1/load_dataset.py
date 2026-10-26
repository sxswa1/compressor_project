"""Minimal loader. Only sensor columns are returned as model features."""
from pathlib import Path
import json
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'


def load_scenes(split='train', data_dir=DATA):
    """Return (sensor features, labels, lineage), indexed by (scene_id, step).

    Group by scene_id before rolling/windowing. Never randomly split rows.
    IDs, time stamps and generator parameters are NOT returned as features.
    Missing observations are intentional; fit any imputer on training only.
    """
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation or test')
    data_dir = Path(data_dir)
    columns = json.loads((data_dir / 'model_input_columns.json').read_text(encoding='utf-8'))
    frames = [pd.read_csv(data_dir / f'{split}_{name}.csv.gz').set_index(['scene_id', 'step'])
              for name in ('observations', 'labels', 'lineage')]
    if not frames[0].index.equals(frames[1].index) or not frames[0].index.equals(frames[2].index):
        raise ValueError('Observation/label/lineage keys differ')
    return frames[0][columns], frames[1], frames[2]


def load_baseline(split='train', data_dir=DATA):
    """Return untouched baseline candidates, plus segment metadata.

    Segment IDs must be respected when forming causal windows/forecast targets.
    Candidate status is a screening proxy, not a certified healthy label.
    """
    if split not in ('train', 'validation', 'test'): raise ValueError('Invalid split')
    data_dir = Path(data_dir)
    raw = pd.read_csv(data_dir / 'raw_observations.csv.gz').set_index('source_row_id')
    meta = pd.read_csv(data_dir / 'raw_metadata.csv').set_index('source_row_id')
    selected = meta['split'].eq(split) & meta.baseline_candidate
    return raw.loc[selected], meta.loc[selected]
