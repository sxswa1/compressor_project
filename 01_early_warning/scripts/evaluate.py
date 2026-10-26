"""Recompute metrics from a saved pilot; no model selection or data access."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.baseline_data import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    predictions = pd.read_csv(args.run_dir/'evaluation_predictions.csv.gz')
    expected = json.loads((args.run_dir/'metrics.json').read_text(encoding='utf-8'))
    reference_keys = None
    for name, frame in predictions.groupby('model', sort=False):
        if frame.duplicated(['scene_id','step']).any():
            raise ValueError('duplicate evaluation keys')
        keys = frame[['scene_id','step','DI_ref']].reset_index(drop=True)
        if reference_keys is not None:
            pd.testing.assert_frame_equal(keys, reference_keys)
        reference_keys = keys
        result = metrics(frame)
        for key in ('points', 'scenes', 'mae', 'rmse', 'scene_macro_mae', 'risk_accuracy'):
            np.testing.assert_allclose(result[key], expected[name][key], rtol=1e-6, atol=1e-8)
        for level, recall in result['risk_recall'].items():
            if recall is None:
                assert expected[name]['risk_recall'][level] is None
            else:
                np.testing.assert_allclose(recall, expected[name]['risk_recall'][level])
        print(name, json.dumps(result, ensure_ascii=False))
    assert set(expected) == set(predictions.model.unique())
    print('All model evaluation keys and saved metrics agree.')


if __name__ == '__main__':
    main()
