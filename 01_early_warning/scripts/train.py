"""Train the first public-model pilot; never reads validation or final-test files."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import sklearn
import torch
from threadpoolctl import threadpool_limits
from common_algorithms.models.lstm import LSTMRegressor
from common_algorithms.models.gradient_boosting import GradientBoostingRegressor
from pipelines.baseline_data import load_training_pilot, tabular_features, metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='existing controlled_dataset_v2/data')
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/public_baselines_v1.json')
    parser.add_argument('--output', type=Path, default=PROJECT/'outputs/public_baselines_v1')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty; use a new directory to preserve earlier runs')
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['history_points'] != 96:
        raise ValueError('the fixed tabular recipe requires 96 history points')
    write_json(args.output/'config.json', config)
    torch.set_num_threads(config['torch_threads'])
    batches, sensors, split = load_training_pilot(args.data_dir, config)
    X, fit_keys = batches['train']
    Z, evaluation_keys = batches['evaluation']
    fit_keys.to_csv(args.output/'fit_sample_keys.csv.gz', index=False)
    evaluation_keys.to_csv(args.output/'evaluation_sample_keys.csv.gz', index=False)
    write_json(args.output/'split.json', split)
    write_json(args.output/'input_schema.json', dict(sensors=sensors, history_points=96,
                sampling_minutes=config['sampling_minutes'], task='DI regression', output_clip=[0,1]))
    print(f'Training windows: {len(X)}; evaluation windows: {len(Z)}', flush=True)
    models = dict(lstm=LSTMRegressor(**config['lstm'], random_state=config['seed']),
                  gradient_boosting=GradientBoostingRegressor(**config['gradient_boosting'], random_state=config['seed']))
    summary, predictions, checks = {}, [], {}
    for name in ('training_median', 'lstm', 'gradient_boosting'):
        started = time.monotonic()
        if name == 'training_median':
            raw = np.full(len(Z), np.median(fit_keys.DI_ref))
        else:
            model = models[name]
            a, b = (X, Z) if name == 'lstm' else (tabular_features(X), tabular_features(Z))
            with threadpool_limits(limits=2):
                model.fit(a, fit_keys.DI_ref.to_numpy())
                raw = model.predict(b)
                artifact = args.output/('lstm.pt' if name == 'lstm' else 'gradient_boosting.pkl')
                model.save(artifact)
                restored = type(model).load(artifact)
                np.testing.assert_allclose(restored.predict(b), raw, rtol=1e-6, atol=1e-7)
            checks[name+'_save_load_equal'] = True
            if name == 'lstm':
                write_json(args.output/'lstm_training_loss.json', model.loss_history_)
        frame = evaluation_keys.copy()
        frame['model'] = name
        frame['DI_pred_raw'] = raw
        frame['DI_pred'] = np.clip(raw, 0, 1)
        summary[name] = dict(**metrics(frame), elapsed_seconds=time.monotonic()-started,
                              prediction_clipped_fraction=float(np.mean((raw < 0) | (raw > 1))))
        predictions.append(frame)
        print(name, summary[name], flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(args.output/'evaluation_predictions.csv.gz', index=False)
    write_json(args.output/'metrics.json', summary)
    write_json(args.output/'checks.json', checks)
    sources = {}
    for name in ('model_input_columns.json', 'scene_manifest.json', 'train_observations.csv.gz', 'train_labels.csv.gz', 'train_lineage.csv.gz'):
        sources[name] = hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
    code = {}
    for path in [Path(__file__), PROJECT/'pipelines/baseline_data.py',
                 REPO/'common_algorithms/models/lstm.py', REPO/'common_algorithms/models/gradient_boosting.py']:
        code[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.output/'provenance.json', dict(data=sources, code=code,
                environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                 sklearn=sklearn.__version__, torch=torch.__version__),
                evaluation_scope='one training-period chronological pilot; not independent final evaluation',
                model_selection=False, existing_delivery_changed=False, final_test_read=False))


if __name__ == '__main__':
    main()
