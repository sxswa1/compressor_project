"""Check CSV prediction adapter against every saved pilot evaluation scene."""
from pathlib import Path
import argparse
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'scripts'))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from predict import predict_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    predictions = pd.read_csv(args.run_dir/'evaluation_predictions.csv.gz')
    obs = pd.read_csv(args.data_dir/'train_observations.csv.gz').set_index(['scene_id','step'])
    lineage = pd.read_csv(args.data_dir/'train_lineage.csv.gz').set_index(['scene_id','step'])
    assert obs.index.equals(lineage.index)
    checks, cases = {}, 0
    with threadpool_limits(limits=2):
        for (model, sid), rows in predictions.loc[predictions.model.ne('training_median')].groupby(['model','scene_id']):
            frame = obs.loc[sid].copy()
            frame['Timestamp'] = lineage.loc[sid].source_timestamp
            output = predict_frame(frame, args.run_dir, model)
            positions = frame.index.get_indexer(rows.step)
            assert (positions >= 0).all()
            assert output.iloc[positions].valid.all()
            np.testing.assert_allclose(output.iloc[positions].DI_pred, rows.DI_pred, rtol=1e-6, atol=1e-6)
            assert len(output) == len(frame)
            assert not output.iloc[:95].valid.any()
            cases += 1
        checks['scene_model_cases'] = cases
        sample_id = predictions.scene_id.iloc[0]
        frame = obs.loc[sample_id].copy().reset_index(drop=True)
        frame['Timestamp'] = pd.to_datetime(lineage.loc[sample_id].source_timestamp.to_numpy())
        for model in ('lstm','gradient_boosting'):
            original = predict_frame(frame, args.run_dir, model)
            changed = frame.copy()
            cutoff = len(frame)//2
            changed.loc[cutoff:, '75TI821.pv'] += 100
            actual = predict_frame(changed, args.run_dir, model)
            pd.testing.assert_frame_equal(actual.iloc[:cutoff], original.iloc[:cutoff])
            for reason in ('missing', 'stop', 'gap', 'segment'):
                changed = frame.copy()
                if reason == 'missing':
                    changed.loc[100, '75TI821.pv'] = np.nan
                elif reason == 'stop':
                    changed.loc[100, '75SI865R.pv'] = 0
                elif reason == 'gap':
                    changed.loc[100:, 'Timestamp'] += pd.Timedelta(minutes=15)
                else:
                    changed['segment_id'] = [0]*100+[1]*(len(frame)-100)
                actual = predict_frame(changed, args.run_dir, model)
                end = min(196 if reason in ('missing','stop') else 195, len(frame))
                assert not actual.iloc[100:end].valid.any(), (model, reason)
                assert actual.iloc[100:end].DI_pred.isna().all()
                checks[model+'_'+reason+'_resets_history'] = True
            checks[model+'_causal_prediction'] = True
        checks['passed'] = True
        checks['final_test_read'] = False
        checks['performance_acceptance'] = False
    (args.run_dir/'adapter_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
