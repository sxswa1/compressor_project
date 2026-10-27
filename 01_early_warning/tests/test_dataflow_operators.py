"""单文件加载、输入边界与已有模型预测的一致性。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
MODEL_DIR = Path(os.environ.get('RCSD_DATAFLOW_MODELS', PROJECT/'outputs/dataflow_models'))
OPERATORS = REPO/'dataflow_operators/health_assessment'


def load_source(name):
    # 平台按源码加载；不给 __file__，也不加入研究模块路径。
    namespace = {'__name__': f'dataflow_{name}'}
    exec(compile((OPERATORS/f'{name}.py').read_text(encoding='utf-8'), name, 'exec'), namespace)
    return namespace


class ContractTests(unittest.TestCase):
    def test_single_file_contract(self):
        import inspect
        for name in ['compressor_early_warning', 'compressor_risk_quantification']:
            ns = load_source(name)
            sig = inspect.signature(ns['execute'])
            self.assertEqual(next(iter(sig.parameters)), 'df')
            self.assertIn('kwargs', sig.parameters)

    def test_threshold_boundaries(self):
        ns = load_source('compressor_risk_quantification')
        values = np.array([1, .85, np.nextafter(.85, 0), .65, np.nextafter(.65, 0), .4, np.nextafter(.4, 0), 0])
        np.testing.assert_array_equal(ns['_risk_levels'](values, [.85,.65,.4]), [1,1,2,2,3,3,4,4])


@unittest.skipUnless((MODEL_DIR/'manifest.json').exists(), '先执行 export_dataflow_models.py')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.warning = staticmethod(load_source('compressor_early_warning')['execute'])
        cls.risk = staticmethod(load_source('compressor_risk_quantification')['execute'])
        sys.path.insert(0, str(PROJECT))
        from pipelines.research_inference import predict
        cls.original = staticmethod(predict)
        root = PROJECT/'experiments/rcsd_v1'
        cls.root = root
        data = root/'RCSD-1YD Data Set/dataset_v1/data'
        obs = pd.read_csv(data/'raw_observations.csv.gz', nrows=520)
        timestamps = pd.read_csv(data/'raw_metadata.csv', nrows=520).source_timestamp
        sensors = json.loads((MODEL_DIR/'preprocessing.json').read_text('utf-8'))['sensors']
        cls.frame = obs[sensors].copy().assign(Timestamp=timestamps)

    def run_ops(self, frame, seed=20261024):
        warning = self.warning(frame, str(MODEL_DIR/'warning.pkl'))
        risk = self.risk(frame, str(MODEL_DIR), seed)
        for result in (warning, risk):
            self.assertEqual(set(result), {'out_1'})
            self.assertIsInstance(result['out_1'], pd.DataFrame)
            self.assertEqual(len(result['out_1']), len(frame))
        return warning['out_1'], risk['out_1']

    def compare_original(self, frame, seed=20261024):
        original = self.original(frame, seed)
        w, r = self.run_ops(frame, seed)
        for actual in (w.drop(columns='warning_score'), r):
            pd.testing.assert_frame_equal(actual, original[list(actual)], check_exact=False, rtol=1e-6, atol=1e-6)
        return w, r

    def test_exported_hashes(self):
        manifest = json.loads((MODEL_DIR/'manifest.json').read_text('utf-8'))
        for name, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((MODEL_DIR/name).read_bytes()).hexdigest(), expected)

    def test_isolated_execution_without_repository_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.frame.to_csv(folder/'input.csv', index=False)
            for name in ['compressor_early_warning', 'compressor_risk_quantification']:
                (folder/f'{name}.py').write_bytes((OPERATORS/f'{name}.py').read_bytes())
            code = '''
import pathlib, sys, pandas as pd, torch
torch.set_num_threads(2)
folder, models = map(pathlib.Path, sys.argv[1:])
df = pd.read_csv(folder/'input.csv')
for name in ['compressor_early_warning', 'compressor_risk_quantification']:
    scope = {'__name__': 'platform_operator'}
    exec(compile((folder/(name+'.py')).read_text('utf-8'), name, 'exec'), scope)
    kwargs = {'model_path': str(models/'warning.pkl')} if name.endswith('warning') else {'model_dir': str(models), 'seed': 20261024}
    output = scope['execute'](df, **kwargs)
    assert set(output) == {'out_1'} and len(output['out_1']) == len(df)
assert not any(k.startswith('pipelines') or k.startswith('common_algorithms') for k in sys.modules)
print('isolated execution PASS')
'''
            result = subprocess.run([sys.executable, '-I', '-c', code, str(folder), str(MODEL_DIR.resolve())],
                                    cwd=folder, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('isolated execution PASS', result.stdout)

    def test_original_predictions_all_seeds(self):
        for seed in [20261023,20261024,20261025]:
            self.compare_original(self.frame, seed)

    def test_alarm_and_recovery(self):
        frame = self.frame.copy()
        frame.loc[140:290, '75TI829.pv'] += 100
        w, _ = self.compare_original(frame)
        self.assertTrue(w.warning_new.any())
        self.assertTrue(w.warning_active.any())

    def test_missing_stop_gap_and_segments(self):
        frame = self.frame.copy()
        frame.loc[120, '75TI829.pv'] = np.nan
        frame.loc[240, '75SI865R.pv'] = 0
        frame['segment_id'] = np.where(np.arange(len(frame)) < 360, 1, 2)
        frame = frame.drop(index=460).reset_index(drop=True)
        w, r = self.compare_original(frame)
        self.assertTrue(r.loc[120:215, 'DI'].isna().all())
        self.assertFalse(r.loc[240:335, 'rul_valid'].any())
        self.assertFalse(w.loc[360:455, 'warning_valid'].any())
        self.assertFalse(r.loc[460:, 'di_valid'].any())

    def test_future_invariance_and_input_unchanged(self):
        frame = self.frame.copy(deep=True)
        w, r = self.run_ops(frame)
        pd.testing.assert_frame_equal(frame, self.frame)
        frame.loc[350:, '75TI829.pv'] += 50
        ww, rr = self.run_ops(frame)
        pd.testing.assert_frame_equal(w.iloc[:350], ww.iloc[:350])
        pd.testing.assert_frame_equal(r.iloc[:350], rr.iloc[:350])

    def test_aliases_and_no_label_dependency(self):
        frame = self.frame.copy().rename(columns={'Timestamp': 'time', '75TI829.pv': 'temperature'})
        frame['DI_ref'] = -100
        frame['RUL_ref_hours'] = -100
        mapping = json.dumps({'75TI829.pv': 'temperature'})
        w, r = self.run_ops(self.frame)
        ww = self.warning(frame, str(MODEL_DIR/'warning.pkl'), timestamp_column='time', sensor_map_json=mapping)['out_1']
        rr = self.risk(frame, str(MODEL_DIR), 20261024, timestamp_column='time', sensor_map_json=mapping)['out_1']
        pd.testing.assert_frame_equal(w, ww)
        pd.testing.assert_frame_equal(r, rr)

    def test_warmup_and_invalid_input(self):
        w, r = self.run_ops(self.frame.iloc[:10])
        self.assertFalse(w.warning_valid.any())
        self.assertTrue(r.DI.isna().all())
        self.assertTrue(r.RUL_hours.isna().all())
        bad = [self.frame.iloc[:0], self.frame.drop(columns='75TI829.pv'), self.frame.iloc[::-1]]
        duplicate = self.frame.copy()
        duplicate.loc[1, 'Timestamp'] = duplicate.loc[0, 'Timestamp']
        bad.append(duplicate)
        for frame in bad:
            with self.assertRaises((ValueError, TypeError)):
                self.warning(frame, str(MODEL_DIR/'warning.pkl'))
            with self.assertRaises((ValueError, TypeError)):
                self.risk(frame, str(MODEL_DIR), 20261024)
        with self.assertRaises(ValueError):
            self.risk(self.frame, str(MODEL_DIR), True)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                self.risk(self.frame, empty, 20261024)


if __name__ == '__main__':
    unittest.main()
