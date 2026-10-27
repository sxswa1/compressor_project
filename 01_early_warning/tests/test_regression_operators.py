"""公共回归算子的输出、资源配对、时序边界和登记一致性。"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PROJECT))
from common_algorithms.models.lstm import LSTMRegressor
from common_algorithms.models.gradient_boosting import GradientBoostingRegressor
from pipelines.baseline_data import windows, tabular_features


def source(name):
    return (REPO/f'dataflow_operators/predictive_analysis/{name}_regression.py').read_text('utf-8')


def load(name):
    scope = {'__name__': 'loaded_operator'}
    exec(compile(source(name), name, 'exec'), scope)
    return scope['execute']


class RegressionOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        rng = np.random.default_rng(37)
        cls.x = rng.normal(size=(35,2)).astype(np.float32)
        cls.frame = pd.DataFrame(cls.x,columns=['a','b'])
        cls.frame['Timestamp'] = pd.date_range('2022-01-01',periods=35,freq='15min')
        cls.win = np.stack([cls.x[i-3:i+1] for i in range(3,35)])
        cls.lstm = LSTMRegressor(hidden_size=4,epochs=2,batch_size=8).fit(cls.win, cls.win[:,-1,0])
        cls.gb = GradientBoostingRegressor(max_iter=10,min_samples_leaf=2).fit(cls.x, cls.x[:,0]**2-cls.x[:,1])
        cls.lstm.save(cls.folder/'lstm.pt')
        cls.gb.save(cls.folder/'gradient_boosting.pkl')
        for name, suffix in [('lstm','pt'),('gradient_boosting','pkl')]:
            schema = dict(model_sha256=hashlib.sha256((cls.folder/f'{name}.{suffix}').read_bytes()).hexdigest(),
                          feature_columns=['a','b'],feature_units={'a':'1','b':'1'},target_name='y',target_unit='1',
                          window_points=4,sampling_minutes=15)
            (cls.folder/f'{name}.json').write_text(json.dumps(schema),'utf-8')
        cls.execute_lstm = staticmethod(load('lstm'))
        cls.execute_gb = staticmethod(load('gradient_boosting'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call(self,name,frame,**kw):
        suffix = 'pt' if name=='lstm' else 'pkl'
        fn = self.execute_lstm if name=='lstm' else self.execute_gb
        result=fn(frame,str(self.folder/f'{name}.{suffix}'),str(self.folder/f'{name}.json'),**kw)
        self.assertEqual(set(result),{'out_1'})
        self.assertIsInstance(result['out_1'],pd.DataFrame)
        self.assertEqual(len(result['out_1']),len(frame))
        return result['out_1']

    def test_native_predictions_and_unchanged_input(self):
        before=self.frame.copy(deep=True)
        a=self.call('lstm',self.frame)
        b=self.call('gradient_boosting',self.frame)
        self.assertTrue(a.loc[:2,'prediction'].isna().all())
        np.testing.assert_allclose(a.loc[3:,'prediction'],self.lstm.predict(self.win),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(b.prediction,self.gb.predict(self.x),rtol=0,atol=0)
        pd.testing.assert_frame_equal(self.frame,before)

    def test_feature_order_aliases_and_labels(self):
        for name in ['lstm','gradient_boosting']:
            expected=self.call(name,self.frame).prediction
            f=self.frame[['b','Timestamp','a']].rename(columns={'a':'renamed'}).copy()
            f['label']=-999
            actual=self.call(name,f,feature_map_json=json.dumps({'a':'renamed'})).prediction
            np.testing.assert_array_equal(expected,actual)

    def test_gap_segment_missing_and_future(self):
        f=self.frame.copy()
        f.loc[8,'a']=np.nan
        f['segment_id']=np.where(np.arange(len(f))<20,1,2)
        f=f.drop(index=27).reset_index(drop=True)
        result=self.call('lstm',f)
        for a,b in [(8,11),(20,22),(27,29)]:
            self.assertFalse(result.loc[a:b,'valid'].any())
        earlier=self.call('lstm',self.frame)
        altered=self.frame.copy();altered.loc[25:,'a']+=100
        later=self.call('lstm',altered)
        pd.testing.assert_frame_equal(earlier.iloc[:25],later.iloc[:25])
        gb=self.call('gradient_boosting',f)
        self.assertFalse(gb.loc[8,'valid'])
        self.assertTrue(np.isnan(gb.loc[8,'prediction']))

    def test_empty_invalid_input_and_hash(self):
        for name in ['lstm','gradient_boosting']:
            self.assertTrue(self.call(name,self.frame.iloc[:0]).empty)
            with self.assertRaises(ValueError):self.call(name,self.frame.drop(columns='a'))
            with self.assertRaises(ValueError):self.call(name,self.frame,feature_map_json='{"a":"b"}')
        with self.assertRaises(ValueError):self.call('lstm',self.frame.iloc[::-1])
        dup=self.frame.copy();dup.loc[1,'Timestamp']=dup.loc[0,'Timestamp']
        with self.assertRaises(ValueError):self.call('lstm',dup)
        numeric=self.frame.copy();numeric['Timestamp']=np.arange(len(numeric))
        with self.assertRaises(ValueError):self.call('lstm',numeric)
        bad=self.folder/'bad.json';bad.write_text((self.folder/'lstm.json').read_text().replace('"model_sha256": "','"model_sha256": "bad'))
        with self.assertRaises(ValueError):self.execute_lstm(self.frame,str(self.folder/'lstm.pt'),str(bad))
        with self.assertRaises(ValueError):self.call('gradient_boosting',self.frame.assign(prediction=0))

    def test_standalone_without_repository_imports(self):
        for name in ['lstm','gradient_boosting']:
            (self.folder/f'{name}.py').write_text(source(name),'utf-8')
        self.frame.to_csv(self.folder/'input.csv',index=False)
        code='''
from pathlib import Path
import sys, pandas as pd, torch
torch.set_num_threads(2)
p=Path(sys.argv[1]);df=pd.read_csv(p/'input.csv')
for name,ext in [('lstm','pt'),('gradient_boosting','pkl')]:
    ns={'__name__':'runtime_operator'}
    exec(compile((p/(name+'.py')).read_text('utf-8'),name,'exec'),ns)
    result=ns['execute'](df,str(p/(name+'.'+ext)),str(p/(name+'.json')))
    assert set(result)=={'out_1'} and result['out_1'].valid.any()
assert not any(k.startswith('common_algorithms') or k.startswith('pipelines') for k in sys.modules)
'''
        r=subprocess.run([sys.executable,'-I','-c',code,str(self.folder)],cwd=self.folder,capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_generated_sources_and_registry(self):
        scope={'__file__':str(REPO/'tools/build_regression_operators.py'),'__name__':'builder'}
        exec((REPO/'tools/build_regression_operators.py').read_text('utf-8'),scope)
        for name in ['lstm','gradient_boosting']:
            self.assertEqual(source(name),scope['build'](name))
        registry={};exec((REPO/'dataflow_operators/registry.py').read_text('utf-8'),registry)
        self.assertEqual(len(registry['LINGCHEN_OPERATORS']),10)
        self.assertEqual(len(registry['GAOHY_OPERATORS']),4)
        self.assertEqual(set(registry['OPERATORS']),set(registry['LINGCHEN_OPERATORS'])|set(registry['GAOHY_OPERATORS']))
        self.assertFalse(set(registry['LINGCHEN_OPERATORS'])&set(registry['GAOHY_OPERATORS']))
        for name,entry in registry['GAOHY_OPERATORS'].items():
            self.assertEqual(entry['output_ports'],{'out_1':'pd.DataFrame'})
            self.assertTrue((REPO/'dataflow_operators'/entry['path']).is_file())


MODELS=PROJECT/'outputs/dataflow_models_registered'
@unittest.skipUnless((MODELS/'gradient_boosting_di.schema.json').exists(),'先导出 dataflow_models_registered')
class ResearchResourceTests(unittest.TestCase):
    def test_existing_di_models(self):
        torch.set_num_threads(2)
        data=PROJECT/'experiments/rcsd_v1/RCSD-1YD Data Set/dataset_v1/data'
        meta=json.loads((MODELS/'preprocessing.json').read_text('utf-8'))
        frame=pd.read_csv(data/'raw_observations.csv.gz',nrows=520)[meta['sensors']]
        frame['Timestamp']=pd.read_csv(data/'raw_metadata.csv',nrows=520).source_timestamp
        a=frame[meta['sensors']].to_numpy(dtype=np.float32)
        a[~frame['75SI865R.pv'].between(6050,8500).to_numpy()]=np.nan
        x,ends=windows(a,frame.Timestamp,96,15)
        self.assertGreater(len(ends),0)
        for seed in [20261023,20261024,20261025]:
            path=MODELS/f'di_{seed}.pt'
            output=load('lstm')(frame,str(path),str(MODELS/f'di_{seed}.schema.json'))['out_1']
            np.testing.assert_array_equal(np.flatnonzero(output.valid),ends)
            np.testing.assert_allclose(output.loc[ends,'prediction'],LSTMRegressor.load(path).predict(x),rtol=1e-6,atol=1e-6)
        schema=json.loads((MODELS/'gradient_boosting_di.schema.json').read_text('utf-8'))
        features=pd.DataFrame(tabular_features(x),columns=schema['feature_columns'])
        output=load('gradient_boosting')(features,str(MODELS/'gradient_boosting_di.pkl'),str(MODELS/'gradient_boosting_di.schema.json'))['out_1']
        expected=GradientBoostingRegressor.load(MODELS/'gradient_boosting_di.pkl').predict(tabular_features(x))
        np.testing.assert_allclose(output.prediction,expected,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
