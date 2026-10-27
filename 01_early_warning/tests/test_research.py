"""Metric semantics and chronological inference contracts."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'experiments/rcsd_v1'
for p in [PROJECT,ROOT,ROOT/'final_evaluation_pipeline_v1']:
    sys.path.insert(0,str(p))
from metrics import di_score,rul_score,warning_summary
from pipelines.baseline_data import risk_levels
from pipelines.research_inference import predict


class MetricTests(unittest.TestCase):
    def test_risk_boundaries(self):
        np.testing.assert_array_equal(risk_levels(np.array([1.,.85,.8499,.65,.6499,.4,.3999,0.])),[1,1,2,2,3,3,4,4])

    def test_scene_vs_point_rmse(self):
        f=pd.DataFrame(dict(scene_id=['a','b','b','b'],block_id=['A','B','B','B'],
            RUL_ref_hours=[0.,10.,10.,10.],prediction=[4.,11.,11.,11.]))
        result=rul_score(f)
        self.assertAlmostEqual(result['rmse_hours'],np.sqrt(19/4))
        self.assertEqual(result['scene_macro_rmse_hours'],2.5)
        self.assertEqual(result['endpoint_mae_hours'],4.)
        self.assertAlmostEqual(result['rmse_div_mean_RUL'],np.sqrt(19/4)/7.5)

    def test_missing_and_late_events_separate(self):
        f=pd.DataFrame(dict(source=['v2']*3,stratum=['main']*3,method=['m']*3,
            scene_id=['a','b','c'],block_id=['A','B','C'],endpoint=[24]*3,
            hit=[True,True,False],paired_hit=[True,True,False],two_hour=[True,False,False],
            paired_two_hour=[True,False,False],paired_lead_hours=[3.,1.,np.nan],valid_points=[10]*3,rows=[20]*3))
        row=warning_summary(f).iloc[0]
        self.assertEqual(row.no_paired_event,1)
        self.assertEqual(row.paired_late_under_2h,1)
        self.assertAlmostEqual(row.paired_two_hour_fraction,1/3)


@unittest.skipUnless((ROOT/'final_model_fit_v1/results/model_registry.csv').exists()
                     and (ROOT/'warning_lstm_budget_v1/results/bundle.pkl').exists(),
                     'Run data preparation and both training commands for model-dependent checks')
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        data=ROOT/'RCSD-1YD Data Set/controlled_dataset_v2/data'
        obs=pd.read_csv(data/'train_observations.csv.gz',nrows=220)
        lineage=pd.read_csv(data/'train_lineage.csv.gz',nrows=220)
        cls.frame=obs.drop(columns=['scene_id','step']).assign(Timestamp=lineage.source_timestamp)

    def test_future_values_do_not_change_past_outputs(self):
        before=predict(self.frame,20261024)
        altered=self.frame.copy()
        sensors=[c for c in altered if c!='Timestamp']
        altered.loc[180:,sensors]=altered.loc[180:,sensors]*1.1
        after=predict(altered,20261024)
        pd.testing.assert_frame_equal(before.iloc[:180],after.iloc[:180])

    def test_gap_restarts_history(self):
        f=self.frame.drop(index=120).reset_index(drop=True)
        out=predict(f,20261024)
        self.assertFalse(out.loc[120:214,'di_valid'].any())
        self.assertFalse(out.loc[120:215,'warning_valid'].any())

    def test_missing_values_and_duplicate_times(self):
        f=self.frame.copy();f.loc[120,f.columns[0]]=np.nan
        out=predict(f,20261024)
        self.assertFalse(out.loc[120:215,'di_valid'].any())
        self.assertTrue(out.loc[120:215,'DI'].isna().all())
        f=self.frame.copy();f.loc[1,'Timestamp']=f.loc[0,'Timestamp']
        with self.assertRaises(ValueError):predict(f,20261024)


if __name__=='__main__':unittest.main()
