"""Behavior checks for reusable models and causal RCSD window construction."""
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from common_algorithms.models.lstm import LSTMRegressor
from common_algorithms.models.gradient_boosting import GradientBoostingRegressor
from pipelines.baseline_data import windows, tabular_features, risk_levels


class PublicModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.pool = threadpool_limits(limits=1)
        rng = np.random.default_rng(81)
        cls.X = rng.normal(size=(96, 8, 3)).astype(np.float32)
        cls.y = cls.X[:, -1, 0] * .4 + cls.X[:, :, 1].mean(axis=1)
        cls.lstm = LSTMRegressor(hidden_size=8, epochs=12, batch_size=32, learning_rate=.01).fit(cls.X, cls.y)
        cls.gb = GradientBoostingRegressor(max_iter=40, min_samples_leaf=5).fit(cls.X[:, -1], cls.y)

    @classmethod
    def tearDownClass(cls):
        cls.pool.restore_original_limits()

    def test_lstm_training_and_normalization(self):
        self.assertLess(self.lstm.loss_history_[-1], self.lstm.loss_history_[0])
        mean = self.lstm.mean_.copy()
        scale = self.lstm.scale_.copy()
        self.lstm.predict(self.X + 100)
        np.testing.assert_array_equal(self.lstm.mean_, mean)
        np.testing.assert_array_equal(self.lstm.scale_, scale)
        np.testing.assert_allclose(mean, self.X.mean(axis=(0,1), dtype=np.float64), rtol=1e-6)

    def test_regressors_learn_beyond_constant(self):
        baseline = np.mean((self.y-self.y.mean())**2)
        self.assertLess(np.mean((self.lstm.predict(self.X)-self.y)**2), baseline)
        self.assertLess(np.mean((self.gb.predict(self.X[:, -1])-self.y)**2), baseline)

    def test_round_trip_and_batch_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, model, X in [('lstm', self.lstm, self.X), ('gb', self.gb, self.X[:, -1])]:
                artifact = Path(directory)/name
                model.save(artifact)
                restored = type(model).load(artifact)
                expected = model.predict(X)
                np.testing.assert_allclose(restored.predict(X), expected, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(model.predict(X[:3]), expected[:3], rtol=1e-5, atol=1e-6)
                # Unrelated future samples cannot alter an earlier sample's prediction.
                changed = X.copy()
                changed[3:] += 100
                np.testing.assert_allclose(model.predict(changed)[:3], expected[:3], rtol=1e-6, atol=1e-7)

    def test_reproducible_training_without_rng_pollution(self):
        state = torch.random.get_rng_state().clone()
        a = LSTMRegressor(hidden_size=4, epochs=2).fit(self.X, self.y)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        b = LSTMRegressor(hidden_size=4, epochs=2).fit(self.X, self.y)
        np.testing.assert_array_equal(a.predict(self.X), b.predict(self.X))

    def test_invalid_input_and_unfitted_models(self):
        for model, X in [(self.lstm, self.X), (self.gb, self.X[:, -1])]:
            invalid = X.copy()
            invalid.flat[0] = np.nan
            with self.assertRaises(ValueError):
                model.predict(invalid)
            with self.assertRaises(ValueError):
                model.predict(X[..., :2])
            with self.assertRaises(ValueError):
                model.fit(X, self.y[:-1])
            with self.assertRaises(RuntimeError):
                type(model)().predict(X)


class CausalWindowTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(80, dtype=np.float32).reshape(40, 2)
        self.t = pd.date_range('2022-01-01', periods=40, freq='15min')

    def test_future_change_does_not_change_past(self):
        a, ai = windows(self.x, self.t, window=4)
        changed = self.x.copy()
        changed[20:] *= -10
        b, bi = windows(changed, self.t, window=4)
        np.testing.assert_array_equal(ai, bi)
        np.testing.assert_array_equal(a[ai < 20], b[bi < 20])

    def test_gap_missing_and_segment_reset(self):
        x = self.x.copy()
        x[10] = np.nan
        _, ends = windows(x, self.t, window=4)
        self.assertFalse(any(i in ends for i in range(10,14)))
        self.assertIn(14, ends)
        shifted = self.t.to_series(index=range(40))
        shifted.loc[20:] += pd.Timedelta(minutes=15)
        _, ends = windows(self.x, shifted, window=4)
        self.assertNotIn(22, ends)
        self.assertIn(23, ends)
        _, ends = windows(self.x, self.t, window=4, segments=[0]*20+[1]*20)
        self.assertNotIn(22, ends)
        self.assertIn(23, ends)

    def test_invalid_times_and_short_input(self):
        for times in [self.t[::-1], pd.DatetimeIndex([self.t[0]]*40)]:
            with self.assertRaises(ValueError):
                windows(self.x, times, window=4)
        x, ends = windows(self.x[:2], self.t[:2], window=4)
        self.assertEqual(x.shape, (0,4,2))
        self.assertEqual(len(ends), 0)

    def test_feature_recipe_and_risk_boundaries(self):
        X = np.arange(96*25, dtype=np.float32).reshape(1,96,25)
        self.assertEqual(tabular_features(X).shape, (1,125))
        np.testing.assert_array_equal(risk_levels([0,.3999,.4,.6499,.65,.8499,.85,1]), [4,4,3,3,2,2,1,1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
