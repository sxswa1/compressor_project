"""特征提取单元测试：验证公共特征提取库输出维度正确。"""
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ))

from common_algorithms.feature_extraction import (  # noqa: E402
    emd_features,
    envelope_features,
    freq_features,
    time_features,
    wavelet_features,
)


def _sig(n=8, L=1024):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, L)).astype(np.float32)


def test_time_features_shape():
    x = _sig()
    f = time_features(x)
    assert f.shape == (8, 13)
    assert np.isfinite(f).all()


def test_freq_features_shape():
    x = _sig()
    f = freq_features(x, 12000)
    assert f.shape == (8, 7)
    assert np.isfinite(f).all()


def test_envelope_features_shape():
    x = _sig()
    f = envelope_features(x, 12000)
    assert f.shape == (8, 3)
    assert np.isfinite(f).all()


def test_wavelet_features_shape():
    x = _sig()
    f = wavelet_features(x, level=4)
    assert f.shape == (8, 5)
    assert np.isfinite(f).all()


def test_emd_features_shape():
    x = _sig()
    f = emd_features(x, n_imfs=4)
    assert f.shape == (8, 4)
    assert np.isfinite(f).all()


if __name__ == "__main__":
    for fn in [test_time_features_shape, test_freq_features_shape,
               test_envelope_features_shape, test_wavelet_features_shape,
               test_emd_features_shape]:
        fn()
        print(f"{fn.__name__} OK")
