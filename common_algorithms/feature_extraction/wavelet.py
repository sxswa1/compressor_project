"""小波变换特征。

使用 db4 小波对信号做多级分解，提取各子带能量占比（含逼近系数），
默认 4 层分解输出 5 维。小波变换能同时刻画频带与局部时域细节。
"""
from __future__ import annotations

import numpy as np


def wavelet_features(x: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """逐样本小波子带能量占比特征。

    Args:
        x: 输入信号 (n_samples, window_len)。
        wavelet: 小波基名称，默认 db4。
        level: 分解层数。

    Returns:
        特征矩阵 (n_samples, level + 1)。缺少 PyWavelets 时返回全零矩阵。
    """
    try:
        import pywt
    except ImportError:
        return np.zeros((len(np.atleast_2d(x)), level + 1), dtype=np.float32)

    x = np.atleast_2d(x).astype(np.float64)
    out = []
    for row in x:
        coeffs = pywt.wavedec(row, wavelet, level=level)
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        total = sum(energies) + 1e-8
        out.append([e / total for e in energies])
    return np.asarray(out, dtype=np.float32)
