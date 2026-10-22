"""时域统计分析特征。

从振动波形直接计算的统计指标，共 13 维：均值、标准差、RMS、峰值、峰峰值、
绝对均值、能量、峰值因子、波形因子、脉冲因子、裕度因子、峭度、偏度。
所有函数逐样本计算，输出形状 (n_samples, 13)。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def time_features(x: np.ndarray) -> np.ndarray:
    """逐样本时域统计特征。

    Args:
        x: 输入信号 (n_samples, window_len) 或单条样本 (window_len,)。

    Returns:
        特征矩阵 (n_samples, 13)，float。
    """
    x = np.atleast_2d(x).astype(np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    rms = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
    peak = np.abs(x).max(axis=1, keepdims=True)
    pp = x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)
    abs_mean = np.abs(x).mean(axis=1, keepdims=True)
    energy = (x ** 2).sum(axis=1, keepdims=True)

    crest = peak / (rms + 1e-8)           # 峰值因子
    shape = rms / (abs_mean + 1e-8)       # 波形因子
    impulse = peak / (abs_mean + 1e-8)    # 脉冲因子
    sqrt_mean = np.sqrt(np.abs(x)).mean(axis=1, keepdims=True)
    margin = peak / (sqrt_mean ** 2 + 1e-8)  # 裕度因子
    kur = kurtosis(x, axis=1, fisher=True).reshape(-1, 1)   # 峭度
    skew_ = skew(x, axis=1).reshape(-1, 1)                  # 偏度

    feats = np.concatenate([
        mean, std, rms, peak, pp, abs_mean, energy,
        crest, shape, impulse, margin, kur, skew_,
    ], axis=1)
    return np.nan_to_num(feats).astype(np.float32)
