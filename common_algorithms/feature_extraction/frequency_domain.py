"""频域与 FFT 分析特征。

对逐样本信号做实数 FFT，提取频域统计指标，共 7 维：
谱质心、谱扩散、低/中/高频带能量占比（3 维）、谱峰频率、谱峰幅值。
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq


def freq_features(x: np.ndarray, sample_rate: int = 12000) -> np.ndarray:
    """逐样本频域特征（基于幅值谱）。

    Args:
        x: 输入信号 (n_samples, window_len)。
        sample_rate: 采样频率 (Hz)。

    Returns:
        特征矩阵 (n_samples, 7)。
    """
    x = np.atleast_2d(x).astype(np.float64)
    n = x.shape[1]
    spec = np.abs(rfft(x, axis=1))[:, 1:]          # 去掉直流分量
    freqs = rfftfreq(n, d=1.0 / sample_rate)[1:]

    total = spec.sum(axis=1, keepdims=True) + 1e-8
    centroid = (spec * freqs).sum(axis=1, keepdims=True) / total
    spread = np.sqrt(
        ((spec * (freqs - centroid) ** 2).sum(axis=1, keepdims=True)) / total)

    # 频带能量占比（低频 / 中频 / 高频）
    fmax = sample_rate / 2
    low = (freqs <= fmax / 3).astype(float)
    mid = ((freqs > fmax / 3) & (freqs <= 2 * fmax / 3)).astype(float)
    high = (freqs > 2 * fmax / 3).astype(float)
    band_ratio = np.stack([
        (spec * low).sum(axis=1),
        (spec * mid).sum(axis=1),
        (spec * high).sum(axis=1),
    ], axis=1) / total

    # 谱峰频率及其幅值
    peak_idx = spec.argmax(axis=1)
    peak_freq = freqs[peak_idx].reshape(-1, 1)
    peak_amp = spec.max(axis=1, keepdims=True)

    feats = np.concatenate([centroid, spread, band_ratio, peak_freq, peak_amp], axis=1)
    return np.nan_to_num(feats).astype(np.float32)
