"""包络谱分析特征。

先做 Hilbert 变换取信号包络，再对包络做频谱，在 0~500Hz 轴承故障频带内
提取谱质心、谱峰频率、谱峰幅值共 3 维。包络谱对周期性冲击（轴承故障）敏感。
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


def envelope_features(x: np.ndarray, sample_rate: int = 12000) -> np.ndarray:
    """逐样本包络谱特征。

    Args:
        x: 输入信号 (n_samples, window_len)。
        sample_rate: 采样频率 (Hz)。

    Returns:
        特征矩阵 (n_samples, 3)。
    """
    x = np.atleast_2d(x).astype(np.float64)
    analytic = signal.hilbert(x, axis=1)
    env = np.abs(analytic)
    env = env - env.mean(axis=1, keepdims=True)
    spec = np.abs(rfft(env, axis=1))
    freqs = rfftfreq(x.shape[1], d=1.0 / sample_rate)

    # 只关注 0 ~ 500 Hz 的典型轴承故障频带
    mask = freqs <= 500
    spec_b = spec[:, mask]
    freqs_b = freqs[mask]
    total = spec_b.sum(axis=1, keepdims=True) + 1e-8
    centroid = (spec_b * freqs_b).sum(axis=1, keepdims=True) / total
    peak_idx = spec_b.argmax(axis=1)
    peak_freq = freqs_b[peak_idx].reshape(-1, 1)
    peak_amp = spec_b.max(axis=1, keepdims=True)
    return np.nan_to_num(np.concatenate([centroid, peak_freq, peak_amp], axis=1)).astype(np.float32)
