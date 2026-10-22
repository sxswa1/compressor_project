"""包络谱分析算子（自包含）。

对切片信号做 Hilbert 包络谱，在 0~500Hz 故障频带内提取谱质心/谱峰频率/谱峰幅值（3 维）。

输入: df —— 振动信号表
输出: out_1 —— 特征表
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq


def _extract_signal(df: pd.DataFrame, signal_col: str) -> np.ndarray:
    if signal_col in df.columns:
        col = signal_col
    else:
        col = next(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    vals = df[col].to_numpy()
    if len(vals) == 1 and hasattr(vals[0], "__len__"):
        return np.asarray(vals[0], dtype=np.float64).ravel()
    return np.asarray(vals, dtype=np.float64).ravel()


def _sliding_windows(signal: np.ndarray, window_len: int, overlap: float) -> np.ndarray:
    step = max(int(window_len * (1.0 - overlap)), 1)
    n = (len(signal) - window_len) // step + 1
    if n <= 0:
        return np.empty((0, window_len))
    idx = np.arange(window_len)[None, :] + step * np.arange(n)[:, None]
    return signal[idx]


def _envelope_features(x: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float64)
    analytic = signal.hilbert(x, axis=1)
    env = np.abs(analytic) - np.abs(analytic).mean(axis=1, keepdims=True)
    spec = np.abs(rfft(env, axis=1))
    freqs = rfftfreq(x.shape[1], d=1.0 / sample_rate)
    mask = freqs <= 500
    spec_b = spec[:, mask]
    freqs_b = freqs[mask]
    total = spec_b.sum(axis=1, keepdims=True) + 1e-8
    centroid = (spec_b * freqs_b).sum(axis=1, keepdims=True) / total
    peak_idx = spec_b.argmax(axis=1)
    peak_freq = freqs_b[peak_idx].reshape(-1, 1)
    peak_amp = spec_b.max(axis=1, keepdims=True)
    return np.nan_to_num(np.concatenate(
        [centroid, peak_freq, peak_amp], axis=1)).astype(np.float32)


COLUMNS = ["env_centroid", "env_peak_freq", "env_peak_amp"]


def execute(df: pd.DataFrame, signal_col: str = "signal", sample_rate: int = 12000,
            window_len: int = 1024, overlap: float = 0.5, **kwargs) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + COLUMNS)}

    signal = _extract_signal(df, signal_col)
    windows = _sliding_windows(signal, window_len, overlap)
    if len(windows) == 0:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + COLUMNS)}

    feats = _envelope_features(windows, sample_rate)
    result = pd.DataFrame(feats, columns=COLUMNS)
    result.insert(0, "sample_id", np.arange(1, len(result) + 1))
    return {"out_1": result}


if __name__ == "__main__":
    sig = np.sin(2 * np.pi * 30 * np.arange(4096) / 12000) + \
        0.1 * np.random.randn(4096)
    out = execute(pd.DataFrame({"signal": [sig]}))
    print(out["out_1"].head())
