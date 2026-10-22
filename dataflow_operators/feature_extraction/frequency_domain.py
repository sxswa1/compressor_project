"""频域与 FFT 分析算子（自包含）。

对切片后的信号做 FFT，计算频域统计特征（谱质心/谱扩散/频带能量占比/谱峰，共 7 维）。

输入: df —— 振动信号表
输出: out_1 —— 特征表（每行一个切片样本）
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
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


def _freq_features(x: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float64)
    n = x.shape[1]
    spec = np.abs(rfft(x, axis=1))[:, 1:]
    freqs = rfftfreq(n, d=1.0 / sample_rate)[1:]
    total = spec.sum(axis=1, keepdims=True) + 1e-8
    centroid = (spec * freqs).sum(axis=1, keepdims=True) / total
    spread = np.sqrt(((spec * (freqs - centroid) ** 2).sum(axis=1, keepdims=True)) / total)
    fmax = sample_rate / 2
    low = (freqs <= fmax / 3).astype(float)
    mid = ((freqs > fmax / 3) & (freqs <= 2 * fmax / 3)).astype(float)
    high = (freqs > 2 * fmax / 3).astype(float)
    band_ratio = np.stack([
        (spec * low).sum(axis=1), (spec * mid).sum(axis=1), (spec * high).sum(axis=1),
    ], axis=1) / total
    peak_idx = spec.argmax(axis=1)
    peak_freq = freqs[peak_idx].reshape(-1, 1)
    peak_amp = spec.max(axis=1, keepdims=True)
    return np.nan_to_num(np.concatenate(
        [centroid, spread, band_ratio, peak_freq, peak_amp], axis=1)).astype(np.float32)


COLUMNS = ["centroid", "spread", "low_band_ratio", "mid_band_ratio",
           "high_band_ratio", "peak_freq", "peak_amp"]


def execute(df: pd.DataFrame, signal_col: str = "signal", sample_rate: int = 12000,
            window_len: int = 1024, overlap: float = 0.5, **kwargs) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + COLUMNS)}

    signal = _extract_signal(df, signal_col)
    windows = _sliding_windows(signal, window_len, overlap)
    if len(windows) == 0:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + COLUMNS)}

    feats = _freq_features(windows, sample_rate)
    result = pd.DataFrame(feats, columns=COLUMNS)
    result.insert(0, "sample_id", np.arange(1, len(result) + 1))
    return {"out_1": result}


if __name__ == "__main__":
    sig = np.sin(2 * np.pi * 30 * np.arange(4096) / 12000) + \
        0.1 * np.random.randn(4096)
    out = execute(pd.DataFrame({"signal": [sig]}))
    print(out["out_1"].head())
