"""时域统计分析算子（自包含）。

将上游振动信号表切片后，逐样本计算时域统计特征（均值/RMS/峰值/峭度等 13 维）。

输入: df —— 振动信号表（signal_col 列，支持长列或单单元格数组）
输出: out_1 —— 特征表（每行一个切片样本，列为 13 维时域特征 + sample_id）
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


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


def _time_features(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    rms = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
    peak = np.abs(x).max(axis=1, keepdims=True)
    pp = x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)
    abs_mean = np.abs(x).mean(axis=1, keepdims=True)
    energy = (x ** 2).sum(axis=1, keepdims=True)
    crest = peak / (rms + 1e-8)
    shape = rms / (abs_mean + 1e-8)
    impulse = peak / (abs_mean + 1e-8)
    sqrt_mean = np.sqrt(np.abs(x)).mean(axis=1, keepdims=True)
    margin = peak / (sqrt_mean ** 2 + 1e-8)
    kur = kurtosis(x, axis=1, fisher=True).reshape(-1, 1)
    skew_ = skew(x, axis=1).reshape(-1, 1)
    return np.nan_to_num(np.concatenate([
        mean, std, rms, peak, pp, abs_mean, energy,
        crest, shape, impulse, margin, kur, skew_,
    ], axis=1)).astype(np.float32)


COLUMNS = ["mean", "std", "rms", "peak", "peak2peak", "abs_mean", "energy",
           "crest_factor", "shape_factor", "impulse_factor", "margin_factor",
           "kurtosis", "skewness"]


def execute(df: pd.DataFrame, signal_col: str = "signal", window_len: int = 1024,
            overlap: float = 0.5, **kwargs) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + COLUMNS)}

    signal = _extract_signal(df, signal_col)
    windows = _sliding_windows(signal, window_len, overlap)
    if len(windows) == 0:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + COLUMNS)}

    feats = _time_features(windows)
    result = pd.DataFrame(feats, columns=COLUMNS)
    result.insert(0, "sample_id", np.arange(1, len(result) + 1))
    return {"out_1": result}


if __name__ == "__main__":
    sig = np.sin(2 * np.pi * 30 * np.arange(4096) / 12000) + \
        0.1 * np.random.randn(4096)
    out = execute(pd.DataFrame({"signal": [sig]}))
    print(out["out_1"].head())
