"""小波变换算子（自包含）。

使用 db4 小波做多级分解，提取各子带能量占比（默认 4 层，输出 5 维）。

输入: df —— 振动信号表
输出: out_1 —— 特征表
依赖: PyWavelets（import 名 pywt，缺失时输出全零特征）
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


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


def _wavelet_features(x: np.ndarray, wavelet: str, level: int) -> np.ndarray:
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


def execute(df: pd.DataFrame, signal_col: str = "signal", wavelet: str = "db4",
            level: int = 4, window_len: int = 1024, overlap: float = 0.5,
            **kwargs) -> Dict[str, pd.DataFrame]:
    columns = [f"wavelet_e{i}" for i in range(level + 1)]
    if df is None or df.empty:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + columns)}

    signal = _extract_signal(df, signal_col)
    windows = _sliding_windows(signal, window_len, overlap)
    if len(windows) == 0:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + columns)}

    feats = _wavelet_features(windows, wavelet, level)
    result = pd.DataFrame(feats, columns=columns)
    result.insert(0, "sample_id", np.arange(1, len(result) + 1))
    return {"out_1": result}


if __name__ == "__main__":
    sig = np.sin(2 * np.pi * 30 * np.arange(4096) / 12000) + \
        0.1 * np.random.randn(4096)
    out = execute(pd.DataFrame({"signal": [sig]}))
    print(out["out_1"].head())
