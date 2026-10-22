"""经验模态分解（EMD）算子（自包含）。

对切片信号做 EMD，提取前若干 IMF 的能量占比特征（默认 4 维）。

输入: df —— 振动信号表
输出: out_1 —— 特征表
依赖: PyEMD（import 名 PyEMD，缺失时输出全零特征）
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


def _emd_features(x: np.ndarray, n_imfs: int) -> np.ndarray:
    try:
        from PyEMD import EMD
    except ImportError:
        return np.zeros((len(np.atleast_2d(x)), n_imfs), dtype=np.float32)
    x = np.atleast_2d(x).astype(np.float64)
    emd = EMD()
    out = []
    for row in x:
        imfs = emd(row, max_imf=n_imfs)
        imfs = np.atleast_2d(imfs)
        energies = [float((imf ** 2).sum()) for imf in imfs[:n_imfs]]
        energies += [0.0] * (n_imfs - len(energies))
        total = sum(energies) + 1e-8
        out.append([e / total for e in energies])
    return np.asarray(out, dtype=np.float32)


def execute(df: pd.DataFrame, signal_col: str = "signal", n_imfs: int = 4,
            window_len: int = 1024, overlap: float = 0.5,
            **kwargs) -> Dict[str, pd.DataFrame]:
    columns = [f"imf_e{i}" for i in range(n_imfs)]
    if df is None or df.empty:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + columns)}

    signal = _extract_signal(df, signal_col)
    windows = _sliding_windows(signal, window_len, overlap)
    if len(windows) == 0:
        return {"out_1": pd.DataFrame(columns=["sample_id"] + columns)}

    feats = _emd_features(windows, n_imfs)
    result = pd.DataFrame(feats, columns=columns)
    result.insert(0, "sample_id", np.arange(1, len(result) + 1))
    return {"out_1": result}


if __name__ == "__main__":
    sig = np.sin(2 * np.pi * 30 * np.arange(4096) / 12000) + \
        0.1 * np.random.randn(4096)
    out = execute(pd.DataFrame({"signal": [sig]}))
    print(out["out_1"].head())
