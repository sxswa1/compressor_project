"""工况识别与片段切分算子（规则版，自包含）。

规则版工况识别采用"负荷标签 + 转速换算"：工况（域）= 电机负荷档位 0/1/2/3 HP，
对应转速 1797/1772/1750/1730 RPM。判定优先级：
  给定转速 rpm > 给定工况档位 domain > 从信号频谱估计转频后映射。

输入: df —— 振动信号表（可选 signal / rpm / domain 列）
输出: out_1 —— 逐片段工况识别结果；out_2 —— 工况分布统计
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

# 工况(负荷档位) -> 电机转速 RPM
DOMAIN_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
FR_MIN, FR_MAX = 26.0, 32.0  # 转频搜索范围 (Hz)


def rpm_to_domain(rpm: float) -> int:
    return int(min(DOMAIN_RPM, key=lambda d: abs(DOMAIN_RPM[d] - float(rpm))))


def domain_to_rpm(domain: int) -> float:
    return DOMAIN_RPM[int(domain)]


def _estimate_shaft_freq(signal: np.ndarray, sample_rate: int):
    x = np.asarray(signal, dtype=np.float64).ravel()
    if len(x) < 8:
        return None
    spec = np.abs(rfft(x))
    freqs = rfftfreq(len(x), d=1.0 / sample_rate)
    mask = (freqs >= FR_MIN) & (freqs <= FR_MAX)
    if not mask.any():
        return None
    return float(freqs[mask][int(np.argmax(spec[mask]))])


def _identify_condition(signal: np.ndarray, sample_rate: int,
                        rpm, domain):
    if domain is not None:
        return int(domain)
    if rpm is not None:
        return rpm_to_domain(rpm)
    fr = _estimate_shaft_freq(signal, sample_rate)
    if fr is not None:
        return rpm_to_domain(fr * 60.0)
    return -1


def _sliding_windows(signal: np.ndarray, window_len: int, overlap: float) -> np.ndarray:
    step = max(int(window_len * (1.0 - overlap)), 1)
    n = (len(signal) - window_len) // step + 1
    if n <= 0:
        return np.empty((0, window_len))
    idx = np.arange(window_len)[None, :] + step * np.arange(n)[:, None]
    return signal[idx]


def _first_numeric_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))


def execute(df: pd.DataFrame, signal_col: str = "signal", rpm_col: str = "rpm",
            domain_col: str = "domain", sample_rate: int = 12000,
            window_len: int = 1024, overlap: float = 0.5,
            **kwargs) -> Dict[str, pd.DataFrame]:
    empty1 = pd.DataFrame(columns=["sample_id", "condition", "rpm", "shaft_freq_hz"])
    empty2 = pd.DataFrame(columns=["condition", "count", "ratio"])
    if df is None or df.empty:
        return {"out_1": empty1, "out_2": empty2}

    col = signal_col if signal_col in df.columns else _first_numeric_col(df)
    vals = df[col].to_numpy()
    if len(vals) == 1 and hasattr(vals[0], "__len__"):
        signal = np.asarray(vals[0], dtype=np.float64).ravel()
    else:
        signal = np.asarray(vals, dtype=np.float64).ravel()

    rpm = float(np.asarray(df[rpm_col]).mean()) if rpm_col in df.columns else None
    domain = int(np.asarray(df[domain_col]).mean()) if domain_col in df.columns else None

    windows = _sliding_windows(signal, window_len, overlap)
    if len(windows) == 0:
        return {"out_1": empty1, "out_2": empty2}

    cond = _identify_condition(signal, sample_rate, rpm, domain)
    est_fr = _estimate_shaft_freq(signal, sample_rate)
    conds = np.full(len(windows), cond, dtype=np.int64)

    result = pd.DataFrame({
        "sample_id": np.arange(1, len(windows) + 1),
        "condition": conds,
        "rpm": (domain_to_rpm(cond) if cond >= 0 else None),
        "shaft_freq_hz": round(est_fr, 3) if est_fr is not None else None,
    })

    n = len(result)
    counts = result["condition"].value_counts()
    summary = pd.DataFrame([{
        "condition": int(c), "count": int(counts[c]),
        "ratio": round(float(counts[c] / n), 4),
    } for c in counts.index])

    return {"out_1": result, "out_2": summary}


if __name__ == "__main__":
    sig = np.sin(2 * np.pi * 29.95 * np.arange(12000) / 12000) + \
        0.05 * np.random.randn(12000)
    df = pd.DataFrame({"signal": [sig]})
    out = execute(df)
    print(out["out_1"].head())
    print(out["out_2"])
