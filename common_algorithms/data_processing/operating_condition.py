"""工况识别与片段切分（规则版）。

工况（域）= 电机负荷档位 0/1/2/3 HP，对应转速 1797/1772/1750/1730 RPM
（CWRU 驱动端轴承，SKF 6205-2RS）。

规则版工况识别采用"负荷标签 + 转速换算"的确定性规则：
  1. 若给定工况档位 domain，直接使用；
  2. 否则若给定转速 rpm，按最近转速映射到工况档位；
  3. 否则从振动信号频谱在转频频带（26~32 Hz）内取能量最大峰估计转频，再映射。

片段切分采用固定窗长 + 重叠率的滑窗，逐片段标注工况。
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq

# 工况(负荷档位) -> 电机转速 RPM
DOMAIN_RPM: dict[int, float] = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

# 转频估计搜索范围（对应 1730~1797 RPM 的转频带，留有余量）
FR_MIN, FR_MAX = 26.0, 32.0  # Hz


def rpm_to_domain(rpm: float) -> int:
    """按最近转速映射到工况档位。"""
    return int(min(DOMAIN_RPM, key=lambda d: abs(DOMAIN_RPM[d] - float(rpm))))


def domain_to_rpm(domain: int) -> float:
    """工况档位 -> 转速 RPM。"""
    if int(domain) not in DOMAIN_RPM:
        raise ValueError(f"未知工况档位: {domain}，可选 {list(DOMAIN_RPM)}")
    return DOMAIN_RPM[int(domain)]


def estimate_shaft_freq(signal: np.ndarray, sample_rate: int) -> float | None:
    """规则版转频估计：在转频频带内取幅值谱最大峰。

    Args:
        signal: 一维振动信号。
        sample_rate: 采样频率 (Hz)。

    Returns:
        转频 (Hz)；若频带内无有效峰则返回 None。
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if len(x) < 8:
        return None
    spec = np.abs(rfft(x))
    freqs = rfftfreq(len(x), d=1.0 / sample_rate)
    mask = (freqs >= FR_MIN) & (freqs <= FR_MAX)
    if not mask.any():
        return None
    return float(freqs[mask][int(np.argmax(spec[mask]))])


def identify_condition(signal: np.ndarray | None = None, sample_rate: int = 12000,
                       rpm: float | None = None, domain: int | None = None) -> int | None:
    """规则版工况识别。

    判定优先级：给定 domain > 给定 rpm > 从信号频谱估计转频。

    Args:
        signal: 一维振动信号（可选）。
        sample_rate: 采样频率 (Hz)。
        rpm: 转速 (RPM，可选)。
        domain: 已知工况档位（可选）。

    Returns:
        工况档位 0/1/2/3；无法判定时返回 None。
    """
    if domain is not None:
        return int(domain)
    if rpm is not None:
        return rpm_to_domain(rpm)
    if signal is not None:
        fr = estimate_shaft_freq(signal, sample_rate)
        if fr is not None:
            return rpm_to_domain(fr * 60.0)
    return None


def sliding_windows(signal: np.ndarray, window_len: int, overlap: float) -> np.ndarray:
    """固定窗长 + 重叠率的滑窗切片。

    Args:
        signal: 一维信号。
        window_len: 窗长（采样点数）。
        overlap: 重叠率 [0, 1)。

    Returns:
        样本矩阵 (n_samples, window_len)；长度不足时返回空矩阵 (0, window_len)。
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    step = max(int(window_len * (1.0 - overlap)), 1)
    n = (len(x) - window_len) // step + 1
    if n <= 0:
        return np.empty((0, window_len), dtype=np.float64)
    idx = np.arange(window_len)[None, :] + step * np.arange(n)[:, None]
    return x[idx]


def identify_and_segment(signal: np.ndarray, sample_rate: int = 12000,
                         window_len: int = 1024, overlap: float = 0.5,
                         rpm: float | None = None, domain: int | None = None):
    """切分为工况片段，并逐片段做规则版工况识别。

    Returns:
        (windows (n, window_len), conditions (n,) int)
        其中 conditions 的每个元素为识别出的工况档位，无法判定时为 -1。
    """
    windows = sliding_windows(signal, window_len, overlap)
    cond = identify_condition(signal, sample_rate, rpm=rpm, domain=domain)
    conditions = np.full(len(windows), -1 if cond is None else cond, dtype=np.int64)
    return windows, conditions
