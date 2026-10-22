"""特征对齐：域自适应损失（CORAL / 多核 MMD）与物理机理阶次域特征。

两部分：
  1. 域损失：CORAL（二阶统计量对齐）、MMD（多核高斯核分布对齐）；
  2. 物理机理特征：轴承故障特征频率与转速成正比，将频率域转到阶次域后，
     故障阶次在不同负荷/转速下保持恒定，得到对工况变化相对稳定的特征。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.fft import rfft, rfftfreq

# ---- 域自适应损失 ----

def _cov(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    x = x - x.mean(0, keepdim=True)
    return (x.t() @ x) / (n - 1)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL：源/目标特征协方差矩阵的 Frobenius 距离。"""
    d = source.size(1)
    cs, ct = _cov(source), _cov(target)
    return (cs - ct).pow(2).sum() / (4 * d * d)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    xx = (x * x).sum(1, keepdim=True)
    yy = (y * y).sum(1, keepdim=True)
    dist = xx + yy.t() - 2 * x @ y.t()
    return torch.exp(-gamma * dist)


def mmd_rbf(source: torch.Tensor, target: torch.Tensor,
            kernel_num: int = 5, kernel_mul: float = 2.0,
            fix_sigma: float | None = None) -> torch.Tensor:
    """多核最大均值差异（MMD，高斯核）。"""
    n_s, n_t = source.size(0), target.size(0)
    total = torch.cat([source, target], dim=0)

    if fix_sigma is not None:
        gammas = [1.0 / (2 * fix_sigma ** 2)]
    else:
        dist = (total.unsqueeze(0) - total.unsqueeze(1)).pow(2).sum(2)
        sigma = dist.detach().median().sqrt().clamp_min(1e-6)
        gammas = [1.0 / (2 * (sigma * (2 ** k)) ** 2)
                  for k in range(kernel_num)]

    k_ss = sum(_rbf_kernel(source, source, g) for g in gammas)
    k_tt = sum(_rbf_kernel(target, target, g) for g in gammas)
    k_st = sum(_rbf_kernel(source, target, g) for g in gammas)

    mmd = k_ss.sum() / (n_s * n_s) + k_tt.sum() / (n_t * n_t) \
        - 2 * k_st.sum() / (n_s * n_t)
    return mmd.clamp_min(0)


def domain_loss_fn(name: str):
    if name == "coral":
        return coral_loss
    if name == "dan":
        return mmd_rbf
    raise ValueError(f"未知域损失: {name}")


# ---- 物理机理阶次域特征 ----

# SKF 6205-2RS JEM（CWRU 驱动端轴承）故障特征阶次（相对转频的倍数）
FAULT_ORDERS = {
    "BPFI": 5.415,   # 内圈
    "BPFO": 3.585,   # 外圈
    "BSF":  2.357,   # 滚动体
    "FTF":  0.3983,  # 保持架
}
ORDER_LABELS = list(FAULT_ORDERS.keys())


def shaft_freq(rpm: float) -> float:
    """转频 (Hz)。"""
    return rpm / 60.0


def fault_frequencies(rpm: float) -> dict[str, float]:
    """按转速计算各故障特征频率 (Hz)。"""
    fr = shaft_freq(rpm)
    return {k: v * fr for k, v in FAULT_ORDERS.items()}


def _envelope_spectrum(x: np.ndarray, sample_rate: int):
    analytic = signal.hilbert(x)
    env = np.abs(analytic) - np.abs(analytic).mean()
    spec = np.abs(rfft(env))
    freqs = rfftfreq(len(x), d=1.0 / sample_rate)
    return freqs, spec


def physics_features(x: np.ndarray, sample_rate: int, rpm: float,
                     n_harmonics: int = 3) -> np.ndarray:
    """阶次域包络谱特征：在各故障阶次及其谐波处提取归一化幅值。

    返回 (n_samples, n_orders * n_harmonics)。由于按阶次归一，
    对负荷 / 转速变化相对稳定，适合跨工况诊断。
    """
    x = np.atleast_2d(x)
    fr = shaft_freq(rpm)
    feats = []
    for row in x:
        freqs, spec = _envelope_spectrum(row, sample_rate)
        total = spec.sum() + 1e-8
        row_feats = []
        for order in FAULT_ORDERS.values():
            for h in range(1, n_harmonics + 1):
                f = order * fr * h
                lo, hi = f * 0.95, f * 1.05   # ±5% 容差，容忍转速波动
                mask = (freqs >= lo) & (freqs <= hi)
                val = spec[mask].max() if mask.any() else 0.0
                row_feats.append(val / total)
        feats.append(row_feats)
    return np.asarray(feats, dtype=np.float32)


def physics_diagnose(x: np.ndarray, sample_rate: int, rpm: float) -> np.ndarray:
    """机理规则诊断：以包络谱中占优的故障阶次判断故障位置。

    返回 0..3 的索引，对应 ORDER_LABELS = [BPFI, BPFO, BSF, FTF]。
    """
    feats = physics_features(x, sample_rate, rpm, n_harmonics=1)
    n_orders = len(FAULT_ORDERS)
    return feats.argmax(axis=1) % n_orders
