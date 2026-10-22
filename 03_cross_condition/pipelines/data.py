"""数据读取与预处理：CWRU 加载、滑窗切片、逐样本归一化、数据集封装。

CWRU 驱动端 (DE) 12kHz 振动信号，按 (类别, 工况) 二维组织。
文件编号映射（12kHz 驱动端，负载 0/1/2/3 HP）见 FILE_MAP。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from configs.config import DataConfig

# 类别 -> 4 个工况对应的 .mat 文件编号
FILE_MAP: dict[str, tuple[int, int, int, int]] = {
    "N":     (97, 98, 99, 100),
    "IR007": (105, 106, 107, 108),
    "IR014": (169, 170, 171, 172),
    "IR021": (209, 210, 211, 212),
    "B007":  (118, 119, 120, 121),
    "B014":  (185, 186, 187, 188),
    "B021":  (222, 223, 224, 225),
    "OR007": (130, 131, 132, 133),
    "OR014": (197, 198, 199, 200),
    "OR021": (234, 235, 236, 237),
}

CLASS_NAMES_ZH = {
    "N": "正常",
    "IR007": "内圈故障-0.007in",
    "IR014": "内圈故障-0.014in",
    "IR021": "内圈故障-0.021in",
    "B007": "滚动体故障-0.007in",
    "B014": "滚动体故障-0.014in",
    "B021": "滚动体故障-0.021in",
    "OR007": "外圈故障-0.007in",
    "OR014": "外圈故障-0.014in",
    "OR021": "外圈故障-0.021in",
}


def _find_de_signal(mat: dict) -> np.ndarray:
    """从 .mat 中定位驱动端 (DE) 12kHz 信号。"""
    candidates = [k for k in mat.keys()
                  if not k.startswith("__") and "DE_time" in k]
    if not candidates:
        candidates = [k for k in mat.keys()
                      if not k.startswith("__") and "DE" in k.upper()
                      and "FE" not in k.upper()]
    if not candidates:
        raise KeyError(f"未找到驱动端信号，可用键: {list(mat.keys())}")
    sigs = {k: np.asarray(mat[k]).ravel() for k in candidates}
    key = min(sigs, key=lambda k: len(sigs[k]))
    sig = sigs[key]
    # 正常基线数据为 48kHz（约 24 万点），统一降采样到 12kHz
    if len(sig) > 200_000:
        sig = resample_poly(sig, up=1, down=4).astype(np.float64)
    return sig


def load_signal(cwru_dir: Path, cls: str, domain: int) -> np.ndarray:
    """读取某类别在某工况下的原始振动信号。"""
    fileno = FILE_MAP[cls][domain]
    mat_path = cwru_dir / f"{fileno}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"缺少数据文件: {mat_path}（类别 {cls}，工况 {domain}）")
    mat = loadmat(str(mat_path))
    return _find_de_signal(mat)


def load_all(cwru_dir: Path, classes: list[str] | None = None,
             domains: tuple = (0, 1, 2, 3)) -> dict:
    """加载全部 (类别, 工况) 信号，缺失文件跳过并告警。

    Returns:
        dict[(cls, domain)] -> 1D np.ndarray
    """
    classes = classes or list(FILE_MAP.keys())
    out: dict[tuple[str, int], np.ndarray] = {}
    for cls in classes:
        for d in domains:
            try:
                out[(cls, d)] = load_signal(cwru_dir, cls, d)
            except FileNotFoundError as e:
                print(f"[跳过] {e}")
    return out


def sliding_windows(signal: np.ndarray, window_len: int, overlap: float) -> np.ndarray:
    """固定窗长 + 重叠率的滑窗切片 -> (n_samples, window_len)。"""
    step = max(int(window_len * (1.0 - overlap)), 1)
    n = (len(signal) - window_len) // step + 1
    if n <= 0:
        return np.empty((0, window_len))
    idx = np.arange(window_len)[None, :] + step * np.arange(n)[:, None]
    return signal[idx]


def normalize(samples: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """逐样本归一化（每条样本独立归一，避免信息泄露）。"""
    if mode == "none":
        return samples
    if mode == "zscore":
        mu = samples.mean(axis=1, keepdims=True)
        std = samples.std(axis=1, keepdims=True) + 1e-8
        return (samples - mu) / std
    if mode == "minmax":
        lo = samples.min(axis=1, keepdims=True)
        hi = samples.max(axis=1, keepdims=True)
        return (samples - lo) / (hi - lo + 1e-8)
    raise ValueError(f"未知归一化方式: {mode}")


def build_dataset(cwru_dir: Path, cfg: DataConfig,
                  classes: list[str] | None = None,
                  domains: tuple | None = None):
    """构建数据集 -> (X, y, d, class_names)。"""
    classes = list(classes) if classes else list(CLASS_NAMES_ZH.keys())
    domains = domains if domains is not None else cfg.domains

    X_list, y_list, d_list = [], [], []
    signals = load_all(cwru_dir, classes, tuple(domains))
    present = sorted({cls for (cls, _) in signals})
    class2idx = {c: i for i, c in enumerate(present)}
    for (cls, d), sig in signals.items():
        windows = sliding_windows(sig, cfg.window_len, cfg.overlap)
        windows = normalize(windows, cfg.norm)
        X_list.append(windows)
        y_list.append(np.full(len(windows), class2idx[cls], dtype=np.int64))
        d_list.append(np.full(len(windows), d, dtype=np.int64))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list)
    d = np.concatenate(d_list)
    return X, y, d, present


class FaultDataset(Dataset):
    """故障诊断数据集。每条样本 x: (1, window_len)，标签 y 与域标签 d 可选。"""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None,
                 d: np.ndarray | None = None, fft: bool = False):
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (n, 1, L)
        if fft:
            self.X = torch.fft.rfft(self.X, dim=2).abs()
        self.y = torch.from_numpy(y).long() if y is not None else None
        self.d = torch.from_numpy(d).long() if d is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        out = {"x": x}
        if self.y is not None:
            out["y"] = self.y[idx]
        if self.d is not None:
            out["d"] = self.d[idx]
        return out
