"""经验模态分解（EMD）特征。

将信号分解为若干本征模态函数（IMF），提取各 IMF 能量占比，
默认取前 4 个 IMF 输出 4 维。
"""
from __future__ import annotations

import numpy as np


def emd_features(x: np.ndarray, n_imfs: int = 4) -> np.ndarray:
    """逐样本 EMD 的 IMF 能量占比特征。

    Args:
        x: 输入信号 (n_samples, window_len)。
        n_imfs: 提取的 IMF 数量。

    Returns:
        特征矩阵 (n_samples, n_imfs)。缺少 PyEMD 时返回全零矩阵。
    """
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
        # 补零到固定长度
        energies += [0.0] * (n_imfs - len(energies))
        total = sum(energies) + 1e-8
        out.append([e / total for e in energies])
    return np.asarray(out, dtype=np.float32)
