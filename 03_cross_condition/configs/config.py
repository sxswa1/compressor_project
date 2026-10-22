"""跨工况泛化诊断 —— 全局配置。

路径、数据切片、训练超参与跨工况协议集中管理。
数据默认位于项目根目录 data/CWRU（公开数据集）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录：configs/ -> 03_cross_condition/ -> 项目根
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJ_ROOT / "data"
CWRU_DIR = DATA_DIR / "CWRU"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class DataConfig:
    """CWRU 轴承数据集与切片配置。"""
    sample_rate: int = 12000          # 采样频率 (Hz)
    window_len: int = 1024            # 每个样本的采样点数
    overlap: float = 0.5              # 滑窗重叠率
    norm: str = "zscore"             # 归一化方式: zscore / minmax / none
    # 工况（负荷）列表：0/1/2/3 HP，作为跨工况的"域"
    domains: tuple = (0, 1, 2, 3)
    # 故障类别 -> 中文描述
    classes: dict = field(default_factory=lambda: {
        "N":       ("正常",),
        "IR007":   ("内圈故障-0.007in",),
        "IR014":   ("内圈故障-0.014in",),
        "IR021":   ("内圈故障-0.021in",),
        "B007":    ("滚动体故障-0.007in",),
        "B014":    ("滚动体故障-0.014in",),
        "B021":    ("滚动体故障-0.021in",),
        "OR007":   ("外圈故障-0.007in",),
        "OR014":   ("外圈故障-0.014in",),
        "OR021":   ("外圈故障-0.021in",),
    })


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"          # auto / cuda / cpu
    val_ratio: float = 0.2
    # 跨工况协议: leave_one_out / source_target
    protocol: str = "leave_one_out"
    # source_target 协议下的源/目标工况，例如 (0, 1, 2) -> (3,)
    source_domains: tuple = (0, 1, 2)
    target_domains: tuple = (3,)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
