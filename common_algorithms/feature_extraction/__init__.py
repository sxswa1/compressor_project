"""公共算法库 —— 特征提取子模块。

供三个研究子项目（超前预警 / 小样本诊断 / 跨工况诊断）直接复用。
李佳耀主责开发：时域统计、频域与 FFT、包络谱、小波、EMD。
"""
from .time_domain import time_features
from .frequency_domain import freq_features
from .envelope_spectrum import envelope_features
from .wavelet import wavelet_features
from .emd import emd_features

__all__ = [
    "time_features",
    "freq_features",
    "envelope_features",
    "wavelet_features",
    "emd_features",
]

from .process_features import process_features
__all__.append('process_features')
