"""传统机器学习基线：随机森林、梯度提升。

在手工特征（时域/频域/包络谱/小波/EMD/物理机理阶次域）上训练，
作为跨工况诊断的对照方法。属"共用/适配"：调用公共特征提取 + 本方向配置。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from common_algorithms.feature_extraction import (
    emd_features,
    envelope_features,
    freq_features,
    time_features,
    wavelet_features,
)

from .feature_alignment import physics_features

TRADITIONAL_MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_features(X: np.ndarray, sample_rate: int, rpm: float | None = None,
                   use_wavelet: bool = True, use_emd: bool = False,
                   use_physics: bool = True) -> np.ndarray:
    """将原始信号样本转为特征矩阵（公共特征 + 物理阶次域特征）。"""
    feats = [time_features(X), freq_features(X, sample_rate),
             envelope_features(X, sample_rate)]
    if use_wavelet:
        feats.append(wavelet_features(X))
    if use_emd:
        feats.append(emd_features(X))
    feats = np.concatenate(feats, axis=1)
    if use_physics and rpm is not None:
        feats = np.concatenate(
            [feats, physics_features(X, sample_rate, rpm)], axis=1)
    return feats.astype(np.float32)


def _default_params(name: str) -> dict:
    if name == "random_forest":
        return {"n_estimators": 200, "max_depth": None,
                "random_state": 42, "n_jobs": -1}
    return {"n_estimators": 200, "learning_rate": 0.1,
            "max_depth": 3, "random_state": 42}


def train_traditional(name: str, X_train: np.ndarray, y_train: np.ndarray):
    """训练传统模型。"""
    if name not in TRADITIONAL_MODELS:
        raise ValueError(f"未知传统模型: {name}，可选: {list(TRADITIONAL_MODELS)}")
    model = TRADITIONAL_MODELS[name](**_default_params(name))
    model.fit(X_train, y_train)
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
