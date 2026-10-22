"""域自适应 / 域泛化模型：DANN、DeepCORAL、DAN(MMD)，以及模型工厂。

统一 forward 约定：返回 dict，至少包含 "logits" 与 "feats"；
DANN 额外返回 "d_logits"。域损失由训练流程依据 feats 计算（见 feature_alignment）。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from .cross_condition_model import BACKBONES, FEAT_DIM, build_backbone


class GradReverse(Function):
    """梯度反转层（GRL）。"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def _grl(x, alpha):
    return GradReverse.apply(x, alpha)


class FaultModel(nn.Module):
    """故障诊断模型基类：骨干 + 分类头。"""

    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = build_backbone(backbone_name)
        self.num_classes = num_classes
        self.classifier = nn.Linear(FEAT_DIM, num_classes)

    def forward(self, x, alpha: float | None = None) -> dict:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}


class DANN(FaultModel):
    """对抗域自适应网络 (Domain-Adversarial Neural Network)。"""

    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__(backbone_name, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(FEAT_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, x, alpha: float | None = None):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        alpha = 1.0 if alpha is None else alpha
        d_logits = self.domain_head(_grl(feats, alpha))
        return {"logits": logits, "feats": feats, "d_logits": d_logits}


class DeepCORAL(FaultModel):
    """Deep CORAL：对齐源域与目标域特征的二阶统计量。"""
    # forward 沿用基类，域损失在训练流程中基于 feats 计算
    pass


class DAN(FaultModel):
    """Deep Adaptation Network：以多核 MMD 对齐源/目标特征分布。"""
    # forward 沿用基类，域损失在训练流程中基于 feats 计算
    pass


# ---- 模型工厂（按名切换，算子集成接口） ----

_MODELS: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        _MODELS[name] = cls
        return cls
    return deco


# 骨干（无域自适应，源域训练直接跨工况测试）
for _name in BACKBONES:
    _MODELS[f"src_{_name}"] = FaultModel

# 域自适应模型（各骨干均可）
for _name in BACKBONES:
    _MODELS[f"dann_{_name}"] = DANN
    _MODELS[f"coral_{_name}"] = DeepCORAL
    _MODELS[f"dan_{_name}"] = DAN


def list_models() -> list[str]:
    return sorted(_MODELS.keys())


def create_model(name: str, num_classes: int, backbone: str | None = None) -> nn.Module:
    """按名称创建模型，例如 "src_wdcnn"、"dann_wdcnn"。

    Args:
        name: 模型名。
        num_classes: 故障类别数。
        backbone: 显式指定骨干；缺省时从 name 的 "_" 后缀解析。
    """
    if name not in _MODELS:
        raise ValueError(f"未知模型: {name}，可选: {list_models()}")
    if backbone is None:
        backbone = name.split("_", 1)[1] if "_" in name else "wdcnn"
    return _MODELS[name](backbone_name=backbone, num_classes=num_classes)


def load_checkpoint(model: nn.Module, path: str, map_location: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model
