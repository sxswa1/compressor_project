"""跨工况泛化诊断 —— 模型层。

- cross_condition_model: 骨干网络（CNN/WDCNN/ResNet/CNN-LSTM）
- domain_adaptation:    域自适应模型（DANN/DeepCORAL/DAN）+ 模型工厂
- feature_alignment:    域对齐损失（CORAL/MMD）+ 物理机理阶次域特征
- traditional:          随机森林 / 梯度提升基线
"""
from .cross_condition_model import BACKBONES, FEAT_DIM, build_backbone
from .domain_adaptation import (
    DANN,
    DAN,
    DeepCORAL,
    FaultModel,
    create_model,
    list_models,
)
from .feature_alignment import (
    FAULT_ORDERS,
    ORDER_LABELS,
    coral_loss,
    fault_frequencies,
    mmd_rbf,
    physics_diagnose,
    physics_features,
)

__all__ = [
    "BACKBONES",
    "FEAT_DIM",
    "build_backbone",
    "DANN",
    "DAN",
    "DeepCORAL",
    "FaultModel",
    "create_model",
    "list_models",
    "FAULT_ORDERS",
    "ORDER_LABELS",
    "coral_loss",
    "fault_frequencies",
    "mmd_rbf",
    "physics_diagnose",
    "physics_features",
]
