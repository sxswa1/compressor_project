"""评估：准确率、混淆矩阵、分类报告、虚警率、漏警率。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list[str], normal_idx: int = 0) -> dict:
    """计算完整评估指标。

    Args:
        y_true / y_pred: 真实/预测标签。
        class_names: 类别名列表。
        normal_idx: "正常" 类在类别列表中的索引（用于虚警/漏警统计）。
    """
    acc = float((y_true == y_pred).mean())
    target_names = [str(c) for c in class_names]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=target_names, digits=4, zero_division=0)

    normal_mask = y_true == normal_idx
    false_alarm_rate = float((y_pred[normal_mask] != normal_idx).mean()) \
        if normal_mask.sum() > 0 else float("nan")

    fault_mask = y_true != normal_idx
    miss_rate = float((y_pred[fault_mask] != y_true[fault_mask]).mean()) \
        if fault_mask.sum() > 0 else float("nan")

    return {
        "acc": acc,
        "confusion_matrix": cm,
        "report": report,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"]
        out = model(x)
        ys.append(y.numpy())
        preds.append(out["logits"].argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def evaluate_loader(model: nn.Module, loader: DataLoader,
                    device: torch.device) -> dict:
    y_true, y_pred = predict(model, loader, device)
    return {"acc": float((y_true == y_pred).mean())}


def evaluate(model: nn.Module, loader: DataLoader,
             class_names: list[str], device: torch.device,
             normal_idx: int | None = None) -> dict:
    # 自动定位"正常"类（类别名 "N"），避免因类别排序导致虚警/漏警统计错位。
    if normal_idx is None:
        normal_idx = class_names.index("N") if "N" in class_names else 0
    y_true, y_pred = predict(model, loader, device)
    return compute_metrics(y_true, y_pred, class_names, normal_idx)
