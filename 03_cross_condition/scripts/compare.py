"""多模型横向对比：深度学习 + 域自适应 + 传统机器学习 + 物理机理规则。

用法:
    python 03_cross_condition/scripts/compare.py --source 0 1 2 --target 3
输出: 03_cross_condition/outputs/comparison.csv（跨工况诊断精度对比表）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT_03 = HERE.parent
PROJ = ROOT_03.parent
sys.path.insert(0, str(ROOT_03))
sys.path.insert(0, str(PROJ))

from common_algorithms.data_processing import DOMAIN_RPM  # noqa: E402
from configs.config import CWRU_DIR, Config  # noqa: E402
from models.domain_adaptation import create_model  # noqa: E402
from models.feature_alignment import (  # noqa: E402
    ORDER_LABELS,
    physics_diagnose,
    physics_features,
)
from models.traditional import build_features, train_traditional  # noqa: E402
from pipelines.data import FaultDataset, build_dataset  # noqa: E402
from pipelines.train import _device, train  # noqa: E402

OUT_DIR = ROOT_03 / "outputs"

# 粗粒度故障位置（用于物理机理规则诊断）
POSITION = {"N": "normal", "IR007": "inner", "IR014": "inner", "IR021": "inner",
            "B007": "ball", "B014": "ball", "B021": "ball",
            "OR007": "outer", "OR014": "outer", "OR021": "outer"}
POS2IDX = {"inner": 0, "outer": 1, "ball": 2, "normal": 3}


def _featurize(X: np.ndarray, d: np.ndarray, sample_rate: int) -> np.ndarray:
    """提取特征（含物理机理特征，按样本所属工况取对应转速）。"""
    base = build_features(X, sample_rate, use_physics=False)
    n_order = len(ORDER_LABELS)
    phys = np.zeros((len(X), n_order * 3), dtype=np.float32)
    for dom in np.unique(d):
        mask = d == dom
        phys[mask] = physics_features(X[mask], sample_rate, DOMAIN_RPM[int(dom)])
    return np.concatenate([base, phys], axis=1)


def run_deep(model_name: str, X, y, d, source, target, cfg, device) -> float:
    from pipelines.evaluate import evaluate
    src_mask = np.isin(d, source)
    tgt_mask = np.isin(d, target)
    src_loader = torch.utils.data.DataLoader(
        FaultDataset(X[src_mask], y[src_mask], d[src_mask]),
        batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    tgt_loader = torch.utils.data.DataLoader(
        FaultDataset(X[tgt_mask], y[tgt_mask], d[tgt_mask]),
        batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    model = create_model(model_name, num_classes=len(np.unique(y)))
    train(model, model_name, src_loader, tgt_loader, cfg.train)
    return evaluate(model, tgt_loader, list(np.unique(y)), device)["acc"]


def run_traditional(model_name: str, X, y, d, source, target,
                    sample_rate: int) -> float:
    src_mask = np.isin(d, source)
    tgt_mask = np.isin(d, target)
    Xs = _featurize(X[src_mask], d[src_mask], sample_rate)
    Xt = _featurize(X[tgt_mask], d[tgt_mask], sample_rate)
    model = train_traditional(model_name, Xs, y[src_mask])
    pred = model.predict(Xt)
    return float((pred == y[tgt_mask]).mean())


def run_physics(X, y, d, source, target, sample_rate: int, class_names) -> float:
    """机理规则：以占优故障阶次判断粗粒度故障位置（3 类）。"""
    tgt_mask = np.isin(d, target)
    Xt, yt, dt = X[tgt_mask], y[tgt_mask], d[tgt_mask]
    pred_pos, true_pos = [], []
    for dom in np.unique(dt):
        mask = dt == dom
        pred_pos.append(physics_diagnose(Xt[mask], sample_rate, DOMAIN_RPM[int(dom)]))
        true_pos.append(np.array([POS2IDX[POSITION[class_names[c]]] for c in yt[mask]]))
    pred_pos = np.concatenate(pred_pos)
    true_pos = np.concatenate(true_pos)
    mask = true_pos != POS2IDX["normal"]
    return float((pred_pos[mask] == true_pos[mask]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target", type=int, nargs="+", default=[3])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=str(OUT_DIR / "comparison.csv"))
    args = ap.parse_args()

    cfg = Config()
    cfg.train.epochs = args.epochs
    device = _device(cfg.train)
    X, y, d, class_names = build_dataset(CWRU_DIR, cfg.data)
    print(f"样本={len(X)} 类别={len(class_names)} 源={args.source} 目标={args.target}")

    methods = {
        "CNN": ("src_cnn1d", "deep"),
        "WDCNN": ("src_wdcnn", "deep"),
        "LSTM": ("src_lstm", "deep"),
        "DANN": ("dann_wdcnn", "deep"),
        "DeepCORAL": ("coral_wdcnn", "deep"),
        "DAN-MMD": ("dan_wdcnn", "deep"),
        "随机森林": ("random_forest", "traditional"),
        "梯度提升": ("gradient_boosting", "traditional"),
        "物理机理规则": (None, "physics"),
    }

    rows = []
    for label, (name, kind) in methods.items():
        print(f"\n>>> {label} ({name})")
        if kind == "deep":
            acc = run_deep(name, X, y, d, args.source, args.target, cfg, device)
        elif kind == "traditional":
            acc = run_traditional(name, X, y, d, args.source, args.target,
                                  cfg.data.sample_rate)
        else:
            acc = run_physics(X, y, d, args.source, args.target,
                              cfg.data.sample_rate, class_names)
        rows.append({"方法": label, "目标工况准确率": round(acc, 4)})
        print(f"    {label}: {acc:.4f}")

    df = pd.DataFrame(rows)
    print("\n===== 对比结果 =====")
    print(df.to_string(index=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
