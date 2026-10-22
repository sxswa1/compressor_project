"""跨工况泛化诊断 —— 训练/评估命令行入口。

用法（在项目根目录执行）:
    python 03_cross_condition/scripts/train.py --model src_wdcnn --source 0 1 2 --target 3
    python 03_cross_condition/scripts/train.py --model dann_wdcnn --source 0 1 2 --target 3
    python 03_cross_condition/scripts/train.py --model src_wdcnn --protocol leave_one_out
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# 路径引导：03_cross_condition/ 与项目根加入 sys.path
HERE = Path(__file__).resolve().parent
ROOT_03 = HERE.parent
PROJ = ROOT_03.parent
sys.path.insert(0, str(ROOT_03))
sys.path.insert(0, str(PROJ))

from configs.config import CWRU_DIR, Config  # noqa: E402
from models.domain_adaptation import create_model, list_models  # noqa: E402
from pipelines.data import FaultDataset, build_dataset  # noqa: E402
from pipelines.evaluate import evaluate  # noqa: E402
from pipelines.train import _device, train  # noqa: E402

OUT_DIR = ROOT_03 / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="跨工况故障泛化诊断")
    p.add_argument("--model", default="src_wdcnn", help=f"模型名，可选: {list_models()}")
    p.add_argument("--source", type=int, nargs="+", default=[0, 1, 2], help="源工况")
    p.add_argument("--target", type=int, nargs="+", default=[3], help="目标工况")
    p.add_argument("--protocol", choices=["source_target", "leave_one_out"],
                   default="source_target")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", action="store_true", help="保存模型权重")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(X, y, d, source, target, batch_size):
    src_mask = np.isin(d, source)
    tgt_mask = np.isin(d, target)
    src_loader = torch.utils.data.DataLoader(
        FaultDataset(X[src_mask], y[src_mask], d[src_mask]),
        batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = torch.utils.data.DataLoader(
        FaultDataset(X[tgt_mask], y[tgt_mask], d[tgt_mask]),
        batch_size=batch_size, shuffle=True, drop_last=True)
    return src_loader, tgt_loader


def run_once(cfg: Config, args, source, target, out_dir: Path):
    device = _device(cfg.train)
    X, y, d, class_names = build_dataset(CWRU_DIR, cfg.data)
    num_classes = len(class_names)
    print(f"样本数={len(X)} 类别数={num_classes} 类别={class_names}")
    print(f"源工况={source} 目标工况={target} 设备={device}")

    src_loader, tgt_loader = build_loaders(
        X, y, d, source, target, cfg.train.batch_size)
    model = create_model(args.model, num_classes=num_classes)
    print(f"模型={args.model} 参数量={sum(p.numel() for p in model.parameters()):,}")

    history = train(model, args.model, src_loader, tgt_loader, cfg.train)
    metrics = evaluate(model, tgt_loader, class_names, device)

    print("\n===== 目标工况评估结果 =====")
    print(metrics["report"])
    print(f"目标工况准确率: {metrics['acc']:.4f}")
    print(f"虚警率: {metrics['false_alarm_rate']:.4f}  漏警率: {metrics['miss_rate']:.4f}")

    result = {
        "model": args.model,
        "source": list(source),
        "target": list(target),
        "target_acc": metrics["acc"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "miss_rate": metrics["miss_rate"],
        "history": history,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    if args.save_model:
        torch.save({"state_dict": model.state_dict(),
                    "class_names": class_names},
                   out_dir / "checkpoint.pt")
        print(f"模型已保存: {out_dir / 'checkpoint.pt'}")

    return metrics["acc"]


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = Config()
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr

    if args.protocol == "leave_one_out":
        accs = []
        for t in cfg.data.domains:
            src = [d for d in cfg.data.domains if d != t]
            acc = run_once(cfg, args, src, [t], OUT_DIR / args.model / f"target_{t}")
            accs.append(acc)
        print(f"\n===== 留一工况平均准确率: {np.mean(accs):.4f} =====")
    else:
        run_once(cfg, args, args.source, args.target, OUT_DIR / args.model)


if __name__ == "__main__":
    main()
