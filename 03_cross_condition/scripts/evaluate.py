"""跨工况泛化诊断 —— 加载权重评估入口。

用法:
    python 03_cross_condition/scripts/evaluate.py --model dann_wdcnn \
        --checkpoint 03_cross_condition/outputs/dann_wdcnn/checkpoint.pt --target 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT_03 = HERE.parent
PROJ = ROOT_03.parent
sys.path.insert(0, str(ROOT_03))
sys.path.insert(0, str(PROJ))

from configs.config import CWRU_DIR, Config  # noqa: E402
from models.domain_adaptation import create_model  # noqa: E402
from pipelines.data import FaultDataset, build_dataset  # noqa: E402
from pipelines.evaluate import evaluate  # noqa: E402
from pipelines.train import _device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dann_wdcnn")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target", type=int, nargs="+", default=[3])
    ap.add_argument("--out", default=str(ROOT_03 / "outputs" / "eval_report.json"))
    args = ap.parse_args()

    cfg = Config()
    device = _device(cfg.train)
    X, y, d, class_names = build_dataset(CWRU_DIR, cfg.data)
    tgt_mask = np.isin(d, args.target)
    tgt_loader = torch.utils.data.DataLoader(
        FaultDataset(X[tgt_mask], y[tgt_mask], d[tgt_mask]),
        batch_size=cfg.train.batch_size, shuffle=False, drop_last=False)

    model = create_model(args.model, num_classes=len(class_names))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    normal_idx = class_names.index("N") if "N" in class_names else 0
    metrics = evaluate(model, tgt_loader, class_names, device,
                       normal_idx=normal_idx)

    result = {
        "class_names": class_names,
        "accuracy": metrics["acc"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "miss_rate": metrics["miss_rate"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "report": metrics["report"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"准确率: {metrics['acc']:.4f}")
    print(f"虚警率: {metrics['false_alarm_rate']:.4f}  漏警率: {metrics['miss_rate']:.4f}")
    print(metrics["report"])
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
