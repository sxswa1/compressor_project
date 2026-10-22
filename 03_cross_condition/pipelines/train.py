"""训练流程：源域训练 / 域自适应训练（DANN、DeepCORAL、DAN）。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from configs.config import TrainConfig
from models.feature_alignment import coral_loss, mmd_rbf


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _class_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(1)
    return (pred == y).float().mean().item()


def train(model: nn.Module, model_name: str,
          source_loader: DataLoader,
          target_loader: DataLoader | None,
          cfg: TrainConfig) -> dict:
    """训练并返回 history。

    - src_* 模型：仅用源域有标签数据训练（目标域仅用于评估）。
    - dann_* / coral_* / dan_* 模型：额外利用目标域无标签数据做域对齐。
    """
    device = _device(cfg)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    adapt = model_name.split("_", 1)[0] in ("dann", "coral", "dan")
    steps_per_epoch = len(source_loader)
    total_steps = steps_per_epoch * cfg.epochs

    history = {"loss": [], "acc": [], "target_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        tgt_iter = iter(target_loader) if target_loader is not None else None
        running_loss, running_acc, n = 0.0, 0.0, 0

        for s in source_loader:
            x_s = s["x"].to(device)
            y_s = s["y"].to(device)

            x_t = None
            if tgt_iter is not None:
                try:
                    t = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(target_loader)
                    t = next(tgt_iter)
                x_t = t["x"].to(device)

            opt.zero_grad()
            out = model(x_s)
            loss = F.cross_entropy(out["logits"], y_s)
            acc = _class_accuracy(out["logits"], y_s)

            if adapt and x_t is not None:
                global_step = epoch * steps_per_epoch + n
                p = global_step / max(total_steps - 1, 1)

                if model_name.startswith("dann"):
                    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
                    out_s = model(x_s, alpha=alpha)
                    out_t = model(x_t, alpha=alpha)
                    d_in = torch.cat([out_s["d_logits"], out_t["d_logits"]], 0)
                    d_y = torch.cat([
                        torch.zeros(x_s.size(0), dtype=torch.long),
                        torch.ones(x_t.size(0), dtype=torch.long),
                    ]).to(device)
                    loss = loss + F.cross_entropy(d_in, d_y)
                    out = out_s
                else:
                    out_t = model(x_t)
                    if model_name.startswith("coral"):
                        loss = loss + coral_loss(out["feats"], out_t["feats"])
                    else:  # dan
                        loss = loss + mmd_rbf(out["feats"], out_t["feats"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running_loss += loss.item()
            running_acc += acc
            n += 1

        hist_acc = running_acc / n
        history["loss"].append(running_loss / n)
        history["acc"].append(hist_acc)

        tgt_acc = None
        if target_loader is not None:
            from pipelines.evaluate import evaluate_loader
            tgt_acc = evaluate_loader(model, target_loader, device)["acc"]
            history["target_acc"].append(tgt_acc)

        if target_loader is not None:
            print(f"[epoch {epoch+1}/{cfg.epochs}] loss={running_loss/n:.4f} "
                  f"src_acc={hist_acc:.4f} tgt_acc={tgt_acc:.4f}")
        else:
            print(f"[epoch {epoch+1}/{cfg.epochs}] loss={running_loss/n:.4f} "
                  f"src_acc={hist_acc:.4f}")

    model.eval()
    return history
