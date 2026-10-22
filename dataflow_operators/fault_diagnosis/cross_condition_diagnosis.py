"""跨工况故障泛化诊断算子（自包含）。

将上游振动信号切片后，通过深度学习模型（域自适应 / 普通骨干）或物理机理规则推理，
输出逐切片诊断结果与类别统计。支持模型按名切换：src_/dann_/coral_/dan_ × 骨干。

输入: df —— 振动信号表（含信号列，可选转速列）
参数:
    mode        诊断模式: deep(深度学习, 默认) / physics(物理机理规则)
    model_name  deep 模式下的模型名，如 src_wdcnn / dann_wdcnn / coral_wdcnn / dan_wdcnn
    checkpoint  deep 模式下的权重路径（必填；由上游训练产出并挂载）
    sample_rate 采样频率(Hz)
    window_len  切片点数
    overlap     滑窗重叠率
    signal_col  信号列名（缺省自动取第一个数值列）
    rpm_col     转速列名（physics 模式用于阶次域特征）
输出:
    out_1 —— 逐切片诊断结果（sample_id / pred_class / class_name / confidence / model）
    out_2 —— 类别样本数统计
依赖: numpy, pandas, torch, scipy（无本地项目文件依赖）
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CrossConditionDiagnosis")

FEAT_DIM = 256

# 类别名（CWRU 10 类）
CLASS_NAMES_ZH = {
    "N": "正常", "IR007": "内圈故障-0.007in", "IR014": "内圈故障-0.014in",
    "IR021": "内圈故障-0.021in", "B007": "滚动体故障-0.007in",
    "B014": "滚动体故障-0.014in", "B021": "滚动体故障-0.021in",
    "OR007": "外圈故障-0.007in", "OR014": "外圈故障-0.014in",
    "OR021": "外圈故障-0.021in",
}


# ===================== 骨干网络 =====================
def _conv_block(in_c, out_c, k, stride=1, pool=True):
    layers = [nn.Conv1d(in_c, out_c, k, stride, padding=k // 2),
              nn.BatchNorm1d(out_c), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 64, 16, 2), _conv_block(64, 128, 5, 2),
            _conv_block(128, 256, 3, 2))
        self.proj = nn.Sequential(nn.Linear(256, FEAT_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(self.features(x).mean(dim=2))


class WDCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 64, 16, 32), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), _conv_block(16, 32, 3), _conv_block(32, 64, 3),
            _conv_block(64, 64, 3), _conv_block(64, 64, 3))
        self.proj = nn.Sequential(nn.Linear(64, FEAT_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(self.features(x).mean(dim=2))


class ResBlock1D(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, k, padding=k // 2),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, k, padding=k // 2),
            nn.BatchNorm1d(channels))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = _conv_block(1, 64, 7, 2)
        self.layer1 = nn.Sequential(ResBlock1D(64), ResBlock1D(64))
        self.layer2 = nn.Sequential(
            _conv_block(64, 128, 3, 2, pool=False), ResBlock1D(128))
        self.layer3 = nn.Sequential(
            _conv_block(128, 256, 3, 2, pool=False), ResBlock1D(256))
        self.proj = nn.Sequential(nn.Linear(256, FEAT_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.proj(x.mean(dim=2))


class LSTMNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 64, 16, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2))
        self.lstm = nn.LSTM(32, hidden, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, FEAT_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])


BACKBONES = {"cnn1d": CNN1D, "wdcnn": WDCNN, "resnet1d": ResNet1D, "lstm": LSTMNet}


# ===================== 域自适应 =====================
class GradReverse(Function):
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
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = BACKBONES[backbone_name]()
        self.classifier = nn.Linear(FEAT_DIM, num_classes)

    def forward(self, x, alpha=None):
        feats = self.backbone(x)
        return {"logits": self.classifier(feats), "feats": feats}


class DANN(FaultModel):
    def __init__(self, backbone_name, num_classes):
        super().__init__(backbone_name, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(FEAT_DIM, 256), nn.ReLU(inplace=True), nn.Linear(256, 2))

    def forward(self, x, alpha=None):
        feats = self.backbone(x)
        alpha = 1.0 if alpha is None else alpha
        d_logits = self.domain_head(_grl(feats, alpha))
        return {"logits": self.classifier(feats), "feats": feats, "d_logits": d_logits}


class DeepCORAL(FaultModel):
    pass


class DAN(FaultModel):
    pass


_MODELS = {}
for _name in BACKBONES:
    _MODELS[f"src_{_name}"] = FaultModel
for _name in BACKBONES:
    _MODELS[f"dann_{_name}"] = DANN
    _MODELS[f"coral_{_name}"] = DeepCORAL
    _MODELS[f"dan_{_name}"] = DAN


def create_model(name: str, num_classes: int) -> nn.Module:
    if name not in _MODELS:
        raise ValueError(f"未知模型: {name}，可选: {sorted(_MODELS)}")
    backbone = name.split("_", 1)[1] if "_" in name else "wdcnn"
    return _MODELS[name](backbone_name=backbone, num_classes=num_classes)


# ===================== 物理机理规则 =====================
FAULT_ORDERS = {"BPFI": 5.415, "BPFO": 3.585, "BSF": 2.357, "FTF": 0.3983}
ORDER_LABELS = list(FAULT_ORDERS.keys())
PHYSICS_ZH = {"BPFI": "内圈故障", "BPFO": "外圈故障",
              "BSF": "滚动体故障", "FTF": "保持架故障"}


def _physics_features(x: np.ndarray, sample_rate: int, rpm: float) -> np.ndarray:
    fr = rpm / 60.0
    feats = []
    for row in np.atleast_2d(x):
        analytic = sp_signal.hilbert(row)
        env = np.abs(analytic) - np.abs(analytic).mean()
        spec = np.abs(rfft(env))
        freqs = rfftfreq(len(row), d=1.0 / sample_rate)
        total = spec.sum() + 1e-8
        row_feats = []
        for order in FAULT_ORDERS.values():
            f = order * fr
            lo, hi = f * 0.95, f * 1.05
            mask = (freqs >= lo) & (freqs <= hi)
            row_feats.append(spec[mask].max() / total if mask.any() else 0.0)
        feats.append(row_feats)
    return np.asarray(feats, dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ===================== 工具 =====================
class FaultDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx]}


def _extract_signal(df: pd.DataFrame, signal_col: str, rpm_col: str):
    col = signal_col if signal_col in df.columns else next(
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    vals = df[col].to_numpy()
    if len(vals) == 1 and hasattr(vals[0], "__len__"):
        x = np.asarray(vals[0], dtype=np.float64).ravel()
    else:
        x = np.asarray(vals, dtype=np.float64).ravel()
    rpm = None
    if rpm_col in df.columns:
        r = np.asarray(df[rpm_col].to_numpy(), dtype=np.float64)
        if len(r) == 1 and hasattr(r[0], "__len__"):
            r = np.asarray(r[0], dtype=np.float64).ravel()
        rpm = float(r.mean())
    return x, rpm


def _sliding_windows(signal: np.ndarray, window_len: int, overlap: float) -> np.ndarray:
    step = max(int(window_len * (1.0 - overlap)), 1)
    n = (len(signal) - window_len) // step + 1
    if n <= 0:
        return np.empty((0, window_len))
    idx = np.arange(window_len)[None, :] + step * np.arange(n)[:, None]
    return signal[idx]


def _normalize(samples: np.ndarray) -> np.ndarray:
    mu = samples.mean(axis=1, keepdims=True)
    std = samples.std(axis=1, keepdims=True) + 1e-8
    return (samples - mu) / std


# ===================== 标准入口 =====================
def execute(df: pd.DataFrame, mode: str = "deep", model_name: str = "src_wdcnn",
            checkpoint: str = "", sample_rate: int = 12000, window_len: int = 1024,
            overlap: float = 0.5, signal_col: str = "signal", rpm_col: str = "rpm",
            **kwargs) -> Dict[str, pd.DataFrame]:
    node = "CrossConditionDiagnosis"
    if df is None or df.empty:
        logger.warning(f"[{node}] DataFrame 为空，跳过诊断。")
        return {"out_1": pd.DataFrame([{"status": "skipped", "diagnosed_samples": 0}])}

    x, rpm = _extract_signal(df, signal_col, rpm_col)
    windows = _sliding_windows(x, window_len, overlap)
    if len(windows) == 0:
        return {"out_1": pd.DataFrame([{
            "status": "insufficient_data", "signal_length": int(len(x)),
            "window_len": int(window_len)}])}

    if mode == "deep":
        if not checkpoint:
            raise FileNotFoundError("deep 模式需要 checkpoint 参数（模型权重路径）。")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        class_names = list(ckpt.get("class_names") or CLASS_NAMES_ZH.keys())
        model = create_model(model_name, num_classes=len(class_names))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        ds = FaultDataset(_normalize(windows).astype(np.float32))
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        probs = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["x"])["logits"]
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        idx = probs.argmax(axis=1)
        conf = probs[np.arange(len(idx)), idx]
        pred_code = [class_names[i] for i in idx]
        pred_zh = [CLASS_NAMES_ZH.get(c, c) for c in pred_code]
        used_model = model_name
    elif mode == "physics":
        rpm_val = rpm if rpm is not None else 1797.0
        feats = _physics_features(windows, sample_rate, rpm_val)
        p = _softmax(feats)
        idx = p.argmax(axis=1)
        conf = p[np.arange(len(idx)), idx]
        pred_code = [ORDER_LABELS[i] for i in idx]
        pred_zh = [PHYSICS_ZH[c] for c in pred_code]
        used_model = "physics_rule"
    else:
        raise ValueError(f"mode 无效: '{mode}'，可选: deep / physics")

    n = len(pred_code)
    result = pd.DataFrame({
        "sample_id": np.arange(1, n + 1),
        "pred_class": pred_code,
        "class_name": pred_zh,
        "confidence": np.round(conf, 4),
    })
    if rpm is not None:
        result["rpm"] = rpm
    result["model"] = used_model

    counts = result["class_name"].value_counts()
    summary = pd.DataFrame([{
        "class_name": cls, "count": int(counts[cls]),
        "ratio": round(float(counts[cls] / n), 4),
    } for cls in counts.index])
    summary["model"] = used_model

    logger.info(f"[{node}] 诊断完成: 样本={n}, 模型={used_model}, "
                f"多数类={counts.index[0]}({counts.iloc[0]})")
    return {"out_1": result, "out_2": summary}


if __name__ == "__main__":
    # 离线冒烟测试：合成一段内圈故障主导信号，走物理机理规则模式
    sr = 12000
    t = np.arange(0, 1.0, 1.0 / sr)
    sig = np.sin(2 * np.pi * 162 * t) + 0.3 * np.sin(2 * np.pi * 200 * t) + \
        0.05 * np.random.randn(len(t))
    df = pd.DataFrame({"signal": [sig], "rpm": [1797.0]})
    out = execute(df, mode="physics")
    print(out["out_1"].head())
    print(out["out_2"])
