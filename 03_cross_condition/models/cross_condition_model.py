"""跨工况故障诊断骨干网络：1D-CNN、WDCNN、ResNet-1D、CNN-LSTM。

所有骨干网络输出固定维度特征 (FEAT_DIM=256)，便于与分类头 / 域自适应拼接。
"""
from __future__ import annotations

import torch
from torch import nn

FEAT_DIM = 256


def _conv_block(in_c, out_c, k, stride=1, pool=True):
    layers = [
        nn.Conv1d(in_c, out_c, k, stride, padding=k // 2),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    """简洁的 1D-CNN 骨干。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 64, 16, 2),
            _conv_block(64, 128, 5, 2),
            _conv_block(128, 256, 3, 2),
        )
        self.proj = nn.Sequential(nn.Linear(256, FEAT_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)              # (B, 256, L')
        x = x.mean(dim=2)                 # 全局平均池化
        return self.proj(x)               # (B, FEAT_DIM)


class WDCNN(nn.Module):
    """宽核深度卷积网络 (WDCNN)，旋转机械故障诊断经典骨干。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 64, 16, 32),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            _conv_block(16, 32, 3),
            _conv_block(32, 64, 3),
            _conv_block(64, 64, 3),
            _conv_block(64, 64, 3),
        )
        self.proj = nn.Sequential(nn.Linear(64, FEAT_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=2)
        return self.proj(x)


class ResBlock1D(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, k, padding=k // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, k, padding=k // 2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))


class ResNet1D(nn.Module):
    """1D ResNet 骨干。"""

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
        x = x.mean(dim=2)
        return self.proj(x)


class LSTMNet(nn.Module):
    """CNN-LSTM 骨干：卷积先压缩长序列，再进双向 LSTM 建模时序。

    原始波形长达 1024 点且为高频振荡，直接逐点送入 LSTM 梯度难以传播；
    先用宽核卷积 + 池化把长度压到 ~32，得到局部谱特征序列后再时序建模。
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 64, 16, 32),   # 1024 -> 65
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                # 65 -> 32
        )
        self.lstm = nn.LSTM(32, hidden, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, FEAT_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.cnn(x)                  # (B, 32, L')
        x = x.transpose(1, 2)            # (B, L', 32)
        out, _ = self.lstm(x)            # (B, L', 2h)
        x = out[:, -1, :]                # 取最后时刻
        return self.proj(x)


BACKBONES = {
    "cnn1d": CNN1D,
    "wdcnn": WDCNN,
    "resnet1d": ResNet1D,
    "lstm": LSTMNet,
}


def build_backbone(name: str) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"未知骨干网络: {name}，可选: {list(BACKBONES)}")
    return BACKBONES[name]()
