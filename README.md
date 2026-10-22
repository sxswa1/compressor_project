# 压缩机项目 —— 算法开发（李佳耀部分）

长庆油田压缩机在线监测与智能诊断 · 跨工况故障泛化诊断技术。

本仓库按《压缩机项目算法开发分工与实施方案》第四部分的目录结构组织：
**公共算法库 + 三个独立研究子项目 + DataFlow 算子交付**。
李佳耀负责 **③ 跨工况泛化诊断（03_cross_condition）**，并贡献
工况识别、时/频/包络谱/小波/EMD 等可复用特征算法到公共算法库与算子交付目录。

## 目录结构

```
compressor_project/
├── README.md
├── requirements.txt
├── common_algorithms/          # 公共算法库（多研究复用，李佳耀部分）
│   ├── data_processing/
│   │   └── operating_condition.py   # 工况识别与片段切分（规则版）
│   └── feature_extraction/
│       ├── time_domain.py           # 时域统计（13 维）
│       ├── frequency_domain.py      # 频域与 FFT（7 维）
│       ├── envelope_spectrum.py     # 包络谱（3 维）
│       ├── wavelet.py               # 小波变换 db4（5 维）
│       └── emd.py                   # EMD 能量占比（4 维）
├── 03_cross_condition/        # ③ 跨工况泛化诊断（李佳耀）
│   ├── data/                  # 公开数据集（实际位于 data/CWRU/*.mat）
│   ├── configs/               # config.py —— 数据切片 / 训练超参 / 跨工况协议
│   ├── models/                # 骨干、域自适应、特征对齐、传统基线
│   ├── pipelines/             # 数据加载、训练、评估
│   ├── scripts/               # train.py / evaluate.py / compare.py
│   ├── tests/                 # 特征、工况识别、模型冒烟测试
│   ├── outputs/               # 实验结果、模型权重
│   └── README.md              # 模块说明
├── dataflow_operators/        # DataFlow 算子交付（自包含单文件）
│   ├── data_processing/
│   │   └── operating_condition.py   # 工况识别算子
│   ├── feature_extraction/
│   │   ├── time_domain.py / frequency_domain.py / envelope_spectrum.py
│   │   └── wavelet.py / emd.py
│   └── fault_diagnosis/
│       └── cross_condition_diagnosis.py   # 跨工况诊断算子（deep/physics）
└── docs/                      # 代码说明、技术报告、测试报告等
```

## 方法概览

| 类别 | 方法 | 说明 |
|------|------|------|
| 信号特征 | 时域统计、频域、包络谱、小波、EMD | 手工特征（公共算法库，可复用） |
| 工况识别 | 负荷标签 + 转速换算 + 频谱转频估计 | 规则版工况标注（0/1/2/3 HP） |
| 深度学习骨干 | CNN1D、WDCNN、ResNet-1D、CNN-LSTM | 输出 256 维特征 |
| 域自适应 | DANN、DeepCORAL、DAN(MMD) | 源工况→目标工况迁移（核心） |
| 物理机理 | 阶次域包络谱 BPFI/BPFO/BSF/FTF | 对转速/负荷变化稳定的规则诊断 |
| 传统基线 | 随机森林、梯度提升 | 手工特征上的对照 |

## 快速开始

```bash
# 训练（源工况 0/1/2 -> 目标工况 3）
python 03_cross_condition/scripts/train.py --model src_wdcnn --source 0 1 2 --target 3

# 域自适应模型
python 03_cross_condition/scripts/train.py --model dann_wdcnn --source 0 1 2 --target 3

# 留一工况交叉验证（跨工况泛化验收指标）
python 03_cross_condition/scripts/train.py --model src_wdcnn --protocol leave_one_out

# 评估 / 多模型对比
python 03_cross_condition/scripts/evaluate.py
python 03_cross_condition/scripts/compare.py
```

## 算子（DataFlow 交付）

每个算子为**自包含单文件**，含标准入口 `execute(df, ...) -> Dict[str, DataFrame]`，
不依赖本地脚本、绝对路径或 Notebook 状态，可直接被平台导入注册：

```python
# 工况识别
from dataflow_operators.data_processing.operating_condition import execute
out = execute(df, rpm_col="rpm")          # out_1 逐片段工况 / out_2 工况分布

# 跨工况诊断（物理规则，无需权重）
from dataflow_operators.fault_diagnosis.cross_condition_diagnosis import execute
out = execute(df, mode="physics", rpm_col="rpm")

# 跨工况诊断（深度学习，需训练产出的 checkpoint.pt）
out = execute(df, mode="deep", model_name="src_wdcnn", checkpoint=".../checkpoint.pt")
```

## 验收口径

- 故障类型 ≥ 8 类（CWRU 10 类）。
- 跨工况泛化准确率 ≥ 80%（留一工况协议下目标域平均准确率）。

## 依赖

见 `requirements.txt`。核心：`torch`、`numpy`、`scipy`、`scikit-learn`、`pandas`、
`PyWavelets`、`PyEMD`（可选）。
