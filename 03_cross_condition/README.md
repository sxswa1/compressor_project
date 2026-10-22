# 03 跨工况泛化诊断（李佳耀）

压缩机在变转速、变负荷等**跨工况**条件下运行，同一故障的振动特征会随工况漂移，
导致固定工况训练的诊断模型在目标工况下准确率显著下降。本模块研究**跨工况故障泛化诊断**技术：
让模型在若干已知工况（源域）上训练后，能在未见工况（目标域）上保持 ≥80% 的准确率。

## 研究目标

- 故障类型 ≥ 8 类（CWRU 轴承 10 类：正常 + 内圈/滚动体/外圈 × 0.007/0.014/0.021 in）。
- 跨工况泛化准确率 ≥ 80%（留一工况协议 leave-one-out）。
- 输出可直接接入 DataFlow 平台的诊断算子。

## 技术路线

| 层次 | 方法 |
| ---- | ---- |
| 骨干网络 | CNN1D、WDCNN（宽核）、LSTM（CNN-LSTM） |
| 域自适应 | DANN（梯度反转）、DeepCORAL（协方差对齐）、DAN（多核 MMD） |
| 特征对齐 | CORAL 距离、MMD-RBF、域判别损失 |
| 物理机理 | 阶次域包络谱（BPFI=5.415fr、BPFO=3.585fr、BSF=2.357fr、FTF=0.398fr） |
| 传统基线 | 随机森林、梯度提升（手工程特征） |

## 目录结构

```
03_cross_condition/
├── data/         # 公开数据集（实际数据位于 <项目根>/data/CWRU，*.mat）
├── configs/      # config.py —— 数据切片 / 训练超参 / 跨工况协议
├── models/       # cross_condition_model.py（骨干）、domain_adaptation.py、
│                 # feature_alignment.py（对齐损失）、traditional.py（传统基线）
├── pipelines/    # data.py（数据加载）、train.py、evaluate.py
├── scripts/      # train.py / evaluate.py / compare.py（命令行入口）
├── tests/        # 特征、工况识别、模型 冒烟测试
├── outputs/      # 本地实验结果
└── README.md
```

## 数据说明

采用 **CWRU 轴承数据集**（公开），驱动端 12 kHz，10 类故障，4 种负荷
（0/1/2/3 HP → 1797/1772/1750/1730 RPM）作为 4 个工况（域）。
数据文件（`100.mat` ~ `234.mat`）放在 `<项目根>/data/CWRU/`，
由 `pipelines/data.py` 的 `FILE_MAP` 自动解析。

## 快速开始

```bash
# 在 03_cross_condition/ 目录下执行（脚本内部自动处理导入路径）
python scripts/train.py     # 训练，输出到 outputs/
python scripts/evaluate.py  # 评估，输出指标
python scripts/compare.py   # 多模型对比
```

## 跨工况协议

- `leave_one_out`（默认）：轮流留 1 个工况做测试，其余 3 个工况训练，取平均。
- `source_target`：固定源域 `(0,1,2)` → 目标域 `(3,)`，用于域自适应实验。

## 指标

- **准确率 Accuracy**：目标工况上的总体分类准确率。
- **类平均准确率**：各故障类别的平均召回，缓解类不平衡。
- **混淆矩阵**：定位易混故障对（如内圈 vs 外圈）。

## 结果记录

实验结果（模型权重、日志、指标 JSON）统一保存到 `outputs/`。
验收口径：留一工况协议下目标域平均准确率 ≥ 80%。
