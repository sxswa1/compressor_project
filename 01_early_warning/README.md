# 超前预警与风险量化

负责人：高淏宇。基于 RCSD-1YD 的 25 个过程测点，完成预警、DI、四级风险和 RUL 的训练、预测与方法对照。

## 方法与结果

| 模块 | 本次主实验中表现较好的方法 | 9—12 月测试结果 |
|---|---|---|
| 预警 | EWMA | 至少提前 2 小时的预警覆盖率 91.19%，原始运行告警占比 7.38% |
| DI | LSTM | 三种子平均 MAE 0.13812 |
| 四级风险 | LSTM 的 DI＋固定阈值 | 平均准确率 53.89%，种子范围 39.78%—75.68% |
| RUL | 背景处理＋追加增强＋ExtraTrees | 场景平均 RMSE 10.824 小时；总体 RMSE÷平均参考 RUL 为 48.54% |

Ridge、梯度提升、CUSUM、合并预警及简单基线均保留。ExtraTrees 的轨迹平均 RMSE 更低，梯度提升的总体 RMSE 和长寿命段误差更低，完整对照见[实验报告](../docs/高淏宇/实验报告.md)和[结果表](../docs/高淏宇/实验结果表.xlsx)。

## 数据与环境

使用 Python 3.12，从仓库根目录安装依赖：

```powershell
python -m pip install -r 01_early_warning/requirements.txt
```

下载 [RCSD-1YD](https://doi.org/10.5281/zenodo.14866092) 中的 `Centrifugal compressor _2022.xlsx`。数据准备命令检查原文件 SHA-256，再构建增强数据。原始数据、生成数据、权重和逐点预测保存在本地。

## 复现

以下命令均从仓库根目录运行。将 `path/to/` 换成下载文件的位置。

```powershell
python 01_early_warning/scripts/research.py prepare --source "path/to/Centrifugal compressor _2022.xlsx"
python 01_early_warning/scripts/research.py train-warning
python 01_early_warning/scripts/research.py train-regression
python 01_early_warning/scripts/research.py seal
python 01_early_warning/scripts/research.py evaluate --split test
python 01_early_warning/scripts/compare_results.py
python -m unittest discover -s 01_early_warning/tests -v
```

`prepare` 包含逐场景重建和标签检查。训练保存全部对照与三个随机种子，`seal` 记录本次代码、输入和模型哈希，评价检查这些文件是否变化。已有结果不覆盖。完整训练和评价需要一段时间，日志在 `outputs/reproduction_logs/`。

实验代码位于 `experiments/rcsd_v1/`，按数据构建、预警、回归拟合和评价分开；运行入口统一为 `scripts/research.py`。原始结果归档在 `results/2026-10-25/`，重跑结果写入实验目录，不覆盖归档。公共 LSTM、梯度提升通过 `common_algorithms.models` 导入。

## 预测

```powershell
python 01_early_warning/scripts/predict_research.py --input input.csv --output prediction.csv --seed 20261024
```

输入为单设备、按时间递增的 CSV，包含 `Timestamp` 和 25 个测点，可选 `segment_id`。输出包括告警状态、新告警、DI、风险等级、RUL 及区间。缺测、时间缺口和历史不足保留无效标记。

默认使用 EWMA、LSTM DI 和 ExtraTrees RUL。可用 `--warning-method`、`--di-model`、`--rul-model`、`--rul-variant` 切换。种子需显式指定；单个种子的预测不等于表中的三种子平均结果。

## DataFlow 算子

算子目录同时登记了公共 [LSTM 和梯度提升回归入口](../common_algorithms/models/README.md#dataflow-独立回归算子)，供单独调用基础模型。它们与下面两个业务算子分别登记在 `dataflow_operators/registry.py`。

两个自包含算子位于 `dataflow_operators/health_assessment/`：

- `compressor_early_warning.py`：EWMA 预警。
- `compressor_risk_quantification.py`：LSTM DI、四级风险、ExtraTrees RUL 与区间。

均提供顶层 `execute(df, ...)`，返回 `{"out_1": result_df}`。单文件源码可直接加载，不依赖研究脚本；输入、单位、参数和输出说明写在文件内。平台卡片由维护方根据源码生成。

完成训练后，从仓库根目录导出已有模型：

```powershell
python 01_early_warning/scripts/export_dataflow_models.py --output 01_early_warning/outputs/dataflow_models
python -m unittest discover -s 01_early_warning/tests -v
```

部署时将导出目录放到算子执行环境，使用该环境的路径传参。目录内含三个种子的权重、预处理配置、背景模型、区间半径及文件哈希；权重不随 Git 提交。输入表的测点和单位必须与模型一致，可以通过 `sensor_map_json` 映射列名。

```python
from dataflow_operators.health_assessment.compressor_early_warning import execute as warn
from dataflow_operators.health_assessment.compressor_risk_quantification import execute as risk

warning_df = warn(df, model_path="models/rcsd/warning.pkl")["out_1"]
risk_df = risk(df, model_dir="models/rcsd", seed=20261024)["out_1"]
```

风险量化必须显式选择种子，单次预测不等于报告中的三种子平均指标。两个算子均按完整历史批量计算，调用之间不保存状态。各有效输出同时带有 `valid` 标记，历史不足时预测为空。完整测试还需准备公开数据和原训练模型；只有接口与阈值测试无需权重。

## 辅助实验

完成数据准备和 `seal` 后，可运行：

```powershell
python 01_early_warning/experiments/auxiliary/warning_fading_cusum_v1/experiment.py
python 01_early_warning/experiments/auxiliary/di_change_only_v1/experiment.py
```

分别比较 CUSUM 记忆衰减和 DI 差分特征。结果与主实验分开保存。

[数据构建说明](../docs/高淏宇/数据与接口.md) · [测试记录](../docs/高淏宇/测试记录.md)
