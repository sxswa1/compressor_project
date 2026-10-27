# 公共回归模型

负责人：高淏宇。提供单目标回归，数据划分、特征构建和任务标签由研究目录负责。

本页对应 LSTM 与梯度提升回归。随机森林分类模块及 `process_random_forest` 算子由凌晨交付，见[算子目录](../../dataflow_operators/README.md)。

| 模块 | 输入 | 输出 |
|---|---|---|
| `lstm.LSTMRegressor` | `[样本, 时间, 特征]` | 每个窗口一个连续预测值 |
| `gradient_boosting.GradientBoostingRegressor` | `[样本, 特征]` | 每个样本一个连续预测值 |

共同接口：`fit(X, y)`、`predict(X)`、`save(path)`、`load(path)`。输入需为有限数值。LSTM 使用训练样本估计标准化参数，保存时一并保存；梯度提升使用 sklearn 的直方图梯度提升回归。

```python
from common_algorithms.models.lstm import LSTMRegressor
from common_algorithms.models.gradient_boosting import GradientBoostingRegressor

lstm = LSTMRegressor(hidden_size=24, epochs=6, random_state=20261024)
boosting = GradientBoostingRegressor(max_iter=100, random_state=20261024)
# lstm.fit(X_windows, y)
# boosting.fit(X_features, y)
```

参数及输入检查见各类定义。LSTM 默认单层、24 个隐藏单元、6 轮、batch 128、学习率 0.001；研究实验的轮数选择由任务代码完成。梯度提升默认 100 轮、学习率 0.05、15 叶节点、最少 20 样本/叶、L2=1。

测试：`python -m unittest discover -s 01_early_warning/tests -p test_public_baselines.py -v`。

研究训练、评价和预测入口见 [01_early_warning](../../01_early_warning/README.md)。权重由训练命令生成，加载时使用可信文件和兼容依赖。

## DataFlow 独立回归算子

`dataflow_operators/predictive_analysis/lstm_regression.py` 和 `gradient_boosting_regression.py` 已加入算子目录。两个文件均可单独加载，入口为 `execute(df, model_path, schema_path, ...)`，返回 `{"out_1": result_df}`。这两个入口只负责预测，训练仍使用上面的 `fit`、`save` 接口。

LSTM 接收按时间递增的单设备长表，逐历史窗口预测。梯度提升接收已构建好的特征表，每行预测一个连续值。两者输出原始回归值，不做 DI 裁剪或风险分级；这些业务规则由风险量化算子负责。

完成研究模型训练后，从仓库根目录导出：

```powershell
python 01_early_warning/scripts/export_dataflow_models.py --output 01_early_warning/outputs/dataflow_models_registered
python -m unittest discover -s 01_early_warning/tests -v
```

导出包括三个种子的 LSTM DI 权重及 `di_<seed>.schema.json`，以及梯度提升 DI 对照模型 `gradient_boosting_di.pkl` 和配套描述。已有非空导出目录不会覆盖，应另选目录。

```python
from dataflow_operators.predictive_analysis.lstm_regression import execute as lstm
from dataflow_operators.predictive_analysis.gradient_boosting_regression import execute as boosting

lstm_result = lstm(df, model_path="models/di_20261024.pt",
                   schema_path="models/di_20261024.schema.json")["out_1"]
gb_result = boosting(feature_df, model_path="models/gradient_boosting_di.pkl",
                     schema_path="models/gradient_boosting_di.schema.json")["out_1"]
```

现有梯度提升 DI 模型使用 125 维特征：对 96 点窗口按“当前值、最近 24 点均值、当前值减最近 24 点首值、96 点均值、当前值减窗口首值”排列，每组内部沿用 25 个测点的训练顺序。使用 `pipelines.baseline_data.windows` 和 `tabular_features` 构建，列名按导出的 `feature_columns` 设置；不能直接把 25 列原始测点当作输入。窗口采用 float32，缺测、停机、时间缺口和分段边界不跨越。

其他训练模型的描述文件必须记录 `model_sha256`、有序 `feature_columns`、逐列 `feature_units`、`target_name`、`target_unit`；LSTM 还须记录 `window_points` 和 `sampling_minutes`。模型描述来自实际训练配置，不能通过修改列名或单位声明替代真实的数据转换。

两个生成算子的公共实现分别为 `lstm_operator.py`、`gradient_boosting_operator.py`。修改后运行 `python tools/build_regression_operators.py`，保持单文件源码与公共模块一致。
