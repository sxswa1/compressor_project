# 公共回归模型

负责人：高淏宇。提供单目标回归，数据划分、特征构建和任务标签由研究目录负责。

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
