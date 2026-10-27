# 结果复核

`di_verification.json`：DI 原方案与配对一致性训练的统一复核，873,420 条预测。

`rul_verification.json`：11 种已有 RUL 配置的同条件比较，856,800 条预测。

本目录保存既有实验的复核记录。本次算子封装复用了已训练的 EWMA、LSTM 和 ExtraTrees，没有重新训练。

指标和对应原始表见 `docs/高淏宇/`。算子一致性、独立加载和边界测试见 `01_early_warning/tests/test_dataflow_operators.py`。
