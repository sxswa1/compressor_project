# DataFlow 算子目录

以下为凌晨负责的小样本过程诊断算子，登记源为 [registry.py](registry.py)。它是仓库目录清单，不是平台数据库注册结果。其他同学的已有算子保留原路径。

每个文件为 UTF-8，自包含算法及顶层 `execute(df, 显式业务参数, **kwargs)`，统一返回 `{"out_1": result_df, "out_2": info_df}`。输出值均为 `pd.DataFrame`，空表保持输出列结构，非法输入抛出异常。依赖须预先安装；模型由部署环境显式挂载。

| 算子名 | 单文件源码 | 功能 | 模型资源 |
|---|---|---|---|
| anomaly_detection | [data_processing/anomaly_detection.py](data_processing/anomaly_detection.py) | 异常点标记、可选离线去噪 | 无 |
| missing_value | [data_processing/missing_value.py](data_processing/missing_value.py) | 有界补缺、一致性检查 | 无 |
| process_timeseries | [data_processing/process_timeseries.py](data_processing/process_timeseries.py) | 分组时钟对齐 | 无 |
| process_window_table | [data_processing/process_window_table.py](data_processing/process_window_table.py) | 连续窗口及拒绝原因 | 无 |
| process_features | [feature_extraction/process_features.py](feature_extraction/process_features.py) | 多测点统计与趋势特征 | 无 |
| process_random_forest | [fault_diagnosis/process_random_forest.py](fault_diagnosis/process_random_forest.py) | RF逐窗诊断 | 可信.joblib |
| process_cnn_diagnosis | [fault_diagnosis/process_cnn_diagnosis.py](fault_diagnosis/process_cnn_diagnosis.py) | CNN逐窗诊断 | kind=cnn 的.pt |
| process_meta_learning | [fault_diagnosis/process_meta_learning.py](fault_diagnosis/process_meta_learning.py) | 元学习编码器＋适配原型 | kind=prototype 的.pt |
| process_transfer_learning | [fault_diagnosis/process_transfer_learning.py](fault_diagnosis/process_transfer_learning.py) | 预训练后微调诊断 | kind=transfer 的.pt |
| process_seed_ensemble | [fault_diagnosis/process_seed_ensemble.py](fault_diagnosis/process_seed_ensemble.py) | 同方法三种子等权集成 | kind=process_seed_ensemble 的.pt |

公共处理仅依赖NumPy/Pandas；RF另需scikit-learn/joblib；神经算子另需PyTorch。入口docstring说明各参数、单位、输出列、默认值和限制，供平台自动生成卡片，无需人工另交卡片JSON/YAML。

重建源码：`python tools/build_rcsd_operators.py`。修改 `common_algorithms/` 共享实现后重建，避免只改生成文件。

本地接入检查：`python tools/verify_rcsd_handover.py`。检查通过源码文本执行、不给 `__file__`，模拟执行器加载 `on_run` 并寻找 `execute`；不等于真实UDO环境联调。平台维护方后续仍需生成卡片、配置输入与业务参数、登记 `out_1/out_2`，挂载模型并验证执行。

调用和交接见 [凌晨算子说明](../docs/凌晨/算子说明.md)。
