"""凌晨负责的过程数据算子目录登记；不是平台数据库注册或算子卡片。

算法源码按《新算法接入与算子封装规范》交给维护方后，由平台生成卡片、
写入on_run并联调。此清单只供仓库检索、构建和本地契约检查，不导入算法依赖。
其他同学已有算子路径不变，本清单不接管其登记方式。
"""

OPERATORS = {
    "anomaly_detection": {
        "path": "data_processing/anomaly_detection.py", "model_required": False,
        "description": "异常点标记及可选离线去噪", "requirements": ["numpy", "pandas"],
    },
    "missing_value": {
        "path": "data_processing/missing_value.py", "model_required": False,
        "description": "有界缺失补全与一致性检查", "requirements": ["numpy", "pandas"],
    },
    "process_timeseries": {
        "path": "data_processing/process_timeseries.py", "model_required": False,
        "description": "分组时钟对齐", "requirements": ["numpy", "pandas"],
    },
    "process_window_table": {
        "path": "data_processing/process_window_table.py", "model_required": False,
        "description": "连续长表切窗与拒绝原因", "requirements": ["numpy", "pandas"],
    },
    "process_features": {
        "path": "feature_extraction/process_features.py", "model_required": False,
        "description": "多测点统计及趋势特征", "requirements": ["numpy", "pandas"],
    },
    "process_random_forest": {
        "path": "fault_diagnosis/process_random_forest.py", "model_required": True,
        "description": "随机森林过程诊断", "requirements": ["numpy", "pandas", "scikit-learn", "joblib"],
    },
    "process_cnn_diagnosis": {
        "path": "fault_diagnosis/process_cnn_diagnosis.py", "model_required": True,
        "description": "CNN过程诊断", "requirements": ["numpy", "pandas", "torch"],
    },
    "process_meta_learning": {
        "path": "fault_diagnosis/process_meta_learning.py", "model_required": True,
        "description": "元学习编码器与适配原型诊断", "requirements": ["numpy", "pandas", "torch"],
    },
    "process_transfer_learning": {
        "path": "fault_diagnosis/process_transfer_learning.py", "model_required": True,
        "description": "预训练后微调的迁移诊断", "requirements": ["numpy", "pandas", "torch"],
    },
    "process_seed_ensemble": {
        "path": "fault_diagnosis/process_seed_ensemble.py", "model_required": True,
        "description": "三个训练种子的等权集成", "requirements": ["numpy", "pandas", "torch"],
    },
}

INPUT_PORTS = ["df"]
OUTPUT_PORTS = {"out_1": "pd.DataFrame", "out_2": "pd.DataFrame"}
