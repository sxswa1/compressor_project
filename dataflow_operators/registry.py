"""仓库算子目录登记；不是平台数据库注册或算子卡片。

算法源码按《新算法接入与算子封装规范》交给维护方后，由平台生成卡片、
写入on_run并联调。此清单只供仓库检索、构建和本地契约检查，不导入算法依赖。
按各自的输入输出契约登记；默认双端口只适用于原有过程诊断算子。
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

# 保留凌晨的交付集合，原有构建和验证不包含其他人的算子。
for entry in OPERATORS.values():
    entry.setdefault("owner", "凌晨")
    entry.setdefault("input_ports", ["df"])
    entry.setdefault("output_ports", {"out_1": "pd.DataFrame", "out_2": "pd.DataFrame"})
LINGCHEN_OPERATORS = dict(OPERATORS)

GAOHY_OPERATORS = {
    "gradient_boosting_regression": {
        "path": "predictive_analysis/gradient_boosting_regression.py", "model_required": True,
        "owner": "高淏宇", "description": "梯度提升特征表单目标回归",
        "requirements": ["numpy", "pandas", "scikit-learn"],
        "input_ports": ["df"], "output_ports": {"out_1": "pd.DataFrame"},
    },
    "lstm_regression": {
        "path": "predictive_analysis/lstm_regression.py", "model_required": True,
        "owner": "高淏宇", "description": "LSTM 历史窗口单目标回归",
        "requirements": ["numpy", "pandas", "torch"],
        "input_ports": ["df"], "output_ports": {"out_1": "pd.DataFrame"},
    },
    "compressor_early_warning": {
        "path": "health_assessment/compressor_early_warning.py", "model_required": True,
        "owner": "高淏宇", "description": "EWMA 压缩机超前预警",
        "requirements": ["numpy", "pandas", "scikit-learn"],
        "input_ports": ["df"], "output_ports": {"out_1": "pd.DataFrame"},
    },
    "compressor_risk_quantification": {
        "path": "health_assessment/compressor_risk_quantification.py", "model_required": True,
        "owner": "高淏宇", "description": "DI、四级风险、RUL 及区间",
        "requirements": ["numpy", "pandas", "scikit-learn", "torch"],
        "input_ports": ["df"], "output_ports": {"out_1": "pd.DataFrame"},
    },
}
OPERATORS.update(GAOHY_OPERATORS)

INPUT_PORTS = ["df"]
# 旧过程诊断接口的兼容默认值；实际端口以每个条目的 output_ports 为准。
OUTPUT_PORTS = {"out_1": "pd.DataFrame", "out_2": "pd.DataFrame"}
