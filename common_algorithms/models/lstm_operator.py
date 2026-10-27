"""高淏宇：公共 LSTM 单目标回归的 DataFlow 入口。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common_algorithms.models.lstm import LSTMRegressor


def execute(
    df: pd.DataFrame,
    model_path: str,
    schema_path: str,
    timestamp_column: str = 'Timestamp',
    segment_column: str = 'segment_id',
    feature_map_json: str = '',
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """使用已训练的单向 LSTM 对连续历史窗口进行单目标回归。

    df：一台设备或一条序列的过程数据长表，含时间列及模型规定的数值特征。
    model_path：LSTMRegressor.save 生成的可信 .pt 权重；本入口只预测，不训练。
    schema_path：与该权重配套的 JSON，必填 model_sha256、feature_columns（有序）、
    feature_units、window_points、sampling_minutes、target_name、target_unit。
    可选 running_column 和 running_bounds：整个窗口均须符合运行范围，边界包含。
    export_dataflow_models.py 会为现有 DI 权重生成 di_<seed>.schema.json。
    其他模型须按训练时的特征、单位和采样间隔提供同结构描述，禁止通过改描述复用不兼容权重。
    timestamp_column：有效、唯一、递增的时间列；不自动排序、补值或重新采样。
    segment_column：可选分段列；存在时切段重置历史，不允许空分段标识。
    feature_map_json：可选 JSON 字符串，映射“模型特征名:输入列名”，不改变单位。
    单位按 schema 的 feature_units 输入；只映射列名不会进行单位转换。
    out_1=result_df：Timestamp、prediction、valid，逐输入行对应；prediction 单位为 target_unit。
    不裁剪预测范围、不进行风险分级。历史不足、非有限特征或停机时 prediction 为空。
    空输入须含必需列，校验资源后返回有结构的空表；缺列、重复时间、错误模型配对抛异常。
    额外标签列不进入特征；每次传入完整历史，调用之间不保存隐状态。不修改输入表。
    依赖 numpy、pandas、torch（已验证 2.5.1，CPU）。
    """
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError('df 必须是列名唯一的 DataFrame')
    schema = json.loads(Path(schema_path).read_text(encoding='utf-8'))
    path = Path(model_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != schema.get('model_sha256'):
        raise ValueError('模型与 schema 不匹配')
    columns = schema.get('feature_columns')
    units = schema.get('feature_units')
    if not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('schema 必须提供非空、不重复的 feature_columns')
    if not isinstance(units, dict) or set(units) != set(columns) or any(not isinstance(v,str) or not v for v in units.values()):
        raise ValueError('schema 必须声明每个特征的单位')
    if any(not isinstance(schema.get(k), str) or not schema[k] for k in ['target_name', 'target_unit']):
        raise ValueError('schema 必须声明目标名称与单位')
    minutes = schema.get('sampling_minutes')
    if isinstance(minutes, bool) or not isinstance(minutes, (int,float)) or not np.isfinite(minutes) or minutes <= 0:
        raise ValueError('sampling_minutes 必须为正数')
    window = schema.get('window_points')
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('window_points 必须为正整数')
    model = LSTMRegressor.load(path)
    if (model.window_, model.n_features_in_) != (window, len(columns)):
        raise ValueError('schema 的窗口和特征数量与权重不一致')
    mapping = json.loads(feature_map_json) if feature_map_json else {}
    if not isinstance(mapping, dict) or set(mapping)-set(columns):
        raise ValueError('feature_map_json 只能映射模型中的特征名')
    actual = [mapping.get(c,c) for c in columns]
    if any(not isinstance(c,str) for c in actual) or len(set(actual)) != len(actual):
        raise ValueError('特征映射应使用不同的字符串列名')
    if timestamp_column in actual or (segment_column and segment_column in actual):
        raise ValueError('时间和分段列不能同时作为特征列')
    missing = set([timestamp_column,*actual])-set(df.columns)
    if missing:
        raise ValueError(f'缺少输入列：{sorted(missing)}')
    if len(df) and pd.api.types.is_numeric_dtype(df[timestamp_column]):
        raise ValueError('数值时间戳须先按明确单位转为日期时间')
    times = pd.DatetimeIndex(pd.to_datetime(df[timestamp_column], errors='raise'))
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('时间必须有效、唯一且递增')
    values = df[actual].to_numpy(dtype=np.float32, copy=True)
    valid_row = np.isfinite(values).all(axis=1)
    running = schema.get('running_column')
    if running is not None:
        bounds = schema.get('running_bounds')
        if running not in columns or not isinstance(bounds,list) or len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:
            raise ValueError('运行范围必须对应模型特征且为有限有序上下界')
        # 用原始精度检查范围，与训练数据的窗口筛选保持一致。
        valid_row &= df[actual[columns.index(running)]].astype(float).between(*bounds).to_numpy()
    boundary = np.ones(len(df), dtype=bool)
    boundary[1:] = np.diff(times.asi8) != pd.Timedelta(minutes=minutes).value
    if segment_column and segment_column in df:
        segments = df[segment_column].to_numpy()
        if pd.isna(segments).any() or any(isinstance(s,str) and not s.strip() for s in segments):
            raise ValueError('分段标识不能为空')
        boundary[1:] |= segments[1:] != segments[:-1]
    ends, count = [], 0
    for row, finite in enumerate(valid_row):
        if boundary[row]:
            count = 0
        count = count+1 if finite else 0
        if count >= window:
            ends.append(row)
    result_df = pd.DataFrame({'Timestamp': times, 'prediction': np.full(len(df),np.nan), 'valid': False})
    # 逐批组窗，避免把长时间序列的全部重叠窗口放入内存。
    for start in range(0,len(ends),256):
        ids = ends[start:start+256]
        windows = np.stack([values[i-window+1:i+1] for i in ids])
        predictions = model.predict(windows)
        if not np.isfinite(predictions).all():
            raise ValueError('模型预测出现非有限值')
        result_df.loc[ids,'prediction'] = predictions
        result_df.loc[ids,'valid'] = True
    return {'out_1': result_df}
