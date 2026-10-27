"""高淏宇：压缩机 EWMA 超前预警算子。

依赖：numpy、pandas、scikit-learn（训练版本 1.5.2）。
model_path 是 export_dataflow_models.py 导出的 warning.pkl，由维护方部署到执行环境。
模型文件只从可信训练产物加载。代码可单文件加载，不导入项目内模块。
"""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd


def _input(df, sensors, timestamp_column, segment_column, sensor_map_json):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df 必须是 pandas.DataFrame')
    if df.empty:
        raise ValueError('df 不能为空；请传入一台设备的连续历史数据')
    if not df.columns.is_unique:
        raise ValueError('输入表不能包含重复列名')
    mapping = json.loads(sensor_map_json) if sensor_map_json else {}
    if not isinstance(mapping, dict) or set(mapping) - set(sensors):
        raise ValueError('sensor_map_json 必须是模型测点名到输入列名的 JSON 对象')
    columns = [mapping.get(s, s) for s in sensors]
    if any(not isinstance(c, str) for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('测点映射必须使用不同的字符串列名')
    missing = set([timestamp_column, *columns]) - set(df.columns)
    if missing:
        raise ValueError(f'缺少输入列：{sorted(missing)}')
    timestamps = pd.DatetimeIndex(pd.to_datetime(df[timestamp_column], errors='raise'))
    if timestamps.hasnans or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError('时间必须有效、唯一且递增')
    values = df[columns].astype(float).reset_index(drop=True)
    values.columns = sensors
    boundary = np.ones(len(df), dtype=bool)
    boundary[1:] = np.diff(timestamps.asi8) != pd.Timedelta(minutes=15).value
    if segment_column and segment_column in df:
        segments = df[segment_column].to_numpy()
        if pd.isna(segments).any():
            raise ValueError('分段列不能含空值')
        boundary[1:] |= segments[1:] != segments[:-1]
    return values, timestamps, boundary


def _alert_state(score, valid, threshold, config):
    state = False
    high = low = 0
    active = np.zeros(len(score), dtype=bool)
    new = np.zeros(len(score), dtype=bool)
    for i, (value, ok) in enumerate(zip(score, valid)):
        if not ok or not np.isfinite(value):
            state = False
            high = low = 0
            continue
        if not state:
            high = high + 1 if value > threshold else 0
            if high >= config['alarm_consecutive_points']:
                state = True
                new[i] = True
                high = 0
        else:
            low = low + 1 if value <= threshold * config['recovery_threshold_ratio'] else 0
            if low >= config['recovery_consecutive_points']:
                state = False
                low = 0
        active[i] = state
    return active, new


def _score(x, model):
    targets, inputs = model['targets'], model['reference_inputs']
    reference = np.full((len(x), len(targets)), np.nan)
    good = np.isfinite(x[inputs]).all(axis=1).to_numpy()
    if good.any():
        reference[good] = model['reference_model'].predict(x.loc[good, inputs])
    residual = x[targets].to_numpy() - reference
    history = pd.DataFrame(residual).shift(1).rolling(96, min_periods=96).median().to_numpy()
    scaler = model['change_scale']
    z = (residual - history - np.asarray(scaler['center'])) / np.asarray(scaler['scale']) / np.asarray(scaler['tail'])
    finite = pd.Series(np.isfinite(x.to_numpy()).all(axis=1))
    valid = (finite.rolling(97, min_periods=97).sum().eq(97).to_numpy()
             & x[model['speed_column']].between(*model['speed_bounds']).to_numpy())
    ema = np.zeros(len(targets))
    score = np.full(len(x), np.nan)
    for i in range(len(x)):
        if valid[i] and np.isfinite(z[i]).all():
            ema = .2 * z[i] + .8 * ema
            score[i] = np.max(abs(ema))
        else:
            ema[:] = 0
    if not np.isfinite(score[valid]).all():
        raise ValueError('模型输出含非有限值，请检查模型与输入单位')
    active, new = _alert_state(score, valid, model['threshold'], model['config'])
    return pd.DataFrame(dict(warning_valid=valid, warning_score=score,
                             warning_active=active, warning_new=new))


def execute(
    df: pd.DataFrame,
    model_path: str,
    timestamp_column: str = 'Timestamp',
    segment_column: str = 'segment_id',
    sensor_map_json: str = '',
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """根据过程测点输出告警状态。

    df：一台设备按时间递增的历史表，15 分钟间隔，含模型指定的 25 个数值测点。
    温度 ℃，压力 kg/cm²，轴位移 mm，振动 μm，转速 rpm；字段清单随模型交付。
    sensor_map_json：可选 JSON 字符串，映射“模型测点名:输入列名”；仅改列名，不转换单位。
    timestamp_column：时间列。segment_column：可选分段列，缺省不存在时按时间缺口分段。
    model_path：执行环境中 warning.pkl 的路径，阈值与参考模型来自训练期。
    out_1：Timestamp、warning_valid、warning_score、warning_active、warning_new。
    当前点之前须有 96 个连续有限采样点，且当前转速在模型范围内。
    时间缺口、显式分段重新开始历史；无效点状态为 False，必须同时检查 warning_valid。
    空表、重复或倒序时间、缺列会抛出异常。每次调用提供完整历史，不跨调用保存状态。
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('df 必须是非空 DataFrame')
    with Path(model_path).open('rb') as stream:
        model = pickle.load(stream)
    if model.get('format') != 'rcsd_ewma_operator_v1':
        raise ValueError('模型格式不匹配，请使用 export_dataflow_models.py 导出的 warning.pkl')
    x, timestamps, boundary = _input(df, model['sensors'], timestamp_column, segment_column, sensor_map_json)
    parts = []
    starts = np.flatnonzero(boundary)
    for start, stop in zip(starts, np.r_[starts[1:], len(x)]):
        part = _score(x.iloc[start:stop].reset_index(drop=True), model)
        part.insert(0, 'Timestamp', timestamps[start:stop])
        parts.append(part)
    result_df = pd.concat(parts, ignore_index=True)
    return {'out_1': result_df}
