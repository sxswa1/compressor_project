"""高淏宇：LSTM DI、固定四级风险、ExtraTrees RUL 及区间算子。

依赖 numpy、pandas、scikit-learn==1.5.2、torch==2.5.1（CPU 可运行）。
model_dir 为 export_dataflow_models.py 导出的模型目录，由维护方部署。
仅加载可信训练文件；不依赖研究脚本、工作目录或 Notebook。
"""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn


class _Network(nn.Module):
    def __init__(self, features, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


def _input(df, sensors, timestamp_column, segment_column, sensor_map_json):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('df 必须是非空 DataFrame')
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


def _windows(values, boundary):
    values = np.asarray(values, dtype=np.float32)
    ends, consecutive = [], 0
    for row, finite in enumerate(np.isfinite(values).all(axis=1)):
        if boundary[row]:
            consecutive = 0
        consecutive = consecutive + 1 if finite else 0
        if consecutive >= 96:
            ends.append(row)
    ends = np.asarray(ends, dtype=int)
    if not len(ends):
        return np.empty((0, 96, values.shape[1]), dtype=np.float32), ends
    return np.stack([values[i-95:i+1] for i in ends]), ends


def _features(x):
    return np.concatenate([x[:, -1], x[:, -24:].mean(axis=1), x[:, -1]-x[:, -24],
                           x.mean(axis=1), x[:, -1]-x[:, 0]], axis=1)


def _risk_levels(di, thresholds):
    return np.where(di >= thresholds[0], 1,
                    np.where(di >= thresholds[1], 2, np.where(di >= thresholds[2], 3, 4)))


def _lstm_predict(path, x):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'lstm_regressor_v1' or x.shape[1:] != (payload['window'], payload['features']):
        raise ValueError('LSTM 格式或测点窗口维度不匹配')
    params = payload['params']
    with torch.random.fork_rng(devices=[]):
        net = _Network(payload['features'], params['hidden_size'], params['num_layers'])
    net.load_state_dict(payload['state_dict'])
    net.eval()
    mean, scale = payload['mean'].numpy(), payload['scale'].numpy()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), params['batch_size']):
            batch = np.ascontiguousarray((x[start:start+params['batch_size']]-mean)/scale)
            predictions.append(net(torch.from_numpy(batch)).numpy())
    return np.concatenate(predictions)*payload['target_scale'] + payload['target_mean']


def execute(
    df: pd.DataFrame,
    model_dir: str,
    seed: int,
    timestamp_column: str = 'Timestamp',
    segment_column: str = 'segment_id',
    sensor_map_json: str = '',
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """输出健康指数、四级风险、剩余寿命及预测区间。

    df：单设备历史数据，15 分钟采样，含模型中的 25 个测点与时间列。
    单位：温度 ℃、压力 kg/cm²、轴位移 mm、振动 μm、转速 rpm。
    sensor_map_json：模型测点名到输入列名的 JSON 字符串，仅作字段映射，不转换单位。
    model_dir：导出目录，含 manifest.json、preprocessing.json、reference.pkl、
    interval_radii.json 和三个种子的 di_*.pt、rul_*.pkl。没有模型文件不能推断。
    seed：显式选择 20261023/20261024/20261025；单次输出不是三种子指标的平均。
    timestamp_column：时间列。segment_column：可选分段列，默认不存在时按时间缺口分段。
    out_1：Timestamp、seed、DI、risk_level、di_valid、RUL_hours、rul_valid、
    RUL_lower_hours、RUL_upper_hours。RUL 单位小时，DI 在 0—1 之间。
    风险：Ⅰ DI≥0.85；Ⅱ 0.65≤DI<0.85；Ⅲ 0.40≤DI<0.65；Ⅳ DI<0.40。
    每个有效点均有 RUL 区间，使用统一校准半径，不按真实寿命或风险标签切换模型。
    96 点窗口包含当前点；缺测、停机、时间缺口或分段切换需重新积累历史。
    无效预测为空且 valid=False，不能解释为正常。空表、缺列、无序时间抛出异常。
    每次调用提供完整历史，不跨调用保存状态。额外标签列不参与计算。
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed not in (20261023, 20261024, 20261025):
        raise ValueError('seed 必须为 20261023、20261024 或 20261025')
    directory = Path(model_dir)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'rcsd_dataflow_models_v1':
        raise ValueError('模型目录格式不匹配')
    meta = json.loads((directory/'preprocessing.json').read_text(encoding='utf-8'))
    if meta['history_points'] != 96 or meta['sampling_minutes'] != 15 or meta['risk_thresholds'] != [.85, .65, .4]:
        raise ValueError('模型配置与算子固定窗口、采样间隔或风险规则不匹配')
    sensors = meta['sensors']
    x, timestamps, boundary = _input(df, sensors, timestamp_column, segment_column, sensor_map_json)
    values = x.to_numpy(dtype=float).copy()
    good = np.isfinite(values).all(axis=1) & x[manifest['speed_column']].between(*meta['running_speed_range']).to_numpy()
    values[~good] = np.nan
    windows, ends = _windows(values, boundary)
    result_df = pd.DataFrame(dict(Timestamp=timestamps, seed=int(seed), DI=np.nan,
        risk_level=pd.array([pd.NA]*len(x), dtype='Int64'), di_valid=False,
        RUL_hours=np.nan, rul_valid=False, RUL_lower_hours=np.nan, RUL_upper_hours=np.nan))
    # Resolve resources even for a warm-up-only input, so a broken deployment fails clearly.
    di_path = directory/f'di_{seed}.pt'
    if not di_path.is_file():
        raise FileNotFoundError(di_path)
    with (directory/f'rul_{seed}.pkl').open('rb') as stream:
        model = pickle.load(stream)
    with (directory/'reference.pkl').open('rb') as stream:
        reference = pickle.load(stream)
    radii = json.loads((directory/'interval_radii.json').read_text(encoding='utf-8'))
    radius = radii[f'trees_history__residual_augmented__{seed}']['radius_hours']
    if radius is None or not np.isfinite(radius) or radius < 0:
        raise ValueError('缺少有限、非负的 RUL 区间校准半径')
    if len(ends):
        di = np.clip(_lstm_predict(di_path, windows), 0, 1)
        positions = [sensors.index(t) for t in reference['targets']]
        prediction = reference['model'].predict(x.loc[good, reference['context']])*reference['scale']+reference['center']
        values[np.ix_(good, positions)] -= prediction
        residual_windows, residual_ends = _windows(values, boundary)
        if not np.array_equal(ends, residual_ends):
            raise ValueError('背景处理后窗口失效，请检查输入或模型')
        rul = np.maximum(model.predict(_features(residual_windows)), 0)
        if not np.isfinite(di).all() or not np.isfinite(rul).all():
            raise ValueError('模型输出含非有限值')
        result_df.loc[ends, 'DI'] = di
        result_df.loc[ends, 'risk_level'] = _risk_levels(di, meta['risk_thresholds'])
        result_df.loc[ends, 'di_valid'] = True
        result_df.loc[ends, 'RUL_hours'] = rul
        result_df.loc[ends, 'rul_valid'] = True
        result_df.loc[ends, 'RUL_lower_hours'] = np.maximum(0, rul-radius)
        result_df.loc[ends, 'RUL_upper_hours'] = rul+radius
    return {'out_1': result_df}
