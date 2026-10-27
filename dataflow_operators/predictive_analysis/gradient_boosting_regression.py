# 高淏宇；由 tools/build_regression_operators.py 生成。
# Source SHA256: 95f63c042c6167bb0ecab3203d33171c621286cb8ef0a347de0f9e685e41aebe
"""Tabular gradient-boosting regression; no compressor-specific fields or rules.

Uses sklearn HistGradientBoostingRegressor, not ExtraTrees or random forest.
Early stopping is disabled: callers must supply chronological validation themselves.
"""
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GradientBoostingRegressor:
    def __init__(self, max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                 min_samples_leaf=20, l2_regularization=1.0, random_state=20261023):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                           l2_regularization=l2_regularization, random_state=random_state)

    @staticmethod
    def _input(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError('X must be a nonempty finite [samples, features] array')
        return X

    def fit(self, X, y):
        X = self._input(X)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError('y must be a finite scalar target for each sample')
        self.model_ = HistGradientBoostingRegressor(early_stopping=False, **self.params)
        self.model_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, 'model_'):
            raise RuntimeError('fit or load a model before prediction')
        X = self._input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('feature dimension differs from training')
        return self.model_.predict(X)

    def save(self, path):
        if not hasattr(self, 'model_'):
            raise RuntimeError('cannot save an unfitted model')
        with Path(path).open('wb') as stream:
            pickle.dump(dict(format='gradient_boosting_v1', params=self.params,
                             features=self.n_features_in_, model=self.model_), stream, protocol=5)

    @classmethod
    def load(cls, path):
        """Load a trusted local pickle, in a compatible sklearn environment."""
        with Path(path).open('rb') as stream:
            payload = pickle.load(stream)
        if payload.get('format') != 'gradient_boosting_v1':
            raise ValueError('unsupported gradient boosting artifact')
        result = cls(**payload['params'])
        result.model_, result.n_features_in_ = payload['model'], payload['features']
        return result

"""高淏宇：梯度提升单目标回归的 DataFlow 入口。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def execute(
    df: pd.DataFrame,
    model_path: str,
    schema_path: str,
    feature_map_json: str = '',
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """对数值特征表进行直方图梯度提升回归。

    df：每行一个样本，含 schema 指定的特征列。时间窗口、特征构建由上游完成。
    model_path：公共 GradientBoostingRegressor.save 保存的可信 .pkl 权重。
    schema_path：配套 JSON，必填 model_sha256、feature_columns（训练时的顺序）、
    feature_units（逐列单位）、target_name、target_unit。输入数值必须与训练单位一致。
    如 schema 声明 feature_recipe，输入还必须由该配方构建，不能仅替换列名凑齐特征。
    feature_map_json：可选 JSON 字符串，映射“模型特征名:输入列名”，不转换单位。
    out_1=result_df：保留输入行、索引和列，新增 prediction、valid。
    prediction 单位为 target_unit，输出原始回归值，不自动裁剪或划分风险。
    含非有限特征的行 prediction=NaN、valid=False；额外标签列不进入模型。
    缺列、重复列、已有 prediction/valid 列、错误模型配对抛异常。
    空表须有必需列，校验模型后返回同结构空结果。不修改输入，不在预测时训练。
    模型使用 sklearn HistGradientBoostingRegressor，不是随机森林或 ExtraTrees。
    依赖 numpy、pandas、scikit-learn（已验证 1.5.2）。仅加载可信 pickle 文件。
    """
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError('df 必须是列名唯一的 DataFrame')
    if {'prediction','valid'} & set(df):
        raise ValueError('输入不能已有 prediction 或 valid 列')
    schema = json.loads(Path(schema_path).read_text(encoding='utf-8'))
    path = Path(model_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != schema.get('model_sha256'):
        raise ValueError('模型与 schema 不匹配')
    columns, units = schema.get('feature_columns'), schema.get('feature_units')
    if not isinstance(columns,list) or not columns or any(not isinstance(c,str) or not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('schema 必须提供非空、不重复的 feature_columns')
    if not isinstance(units,dict) or set(units)!=set(columns) or any(not isinstance(v,str) or not v for v in units.values()):
        raise ValueError('schema 必须声明每个特征的单位')
    if any(not isinstance(schema.get(k),str) or not schema[k] for k in ['target_name','target_unit']):
        raise ValueError('schema 必须声明目标名称与单位')
    mapping = json.loads(feature_map_json) if feature_map_json else {}
    if not isinstance(mapping,dict) or set(mapping)-set(columns):
        raise ValueError('feature_map_json 只能映射模型中的特征名')
    actual = [mapping.get(c,c) for c in columns]
    if any(not isinstance(c,str) for c in actual) or len(set(actual))!=len(actual):
        raise ValueError('特征映射应使用不同的字符串列名')
    missing = set(actual)-set(df.columns)
    if missing:
        raise ValueError(f'缺少输入列：{sorted(missing)}')
    model = GradientBoostingRegressor.load(path)
    if model.n_features_in_ != len(columns):
        raise ValueError('schema 特征数量与权重不一致')
    values = df[actual].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    predictions = np.full(len(df),np.nan)
    if valid.any():
        predictions[valid] = model.predict(values[valid])
        if not np.isfinite(predictions[valid]).all():
            raise ValueError('模型预测出现非有限值')
    result_df = df.copy()
    result_df['prediction'] = predictions
    result_df['valid'] = valid
    return {'out_1': result_df}
