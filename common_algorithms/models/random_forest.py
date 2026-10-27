"""Reusable random-forest classifier and feature-table DataFlow interface.

Only numpy, pandas, scikit-learn and joblib are required. Dataset-specific
feature construction and label interpretation belong to the calling pipeline.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X, y, n_estimators=200, max_depth=None, random_state=42):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or not len(X) or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError("X must be a finite nonempty feature matrix aligned with y")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=1, class_weight="balanced")
    return model.fit(X, y)


def execute(df: pd.DataFrame, feature_columns, model_path: str, mode="predict",
            label_column=None, n_estimators=200, max_depth=None, random_state=42,
            input_contract=None, **kwargs):
    """out_1 predictions; out_2 model information. mode=train saves a model.

    Feature names/order are stored at training and checked at inference.
    model_path must be supplied by the deployment; only trusted joblib model
    resources may be loaded. Input metadata columns are retained in out_1.
    For signal-specific artifacts, pass input_contract or use the attrs emitted
    by the diagnosis feature operator. Physical type, unit, sampling rate and
    feature-window settings must match; feature names alone are insufficient.
    """
    columns = [feature_columns] if isinstance(feature_columns, str) else list(feature_columns)
    if not columns or len(set(columns)) != len(columns) or any(c not in df for c in columns):
        raise ValueError("Supply distinct existing feature_columns")
    if not model_path or mode not in {"train", "predict"}:
        raise ValueError("model_path and mode=train/predict are required")
    X = df[columns].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Features contain missing or nonfinite values")
    path = Path(model_path)
    actual_contract = input_contract if input_contract is not None else df.attrs.get("signal_contract")
    contract_keys = ["signal_type", "signal_unit", "sample_rate", "window_len", "windows_per_record", "fine_bands"]
    if actual_contract is not None and any(key not in actual_contract or actual_contract[key] is None for key in contract_keys):
        raise ValueError("Incomplete signal input_contract")
    if mode == "train":
        if label_column not in df or label_column in columns:
            raise ValueError("Select a separate existing label_column")
        model = train_random_forest(X, df[label_column].to_numpy(), n_estimators, max_depth, random_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_columns": columns, "input_contract": actual_contract}, path)
    else:
        artifact = joblib.load(path)
        if artifact["feature_columns"] != columns:
            raise ValueError("Feature columns/order differ from the trained model")
        expected_contract = artifact.get("input_contract")
        if expected_contract is not None:
            if actual_contract is None or any(actual_contract.get(key) != expected_contract.get(key) for key in contract_keys):
                raise ValueError("Signal input_contract differs from training (type/unit/rate/window/features)")
        model = artifact["model"]
    result_df = df.copy()
    if len(X):
        scores = model.predict_proba(X)
        result_df["predicted_label"] = model.classes_[scores.argmax(axis=1)]
        if mode == "predict" and "class_names" in artifact:
            result_df["predicted_label"] = [artifact["class_names"][int(i)] for i in result_df["predicted_label"]]
        result_df["model_score"] = scores.max(axis=1)
    else:
        result_df["predicted_label"] = pd.Series(index=df.index, dtype=object)
        result_df["model_score"] = pd.Series(index=df.index, dtype=float)
    info_df = pd.DataFrame([{"model": "random_forest", "mode": mode,
                             "model_path": str(path), "n_features": len(columns)}])
    return {"out_1": result_df, "out_2": info_df}
