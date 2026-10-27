"""Conservative offline time-series quality marking and optional denoising.

Requires numpy and pandas. Default action only flags suspected outliers: a real
fault impulse must not automatically be deleted. Statistical thresholds are
computed per call/group, so callers must process train/test separately. The
optional centred filters are offline operations, not streaming predictors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from common_algorithms.data_processing.process_clock import normalize_process_time, validate_group_identifiers


def _columns(value, name):
    result = [value] if isinstance(value, str) else list(value or [])
    if not result and name == "value_columns":
        raise ValueError("value_columns must name at least one numeric signal column")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate names")
    return result


def clean_timeseries(
    df: pd.DataFrame, value_columns, group_columns=None, time_column=None,
    outlier_method: str = "mad", threshold: float = 6.0, action: str = "flag",
    denoise: str = "none", window: int = 5, range_limits: dict | None = None,
    expected_interval: float | None = None, source_timezone: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (processed_table, per_row_quality) without mutating the input.

    value_columns/group_columns accept a column name or a list. time_column,
    when supplied, orders each group for filtering; it must have no nulls or
    duplicates within that group. range_limits maps column -> [low, high],
    with either bound optionally None. action='replace_nan' explicitly masks
    suspected outliers; 'flag' preserves them. denoise='median' or 'mean' is an
    explicit offline, centred filter. Filtering never bridges existing NaNs.
    Row order, original index, metadata and unrelated columns are preserved.
    expected_interval uses numeric time units or seconds for datetimes; when
    omitted, the median positive group interval is used to split time gaps.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    values = _columns(value_columns, "value_columns")
    groups = _columns(group_columns, "group_columns")
    required = values + groups + ([time_column] if time_column else [])
    missing = [c for c in required if c not in df.columns]
    if missing or not df.columns.is_unique:
        raise ValueError(f"Missing columns {missing} or duplicate input column names")
    if set(values) & (set(groups) | ({time_column} if time_column else set())):
        raise ValueError("Signal columns must be separate from group/time metadata")
    validate_group_identifiers(df, groups)
    if outlier_method not in {"mad", "none"} or action not in {"flag", "replace_nan"}:
        raise ValueError("Use outlier_method=mad/none and action=flag/replace_nan")
    if denoise not in {"none", "median", "mean"}:
        raise ValueError("denoise must be none, median or mean")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if expected_interval is not None and (not np.isfinite(expected_interval) or expected_interval <= 0):
        raise ValueError("expected_interval must be finite and positive")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")
    limits = range_limits or {}
    if not isinstance(limits, dict) or set(limits) - set(values):
        raise ValueError("range_limits must map selected signal columns to [low, high]")
    for column, bounds in limits.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Invalid bounds for {column}")
        low, high = bounds
        if any(v is not None and not np.isfinite(v) for v in bounds) or (low is not None and high is not None and low > high):
            raise ValueError(f"Invalid bounds for {column}")
    work = df.reset_index(drop=True).copy(deep=True)
    quality = pd.DataFrame({"row_position": np.arange(len(df), dtype=np.int64)})
    quality["time_gap"] = False
    for col in values:
        if not pd.api.types.is_numeric_dtype(work[col]):
            raise ValueError(f"Signal column {col} must be numeric")
        quality[f"{col}__nonfinite"] = ~np.isfinite(work[col].to_numpy(dtype=float, na_value=np.nan))
        quality[f"{col}__outlier"] = False
        quality[f"{col}__modified"] = False
    positions = list(work.groupby(groups, sort=False, dropna=False).indices.values()) if groups else [np.arange(len(work))]
    for indices in positions:
        indices = np.asarray(indices)
        time_break = np.zeros(len(indices), dtype=bool)
        if time_column:
            time = work.iloc[indices][time_column]
            if pd.api.types.is_numeric_dtype(time):
                axis = time.to_numpy(dtype=float, na_value=np.nan)
            else:
                parsed = normalize_process_time(time, source_timezone)
                axis = parsed.dt.as_unit("ns").astype("int64").to_numpy(dtype=float) / 1e9
                axis[parsed.isna().to_numpy()] = np.nan
            if not np.isfinite(axis).all() or pd.Series(axis).duplicated().any():
                raise ValueError("Filtering order requires valid unique time values within each group")
            order = np.argsort(axis, kind="stable")
            indices, axis = indices[order], axis[order]
            delta = np.diff(axis)
            interval = expected_interval if expected_interval is not None else (np.median(delta) if len(delta) else 1.)
            time_break[1:] = delta > interval * 1.5
            quality.loc[indices, "time_gap"] = time_break
        for col in values:
            original = work.iloc[indices][col].to_numpy(dtype=float, na_value=np.nan).copy()
            finite = np.isfinite(original)
            flags = np.zeros(len(indices), dtype=bool)
            if finite.any() and outlier_method == "mad":
                finite_values = original[finite]
                centre = np.median(finite_values)
                scale = 1.4826 * np.median(np.abs(finite_values - centre))
                if scale <= np.finfo(float).eps:
                    scale = finite_values.std()
                if scale > np.finfo(float).eps:
                    flags[finite] = np.abs(original[finite] - centre) > threshold * scale
            if col in limits:
                low, high = limits[col]
                if low is not None:
                    flags |= finite & (original < low)
                if high is not None:
                    flags |= finite & (original > high)
            result = original.copy()
            if action == "replace_nan":
                result[flags | ~finite] = np.nan
            if denoise != "none":
                valid_positions = np.flatnonzero(np.isfinite(result))
                discontinuity = (np.diff(valid_positions) > 1) | time_break[valid_positions[1:]]
                runs = np.split(valid_positions, np.flatnonzero(discontinuity) + 1)
                for run in runs:
                    if not len(run):
                        continue
                    rolling = pd.Series(result[run]).rolling(window, center=True, min_periods=1)
                    result[run] = (rolling.median() if denoise == "median" else rolling.mean()).to_numpy()
            if action != "flag" or denoise != "none":
                work[col] = work[col].astype(float)
                work.iloc[indices, work.columns.get_loc(col)] = result
            modified = ~np.isclose(original, result, equal_nan=True, rtol=0, atol=0)
            quality.loc[indices, f"{col}__outlier"] = flags
            quality.loc[indices, f"{col}__modified"] = modified
    work.index = df.index.copy()
    quality.index = df.index.copy()
    return work, quality


def execute(
    df: pd.DataFrame, value_columns, group_columns=None, time_column=None,
    outlier_method: str = "mad", threshold: float = 6.0, action: str = "flag",
    denoise: str = "none", window: int = 5, range_limits: dict | None = None,
    expected_interval: float | None = None, source_timezone: str | None = None, **kwargs,
) -> dict[str, pd.DataFrame]:
    """DataFlow: out_1 processed input; out_2 per-row quality flags.

    Required: numeric value_columns. Defaults only mark suspected bad points;
    labels are neither required nor used. Requires numpy, pandas. See
    clean_timeseries for units, grouping, mutation and offline-filter limits.
    """
    result_df, quality_df = clean_timeseries(
        df, value_columns, group_columns, time_column, outlier_method,
        threshold, action, denoise, window, range_limits, expected_interval, source_timezone,
    )
    return {"out_1": result_df, "out_2": quality_df}
