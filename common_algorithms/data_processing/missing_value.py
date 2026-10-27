"""Bounded missing-value imputation and time consistency checks.

Only existing rows are processed: absent timestamps are reported, not invented.
Linear interpolation and whole-gap length decisions are offline. Explicit
causal=True with ffill fills only the first max_gap rows as they arrive; it
requires a declared clock interval and ordered valid times when a clock is
supplied. No filling crosses record groups, large time gaps or splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from common_algorithms.data_processing.process_clock import normalize_process_time, validate_group_identifiers


def impute_timeseries(
    df: pd.DataFrame, value_columns, group_columns=None, time_column=None,
    method: str = "linear", max_gap: int = 5, expected_interval: float | None = None,
    duplicate_policy: str = "raise", source_timezone: str | None = None, causal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return processed rows and a row-aligned quality table, preserving order.

    Numeric time values use their supplied unit; datetime intervals use seconds.
    expected_interval has that same unit, or is inferred as the median positive
    interval per group. max_gap is the full number of consecutive missing rows;
    a longer run is left entirely missing in the default offline mode. In
    causal ffill, only the first max_gap missing rows are filled, regardless
    of how long the gap will later become. Duplicate times raise by default;
    duplicate_policy='flag' reports them and skips filling that group. NaT rows
    are reported and never used for interpolation. Nonfinite signal values
    become missing. Unrelated metadata columns and input objects are preserved.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    values = [value_columns] if isinstance(value_columns, str) else list(value_columns or [])
    groups = [group_columns] if isinstance(group_columns, str) else list(group_columns or [])
    required = values + groups + ([time_column] if time_column else [])
    if not values or len(set(values)) != len(values) or len(set(groups)) != len(groups):
        raise ValueError("Select unique value_columns and group_columns")
    if not df.columns.is_unique or any(c not in df.columns for c in required):
        raise ValueError("Missing required columns or duplicate input column names")
    if set(values) & (set(groups) | ({time_column} if time_column else set())):
        raise ValueError("Signal columns must be separate from group/time metadata")
    validate_group_identifiers(df, groups)
    if method not in {"none", "linear", "ffill"} or duplicate_policy not in {"raise", "flag"}:
        raise ValueError("Use method=none/linear/ffill and duplicate_policy=raise/flag")
    if not isinstance(causal,bool) or (causal and (method not in {"none","ffill"} or duplicate_policy!='raise')):
        raise ValueError('causal=True requires none/ffill and duplicate_policy=raise')
    if causal and time_column and expected_interval is None:
        raise ValueError('Causal clock processing requires an explicit expected_interval; do not infer it from future rows')
    if isinstance(max_gap, bool) or not isinstance(max_gap, (int, np.integer)) or max_gap < 0:
        raise ValueError("max_gap must be a nonnegative integer")
    if expected_interval is not None and (not np.isfinite(expected_interval) or expected_interval <= 0):
        raise ValueError("expected_interval must be finite and positive")
    work = df.reset_index(drop=True).copy(deep=True)
    quality = pd.DataFrame({"row_position": np.arange(len(work), dtype=np.int64)})
    quality['filling_uses_future_context'] = bool(method!='none' and not causal)
    for key in ["invalid_time", "duplicate_time", "out_of_order", "irregular_interval", "time_gap"]:
        quality[key] = False
    for col in values:
        if not pd.api.types.is_numeric_dtype(work[col]):
            raise ValueError(f"Signal column {col} must be numeric")
        a = work[col].to_numpy(dtype=float, na_value=np.nan)
        work[col] = np.where(np.isfinite(a), a, np.nan)
        quality[f"{col}__missing_before"] = ~np.isfinite(a)
        quality[f"{col}__filled"] = False
        quality[f"{col}__missing_after"] = ~np.isfinite(a)
    positions = list(work.groupby(groups, sort=False, dropna=False).indices.values()) if groups else [np.arange(len(work))]
    for indices in positions:
        indices = np.asarray(indices)
        if not len(indices):
            continue
        if time_column:
            time = work.iloc[indices][time_column]
            if pd.api.types.is_numeric_dtype(time):
                axis = time.to_numpy(dtype=float, na_value=np.nan)
            else:
                parsed = normalize_process_time(time, source_timezone)
                axis = parsed.dt.as_unit("ns").astype("int64").to_numpy(dtype=float) / 1e9
                axis[parsed.isna().to_numpy()] = np.nan
            invalid = ~np.isfinite(axis)
            quality.loc[indices, "invalid_time"] = invalid
            duplicated = pd.Series(axis).duplicated(keep=False).to_numpy() & ~invalid
            quality.loc[indices, "duplicate_time"] = duplicated
            if duplicated.any():
                if duplicate_policy == "raise":
                    raise ValueError("Duplicate timestamps within a group; resolve or use duplicate_policy=flag")
                continue
            quality.loc[indices[1:], "out_of_order"] = np.diff(axis) < 0
            if causal and (invalid.any() or np.any(np.diff(axis)<=0)):
                raise ValueError('Causal processing requires valid times in strictly increasing arrival order per group')
            order = np.argsort(axis, kind="stable")
            indices, axis = indices[order], axis[order]
        else:
            axis = np.arange(len(indices), dtype=float)
        delta = np.diff(axis)
        positive = delta[np.isfinite(delta) & (delta > 0)]
        interval = expected_interval if expected_interval is not None else (np.median(positive) if len(positive) else 1.)
        boundary = np.zeros(len(indices), dtype=bool)
        boundary[0] = True
        boundary[1:] = ~np.isfinite(delta) | (delta > interval * 1.5)
        quality.loc[indices[1:], "time_gap"] = np.isfinite(delta) & (delta > interval * 1.5)
        quality.loc[indices[1:], "irregular_interval"] = np.isfinite(delta) & ~np.isclose(delta, interval, rtol=.01, atol=1e-12)
        # Split at actual timestamp gaps before any interpolation.
        segments = np.split(np.arange(len(indices)), np.flatnonzero(boundary)[1:])
        for col in values:
            result = work.iloc[indices][col].to_numpy(dtype=float).copy()
            before = ~np.isfinite(result)
            if method != "none" and max_gap:
                for segment in segments:
                    if causal:
                        last=np.nan;missing_count=0
                        for position in segment:
                            if np.isfinite(result[position]):
                                last=result[position];missing_count=0
                            else:
                                missing_count+=1
                                if missing_count<=max_gap and np.isfinite(last):
                                    result[position]=last
                        continue
                    missing = segment[~np.isfinite(result[segment]) | ~np.isfinite(axis[segment])]
                    for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
                        if not len(run) or len(run) > max_gap or not np.isfinite(axis[run]).all():
                            continue
                        left, right = run[0] - 1, run[-1] + 1
                        if left < segment[0] or not np.isfinite(result[left]):
                            continue
                        if method == "ffill":
                            result[run] = result[left]
                        elif right <= segment[-1] and np.isfinite(result[right]):
                            result[run] = np.interp(axis[run], [axis[left], axis[right]], [result[left], result[right]])
            work.iloc[indices, work.columns.get_loc(col)] = result
            quality.loc[indices, f"{col}__filled"] = before & np.isfinite(result)
            quality.loc[indices, f"{col}__missing_after"] = ~np.isfinite(result)
    work.index = df.index.copy()
    quality.index = df.index.copy()
    return work, quality


def execute(
    df: pd.DataFrame, value_columns, group_columns=None, time_column=None,
    method: str = "linear", max_gap: int = 5, expected_interval: float | None = None,
    duplicate_policy: str = "raise", source_timezone: str | None = None, causal: bool = False, **kwargs,
) -> dict[str, pd.DataFrame]:
    """DataFlow out_1: filled input; out_2: consistency and fill flags.

    Requires numeric value_columns and numpy/pandas. No new timestamps are
    inserted. Default whole-gap limits are offline even for ffill. Explicit
    causal=True fills only the first max_gap rows and needs a known interval
    and ordered valid clock. It cannot retrospectively unfill a long gap.
    Group columns must include recording IDs and split IDs where applicable.
    """
    result_df, quality_df = impute_timeseries(
        df, value_columns, group_columns, time_column, method, max_gap,
        expected_interval, duplicate_policy, source_timezone, causal,
    )
    return {"out_1": result_df, "out_2": quality_df}
