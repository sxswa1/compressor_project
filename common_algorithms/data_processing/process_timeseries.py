"""Explicit time-grid alignment and leakage-aware windows for process signals.

No imputation, fault labelling, unit conversion or train/test splitting happens
implicitly. Callers must group by device and split/record before alignment.
"""
import numpy as np
import pandas as pd
from common_algorithms.data_processing.process_clock import normalize_process_time, validate_group_identifiers


def align_process_table(df, value_columns, time_column="timestamp", group_columns=None,
                        interval_seconds=360., snap_tolerance_seconds=0., max_rows=1_000_000,source_timezone=None):
    """Align numeric channels to each group's grid anchored at its first time.

    Naive source clocks stay naive unless source_timezone is explicit; aware
    timestamps convert to UTC. Mixed naive/aware values raise. Snapping
    requires an explicit tolerance smaller than half an interval. Duplicate
    times always raise, including collisions caused by snapping. Missing grid
    rows are inserted as NaN and marked. Original data and metadata are kept.
    The returned quality table is row-aligned with the result.
    """
    values = [value_columns] if isinstance(value_columns, str) else list(value_columns)
    groups = [group_columns] if isinstance(group_columns, str) else list(group_columns or [])
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("A DataFrame with unique columns is required")
    if not values or len(set(values+groups+[time_column])) != len(values+groups+[time_column]):
        raise ValueError("Select distinct value, group and time columns")
    if any(c not in df for c in values+groups+[time_column]):
        raise ValueError("Missing required columns")
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if not np.isfinite(snap_tolerance_seconds) or not 0 <= snap_tolerance_seconds < interval_seconds/2:
        raise ValueError("Snap tolerance must be nonnegative and smaller than half an interval")
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    work = df.copy(deep=True)
    work[time_column] = normalize_process_time(work[time_column],source_timezone)
    validate_group_identifiers(work, groups)
    if work[time_column].isna().any():
        raise ValueError("Invalid timestamps; resolve before alignment")
    for c in values:
        if not pd.api.types.is_numeric_dtype(work[c]):
            raise ValueError(f"{c} must be numeric")
    reserved = {"__original_timestamp", "__present", "__snapped"}
    if reserved & set(work.columns):
        raise ValueError("Input uses reserved alignment columns")
    tables, qualities, count = [], [], 0
    grouped = work.groupby(groups, sort=False, dropna=False) if groups else [((),work)]
    for key, part in grouped:
        part = part.sort_values(time_column, kind="stable").copy()
        if part.empty:
            continue
        if part[time_column].duplicated().any():
            raise ValueError("Duplicate timestamps within a group")
        anchor = part[time_column].iloc[0]
        offset = (part[time_column]-anchor).dt.total_seconds().to_numpy()
        slots = np.rint(offset/interval_seconds).astype(np.int64)
        error = np.abs(offset-slots*interval_seconds)
        if np.any(error > snap_tolerance_seconds+1e-6):
            raise ValueError("Off-grid timestamps exceed the explicit snap tolerance")
        if len(set(slots)) != len(slots):
            raise ValueError("Snapping caused duplicate timestamps")
        size = int(slots.max())+1
        count += size
        if count > max_rows:
            raise ValueError("Alignment exceeds max_rows; check interval and group identifiers")
        part["__original_timestamp"] = part[time_column]
        part["__present"] = True
        part["__snapped"] = error > 1e-6
        part[time_column] = anchor+pd.to_timedelta(slots*interval_seconds,unit="s")
        grid = pd.date_range(anchor,periods=size,freq=pd.to_timedelta(interval_seconds,unit="s"))
        aligned = part.set_index(time_column).reindex(grid)
        aligned.index.name = time_column
        keys = key if isinstance(key,tuple) else (key,)
        for c,v in zip(groups,keys):
            aligned[c] = v
        present = aligned.pop("__present").eq(True).to_numpy()
        snapped = aligned.pop("__snapped").eq(True).to_numpy()
        quality = aligned[groups].reset_index().copy()
        quality["inserted_timestamp"] = ~present
        quality["snapped_timestamp"] = snapped
        quality["original_timestamp"] = aligned.pop("__original_timestamp").to_numpy()
        for c in values:
            quality[c+"__missing"] = ~np.isfinite(aligned[c].to_numpy(dtype=float,na_value=np.nan))
        tables.append(aligned.reset_index()[work.columns])
        qualities.append(quality)
    if not tables:
        return work.iloc[:0].copy(), pd.DataFrame(columns=groups+[time_column,"inserted_timestamp","snapped_timestamp","original_timestamp"]+[c+"__missing" for c in values])
    return pd.concat(tables,ignore_index=True), pd.concat(qualities,ignore_index=True)


def process_windows(df, value_columns, time_column="timestamp", group_columns=None,
                    interval_seconds=360., window_points=20, stride_points=20,source_timezone=None):
    """Return finite N x channels x time windows and provenance metadata.

    Windows never cross groups, actual time gaps or a remaining nonfinite
    observation. Groups should include device, split and continuity/record IDs.
    Overlapping windows retain their source positions; they are not independent
    fault records. Missing data must be handled explicitly before calling.
    """
    values = [value_columns] if isinstance(value_columns,str) else list(value_columns)
    groups = [group_columns] if isinstance(group_columns,str) else list(group_columns or [])
    if not values or len(set(values+groups+[time_column])) != len(values+groups+[time_column]):
        raise ValueError("Select distinct value, group and time columns")
    if not isinstance(df,pd.DataFrame) or not df.columns.is_unique or any(c not in df for c in values+groups+[time_column]):
        raise ValueError("Missing or duplicate input columns")
    for n in [window_points,stride_points]:
        if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1:
            raise ValueError("Window and stride must be positive integers")
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    work=df.reset_index(drop=True).copy()
    work[time_column]=normalize_process_time(work[time_column],source_timezone)
    validate_group_identifiers(work, groups)
    if work[time_column].isna().any():
        raise ValueError("Invalid time values")
    windows, records=[],[]
    grouped=work.groupby(groups,sort=False,dropna=False) if groups else [((),work)]
    for key,part in grouped:
        part=part.sort_values(time_column,kind="stable")
        if part[time_column].duplicated().any():
            raise ValueError("Duplicate timestamps within window group")
        x=part[values].to_numpy(dtype=float)
        finite=np.isfinite(x).all(axis=1)
        delta=part[time_column].diff().dt.total_seconds().to_numpy()
        breaks=np.flatnonzero(~np.isclose(delta[1:],interval_seconds,rtol=0,atol=1e-6))+1
        keys=key if isinstance(key,tuple) else (key,)
        for segment,positions in enumerate(np.split(np.arange(len(part)),breaks)):
            for begin in range(0,len(positions)-window_points+1,stride_points):
                ids=positions[begin:begin+window_points]
                if not finite[ids].all():
                    continue
                windows.append(x[ids].T.astype(np.float32))
                records.append({**dict(zip(groups,keys)),"segment_number":segment,
                    "start_time":part[time_column].iloc[ids[0]],"end_time":part[time_column].iloc[ids[-1]],
                    "source_row_start":int(part.index[ids[0]]),"source_row_end":int(part.index[ids[-1]])})
    columns=groups+["segment_number","start_time","end_time","source_row_start","source_row_end"]
    return (np.stack(windows) if windows else np.empty((0,len(values),window_points),dtype=np.float32),pd.DataFrame(records,columns=columns))


def execute(df: pd.DataFrame, value_columns, time_column="timestamp", group_columns=None,
            interval_seconds=360., snap_tolerance_seconds=0., max_rows=1_000_000,source_timezone=None, **kwargs):
    """DataFlow out_1 aligned table; out_2 row quality. No interpolation.

    Numeric channels retain original units. Group by split/device/record before
    use. Naive clocks are preserved; optional source_timezone requires a known
    timezone. Already-aware inputs convert to UTC. Mixed clocks are rejected.
    """
    result_df,quality_df=align_process_table(df,value_columns,time_column,group_columns,
        interval_seconds,snap_tolerance_seconds,max_rows,source_timezone)
    return {"out_1":result_df,"out_2":quality_df}
