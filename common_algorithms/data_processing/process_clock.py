"""Validate record identifiers and preserve explicitly resolved source clocks."""
import math
from numbers import Number
import pandas as pd


def validate_group_identifiers(df, group_columns):
    """Reject unknown record IDs before grouping, filling or time alignment.

    Missing values, blank strings and nonfinite numeric IDs cannot identify
    a device/record/split. Object-dtype numeric values are checked as well.
    Callers validate column presence before using this helper.
    """
    if not group_columns:
        return
    if df[group_columns].isna().any().any():
        raise ValueError('Missing group identifiers must be resolved before processing')
    for column in group_columns:
        for value in df[column]:
            if isinstance(value, str) and not value.strip():
                raise ValueError('Blank group identifiers must be resolved before processing')
            if isinstance(value, Number):
                try:
                    finite = math.isfinite(value)
                except (TypeError, ValueError, OverflowError):
                    finite = False
                if not finite:
                    raise ValueError('Nonfinite or invalid numeric group identifiers must be resolved before processing')


def normalize_process_time(values,source_timezone=None):
    """Return parsed timestamps without inventing a timezone for naive input.

    Fully timezone-aware inputs convert to UTC. Naive inputs stay naive unless
    source_timezone is explicitly supplied, in which case localization must be
    unambiguous and converts to UTC. Mixed naive/aware inputs raise. Invalid
    strings become NaT so callers can reject them. Numeric clocks require an
    explicit unit conversion by the caller before using this datetime API.
    """
    if not isinstance(values,pd.Series):
        values=pd.Series(values)
    if len(values) and pd.api.types.is_numeric_dtype(values):
        raise ValueError('Numeric timestamps require an explicit unit conversion before process-time parsing')
    awareness=set()
    for value in values:
        try:
            parsed=pd.Timestamp(value)
        except (ValueError,TypeError,OverflowError):
            continue
        if not pd.isna(parsed):
            awareness.add(parsed.tzinfo is not None)
    if len(awareness)>1:
        raise ValueError('Mixed naive and timezone-aware timestamps require explicit source-clock resolution')
    if awareness=={True}:
        return pd.to_datetime(values,errors='coerce',format='mixed',utc=True)
    result=pd.to_datetime(values,errors='coerce',format='mixed')
    if source_timezone is not None:
        result=result.dt.tz_localize(source_timezone,ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')
    return result
