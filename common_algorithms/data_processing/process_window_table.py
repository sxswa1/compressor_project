"""Table-to-window bridge with explicit provenance and quality decisions.

This adds no fault labels, interpolation, scaling or train/test assignment.
It prepares complete windows for existing feature and diagnosis operators.
"""
import numpy as np
import pandas as pd
from common_algorithms.data_processing.process_clock import normalize_process_time


def _names(value, name, allow_empty=False):
    result = [value] if isinstance(value, str) else list(value or [])
    if ((not result and not allow_empty) or len(result) != len(set(result))
            or any(not isinstance(v, str) or not v for v in result)):
        raise ValueError(f'{name} requires distinct nonempty column names')
    return result


def execute(df: pd.DataFrame, value_columns, group_columns, interval_seconds,
            window_points, stride_points=None, time_column='timestamp',
            quality_columns=None, window_id_prefix='process', source_timezone=None,
            max_output_rows=1_000_000, max_candidate_windows=100_000, **kwargs):
    """Return accepted long-table windows and all candidate/tail decisions.

    group_columns is explicit and nonempty: include device, original
    record/event and split identities as applicable. A record's normal and
    faulty portions must retain their true shared identity. Timestamps are
    sorted per group; duplicates/invalid times fail. Every cadence break
    starts another segment. Missing/nonfinite measurements and True boolean
    quality flags reject a candidate. One incomplete tail per segment is
    reported; observations are never compressed together across a gap.

    out_1 retains input columns and adds window_id/source_row_position;
    source_row_position is positional provenance, not an incident identity.
    out_2 contains accepted/rejected candidates, reasons and source groups.
    IDs are scoped to this call; use a distinct window_id_prefix per batch
    before concatenating outputs from multiple calls. Overlapping windows
    share observations and are not independent faults. Input is unchanged.
    """
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError('A DataFrame with unique column names is required')
    values = _names(value_columns, 'value_columns')
    groups = _names(group_columns, 'group_columns')
    quality = _names(quality_columns, 'quality_columns', allow_empty=True)
    if not isinstance(time_column, str) or not time_column:
        raise ValueError('time_column must be a nonempty column name')
    roles = values + groups + quality + [time_column]
    if len(roles) != len(set(roles)) or any(c not in df for c in roles):
        raise ValueError('Missing columns or overlapping signal/group/quality/time roles')
    decision_names = {'window_id', 'source_row_position', 'segment_number', 'start_time', 'end_time',
                      'candidate_points', 'missing_scalar_values', 'quality_flagged_rows',
                      'status', 'reason', 'source_row_start', 'source_row_end'}
    if {'window_id', 'source_row_position'} & set(df.columns) or set(groups) & decision_names:
        raise ValueError('Input uses reserved window/provenance/decision names')
    if (isinstance(interval_seconds, (bool, np.bool_)) or not np.isfinite(interval_seconds)
            or interval_seconds <= 0):
        raise ValueError('interval_seconds must be finite and positive')
    stride = window_points if stride_points is None else stride_points
    for name, value, minimum in [('window_points', window_points, 2), ('stride_points', stride, 1),
                                  ('max_output_rows', max_output_rows, 1), ('max_candidate_windows', max_candidate_windows, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer at least {minimum}')
    if not isinstance(window_id_prefix, str) or not window_id_prefix.strip():
        raise ValueError('window_id_prefix must be a nonempty batch namespace')
    if df[groups].isna().any().any():
        raise ValueError('Missing group identifiers must be resolved before windowing')
    for name in groups:
        if any(isinstance(value, str) and not value.strip() for value in df[name]):
            raise ValueError('Blank group identifiers must be resolved before windowing')
        if pd.api.types.is_numeric_dtype(df[name]) and not np.isfinite(df[name].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite group identifiers must be resolved before windowing')
    for name in values:
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise ValueError(f'Signal column {name} must be numeric')
    for name in quality:
        if not pd.api.types.is_bool_dtype(df[name]) or df[name].isna().any():
            raise ValueError('Quality columns must be explicit boolean flags without missing values')
    work = df.reset_index(drop=True).copy(deep=True)
    work[time_column] = normalize_process_time(work[time_column], source_timezone)
    if work[time_column].isna().any():
        raise ValueError('Invalid timestamps must be resolved before windowing')
    work['source_row_position'] = np.arange(len(work), dtype=np.int64)
    accepted, decisions = [], []
    output_rows = 0
    for key, part in work.groupby(groups, sort=False, dropna=False):
        part = part.sort_values(time_column, kind='stable')
        if part[time_column].duplicated().any():
            raise ValueError('Duplicate timestamps within a windowing group')
        delta = part[time_column].diff().dt.total_seconds().to_numpy()
        boundaries = np.flatnonzero(~np.isclose(delta[1:], interval_seconds, rtol=0., atol=1e-6)) + 1
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(groups, keys))
        for segment, ids in enumerate(np.split(np.arange(len(part)), boundaries)):
            for begin in range(0, len(ids), stride):
                candidate = part.iloc[ids[begin:begin + window_points]].copy()
                matrix = candidate[values].to_numpy(dtype=float, na_value=np.nan)
                missing = int((~np.isfinite(matrix)).sum())
                flagged = int(candidate[quality].any(axis=1).sum()) if quality else 0
                reasons = []
                if len(candidate) < window_points:
                    reasons.append('incomplete_tail')
                if missing:
                    reasons.append('missing_or_nonfinite_values')
                if flagged:
                    reasons.append('flagged_quality')
                if len(decisions) >= max_candidate_windows:
                    raise ValueError('Windowing exceeds max_candidate_windows; inspect cadence/stride and groups')
                window_id = f'{window_id_prefix}_{len(decisions):08d}'
                positions = candidate.source_row_position.to_numpy()
                decisions.append({**identity, 'window_id': window_id, 'segment_number': segment,
                                  'start_time': candidate[time_column].iloc[0], 'end_time': candidate[time_column].iloc[-1],
                                  'candidate_points': len(candidate), 'missing_scalar_values': missing,
                                  'quality_flagged_rows': flagged, 'status': 'rejected' if reasons else 'accepted',
                                  'reason': '|'.join(reasons) if reasons else 'accepted',
                                  'source_row_start': int(positions[0]), 'source_row_end': int(positions[-1])})
                if not reasons:
                    output_rows += len(candidate)
                    if output_rows > max_output_rows:
                        raise ValueError('Windowing exceeds max_output_rows; inspect overlap and groups')
                    candidate['window_id'] = window_id
                    accepted.append(candidate)
                if len(candidate) < window_points:
                    break
    result_columns = list(df.columns) + ['source_row_position', 'window_id']
    decision_columns = groups + ['window_id', 'segment_number', 'start_time', 'end_time', 'candidate_points',
                                'missing_scalar_values', 'quality_flagged_rows', 'status', 'reason',
                                'source_row_start', 'source_row_end']
    result = pd.concat(accepted, ignore_index=True)[result_columns] if accepted else work.iloc[:0].assign(window_id=pd.Series(dtype=str))[result_columns]
    return {'out_1': result, 'out_2': pd.DataFrame(decisions, columns=decision_columns)}
