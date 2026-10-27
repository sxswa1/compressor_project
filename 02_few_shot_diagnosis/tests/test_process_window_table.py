"""Window preparation must not fabricate continuity or independent incidents."""
import importlib.util
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common_algorithms.data_processing.process_window_table import execute
from common_algorithms.data_processing.process_timeseries import process_windows

ROOT = Path(__file__).resolve().parents[2]


def table(points=9):
    return pd.DataFrame({'timestamp': pd.date_range('2000-01-01', periods=points, freq='360s'),
                         'device': 'A', 'record': 'run_1', 'split': 'train',
                         'x': np.arange(points, dtype=float), 'bad': False})


def call(frame, **kwargs):
    return execute(frame, ['x'], ['device', 'record', 'split'], 360., 4, **kwargs)


def test_keeps_original_groups_values_and_rejects_one_partial_tail():
    first = table()
    second = table(4).assign(record='run_2', split='query', x=100.)
    frame = pd.concat([first, second], ignore_index=True)
    frame.index = [7] * len(frame)
    before = frame.copy(deep=True)
    result = call(frame)
    pd.testing.assert_frame_equal(frame, before)
    accepted = result['out_1']
    assert len(accepted) == 12 and accepted.window_id.nunique() == 3
    assert result['out_2'].status.tolist() == ['accepted', 'accepted', 'rejected', 'accepted']
    assert result['out_2'].iloc[2].reason == 'incomplete_tail'
    assert all(part.record.nunique() == part.split.nunique() == 1 for _, part in accepted.groupby('window_id'))
    np.testing.assert_array_equal(accepted.source_row_position, list(range(8)) + list(range(9, 13)))


def test_clock_gap_is_not_compressed_into_a_model_window():
    frame = table(8).drop(index=2)
    result = call(frame)
    assert len(result['out_1']) == 4
    assert result['out_1'].x.tolist() == [3., 4., 5., 6.]
    assert result['out_2'].reason.tolist() == ['incomplete_tail', 'accepted', 'incomplete_tail']
    assert result['out_2'].segment_number.tolist() == [0, 1, 1]


def test_missing_and_quality_windows_are_explicitly_rejected_not_imputed():
    frame = table(12)
    frame.loc[2, 'x'] = np.nan
    frame.loc[7, 'bad'] = True
    result = call(frame, quality_columns=['bad'])
    assert result['out_2'].reason.tolist() == ['missing_or_nonfinite_values', 'flagged_quality', 'accepted']
    assert result['out_1'].x.tolist() == [8., 9., 10., 11.]
    assert np.isnan(frame.loc[2, 'x'])


def test_overlapping_windows_keep_shared_row_provenance():
    result = call(table(6), stride_points=2)
    assert result['out_1'].window_id.nunique() == 2
    assert result['out_1'].source_row_position.tolist() == [0, 1, 2, 3, 2, 3, 4, 5]
    assert result['out_2'].iloc[-1].reason == 'incomplete_tail'


def test_regular_window_values_match_existing_array_window_helper():
    frame = pd.concat([table(9), table(5).assign(record='run_2')], ignore_index=True)
    expected, meta = process_windows(frame, ['x'], group_columns=['device', 'record', 'split'],
                                    interval_seconds=360., window_points=4, stride_points=4)
    accepted = call(frame)['out_1']
    actual = np.stack([part[['x']].to_numpy(dtype=np.float32).T for _, part in accepted.groupby('window_id', sort=False)])
    np.testing.assert_array_equal(actual, expected)
    assert len(meta) == accepted.window_id.nunique()


def test_explicit_missing_group_and_duplicate_times_are_rejected():
    with pytest.raises(ValueError, match='group_columns'):
        execute(table(), ['x'], [], 360., 4)
    with pytest.raises(ValueError, match='Missing group'):
        call(table().assign(record=pd.NA))
    with pytest.raises(ValueError, match='Blank group'):
        call(table().assign(record='  '))
    with pytest.raises(ValueError, match='Nonfinite group'):
        call(table().assign(record=np.inf))
    categorical = table()
    categorical['record'] = categorical.record.astype('category')
    assert len(call(categorical)['out_1']) == 8
    frame = table()
    frame.loc[1, 'timestamp'] = frame.loc[0, 'timestamp']
    with pytest.raises(ValueError, match='Duplicate timestamps'):
        call(frame)


def test_flags_require_boolean_nonmissing_values_and_separate_roles():
    with pytest.raises(ValueError, match='boolean flags'):
        call(table().assign(bad=1), quality_columns=['bad'])
    frame = table()
    frame['bad'] = frame.bad.astype('boolean')
    frame.loc[0, 'bad'] = pd.NA
    with pytest.raises(ValueError, match='boolean flags'):
        call(frame, quality_columns=['bad'])
    with pytest.raises(ValueError, match='overlapping'):
        call(table(), quality_columns=['x'])


def test_bad_configuration_and_reserved_columns_fail_instead_of_partial_results():
    for kwargs, match in [({'stride_points': 0}, 'stride_points'), ({'window_id_prefix': ''}, 'namespace'),
                          ({'max_output_rows': 7}, 'max_output_rows'),
                          ({'max_candidate_windows': 1}, 'max_candidate_windows')]:
        with pytest.raises(ValueError, match=match):
            call(table(), **kwargs)
    with pytest.raises(ValueError, match='reserved'):
        call(table().assign(window_id='already_windowed'))
    with pytest.raises(ValueError, match='window_points'):
        execute(table(), ['x'], ['record'], 360., 1)


def test_naive_clock_is_preserved_and_invalid_or_numeric_time_is_not_guessed():
    result = call(table())
    assert result['out_1'].timestamp.dt.tz is None
    with pytest.raises(ValueError):
        call(table().assign(timestamp=range(9)))
    frame = table()
    frame.loc[0, 'timestamp'] = pd.NaT
    with pytest.raises(ValueError, match='Invalid timestamps'):
        call(frame)


def test_empty_schema_and_complete_rejection_are_usable_downstream():
    empty = call(table().iloc[:0])
    assert empty['out_1'].empty and empty['out_2'].empty
    assert list(empty['out_1']) == list(table()) + ['source_row_position', 'window_id']
    rejected = call(table().assign(x=np.inf))
    assert rejected['out_1'].empty
    assert rejected['out_2'].missing_scalar_values.sum() == 9


def test_generated_standalone_matches_common_module_without_project_imports():
    path = ROOT / 'dataflow_operators/data_processing/process_window_table.py'
    spec = importlib.util.spec_from_file_location('standalone_window_bridge', path)
    standalone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone)
    syntax = ast.parse(path.read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or '').startswith('common_algorithms')
                   for node in ast.walk(syntax))
    expected = call(table())
    actual = standalone.execute(table(), ['x'], ['device', 'record', 'split'], 360., 4)
    for port in expected:
        pd.testing.assert_frame_equal(actual[port], expected[port])
