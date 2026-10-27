"""Public P0/P1 behavior checks; no external datasets or diagnosis libraries."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, standalone=False):
    folder = 'dataflow_operators' if standalone else 'common_algorithms'
    path = ROOT / folder / 'data_processing' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'public_test_{folder}_{name}', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('standalone', [False, True])
def test_cleaning_preserves_fault_like_changes_and_input(standalone):
    df = pd.DataFrame({'x': [10., 10., 50., 50., 999., np.nan]}, index=[7] * 6)
    before = df.copy(deep=True)
    result = module('anomaly_detection', standalone).execute(df, 'x', outlier_method='none', range_limits={'x': [0., 100.]})
    pd.testing.assert_frame_equal(result['out_1'], before)
    pd.testing.assert_frame_equal(df, before)
    assert result['out_2']['x__outlier'].tolist() == [False, False, False, False, True, False]
    assert not result['out_2']['x__modified'].any()


@pytest.mark.parametrize('standalone', [False, True])
def test_denoising_never_crosses_missing_clock_gap_or_record(standalone):
    df = pd.DataFrame({'t': [0., 1., 10., 11., 12., 0., 1.], 'record': ['a'] * 5 + ['b'] * 2,
                       'x': [0., 0., 100., np.nan, 100., 200., 200.]})
    result = module('anomaly_detection', standalone).execute(df, 'x', 'record', 't',
        outlier_method='none', denoise='mean', window=3, expected_interval=1.)
    np.testing.assert_allclose(result['out_1']['x'], df['x'], equal_nan=True)
    assert result['out_2']['time_gap'].iloc[2]


@pytest.mark.parametrize('standalone', [False, True])
def test_offline_short_gaps_and_long_gaps(standalone):
    df = pd.DataFrame({'x': [np.nan, 0., np.nan, 2., np.nan, np.nan, 5., np.nan]})
    result = module('missing_value', standalone).execute(df, 'x', method='linear', max_gap=1)
    np.testing.assert_allclose(result['out_1']['x'], [np.nan, 0., 1., 2., np.nan, np.nan, 5., np.nan], equal_nan=True)
    assert result['out_2']['x__filled'].sum() == 1
    assert result['out_2']['filling_uses_future_context'].all()


@pytest.mark.parametrize('standalone', [False, True])
def test_fill_preserves_record_and_time_gap_boundaries(standalone):
    df = pd.DataFrame({'record': ['a'] * 4 + ['b'] * 2, 't': [0., 1., 20., 21., 0., 1.],
                       'x': [1., np.nan, np.nan, 9., np.nan, 100.]})
    result = module('missing_value', standalone).execute(df, 'x', 'record', 't', expected_interval=1., max_gap=2)
    assert result['out_1']['x'].iloc[[1, 2, 4]].isna().all()


@pytest.mark.parametrize('standalone', [False, True])
def test_causal_ffill_prefix_never_changes_with_future_rows(standalone):
    df = pd.DataFrame({'t': [0., 1., 2., 3., 4., 20.], 'x': [7., np.nan, np.nan, np.nan, 11., np.nan]})
    fn = module('missing_value', standalone).execute
    kwargs = {'value_columns': 'x', 'time_column': 't', 'method': 'ffill', 'causal': True,
              'max_gap': 2, 'expected_interval': 1.}
    result = fn(df, **kwargs)
    np.testing.assert_allclose(result['out_1']['x'], [7., 7., 7., np.nan, 11., np.nan], equal_nan=True)
    assert not result['out_2']['filling_uses_future_context'].any()
    for end in range(1, len(df) + 1):
        pd.testing.assert_frame_equal(fn(df.iloc[:end], **kwargs)['out_1'], result['out_1'].iloc[:end])


@pytest.mark.parametrize('standalone', [False, True])
def test_alignment_inserts_missing_rows_without_fabricating_values(standalone):
    df = pd.DataFrame({'timestamp': pd.to_datetime(['2000-01-01 00:00', '2000-01-01 00:12']),
                       'record': 'a', 'x': [1., 3.]})
    before = df.copy(deep=True)
    result = module('process_timeseries', standalone).execute(df, 'x', group_columns='record', interval_seconds=360.)
    np.testing.assert_allclose(result['out_1']['x'], [1., np.nan, 3.], equal_nan=True)
    assert result['out_2']['inserted_timestamp'].tolist() == [False, True, False]
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize('standalone', [False, True])
def test_explicit_snapping_duplicates_and_row_limit(standalone):
    df = pd.DataFrame({'timestamp': pd.to_datetime(['2000-01-01 00:00:00', '2000-01-01 00:06:01']), 'x': [1., 2.]})
    fn = module('process_timeseries', standalone).execute
    with pytest.raises(ValueError, match='Off-grid'):
        fn(df, 'x')
    result = fn(df, 'x', snap_tolerance_seconds=1.)
    assert result['out_2']['snapped_timestamp'].sum() == 1
    with pytest.raises(ValueError, match='Duplicate'):
        fn(pd.concat([df.iloc[:1], df.iloc[:1]]), 'x')
    with pytest.raises(ValueError, match='max_rows'):
        fn(df, 'x', snap_tolerance_seconds=1., max_rows=1)


@pytest.mark.parametrize('name', ['anomaly_detection', 'missing_value', 'process_timeseries'])
@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('bad_id', [None, '', '  ', np.inf])
def test_unknown_record_identifiers_are_rejected(name, standalone, bad_id):
    df = pd.DataFrame({'record': pd.Series([bad_id] * 3, dtype=object), 'x': [1., np.nan, 3.],
                       'timestamp': pd.date_range('2000-01-01', periods=3, freq='6min')})
    with pytest.raises(ValueError, match='group identifiers'):
        module(name, standalone).execute(df, 'x', group_columns='record', time_column='timestamp')


@pytest.mark.parametrize('name', ['anomaly_detection', 'missing_value', 'process_timeseries'])
@pytest.mark.parametrize('standalone', [False, True])
def test_empty_ports_are_explicit_and_preserve_schema(name, standalone):
    df = pd.DataFrame({'timestamp': pd.Series(dtype='datetime64[ns]'), 'x': pd.Series(dtype=float)})
    result = module(name, standalone).execute(df, 'x')
    assert set(result) == {'out_1', 'out_2'}
    assert all(isinstance(v, pd.DataFrame) and v.empty for v in result.values())
    assert list(result['out_1']) == list(df)


@pytest.mark.parametrize('name', ['anomaly_detection', 'missing_value', 'process_timeseries'])
def test_unknown_timezone_is_preserved_and_mixed_clocks_fail(name):
    fn = module(name).execute
    df = pd.DataFrame({'timestamp': ['2000-01-01 00:00', '2000-01-01 00:06'], 'x': [1., 2.]})
    result = fn(df, 'x', time_column='timestamp')
    if name == 'process_timeseries':
        assert result['out_1']['timestamp'].dt.tz is None
    else:
        pd.testing.assert_series_equal(result['out_1']['timestamp'], df['timestamp'])
    with pytest.raises(ValueError, match='Mixed naive'):
        fn(df.assign(timestamp=['2000-01-01 00:00', '2000-01-01 00:06+08:00']), 'x', time_column='timestamp')


def test_public_import_does_not_load_model_or_waveform_dependencies():
    code = '''
import importlib.abc, sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sklearn', 'scipy', 'pywt'}:
            raise AssertionError('Public P1 module tried to import ' + fullname)
sys.meta_path.insert(0, RejectOptional())
sys.path.insert(0, sys.argv[1])
from common_algorithms.data_processing import clean_timeseries, impute_timeseries, align_process_table
assert callable(clean_timeseries) and callable(impute_timeseries) and callable(align_process_table)
'''
    subprocess.run([sys.executable, '-I', '-c', code, str(ROOT)], check=True, cwd=ROOT.parent)


def test_standalone_files_need_no_project_import(tmp_path):
    code = '''
import importlib.abc, importlib.util, sys
class RejectProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'common_algorithms', 'torch', 'sklearn', 'scipy'}:
            raise AssertionError('Standalone operator tried to import ' + fullname)
sys.meta_path.insert(0, RejectProject())
for index, path in enumerate(sys.argv[1:]):
    spec=importlib.util.spec_from_file_location('operator_'+str(index), path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert callable(mod.execute)
'''
    files = [str(ROOT / f'dataflow_operators/data_processing/{name}.py')
             for name in ['anomaly_detection', 'missing_value', 'process_timeseries']]
    subprocess.run([sys.executable, '-I', '-c', code, *files], check=True, cwd=tmp_path)
