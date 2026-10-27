"""公共算法库 —— 数据处理子模块。"""
from .anomaly_detection import clean_timeseries
from .missing_value import impute_timeseries
from .process_timeseries import align_process_table, process_windows
from importlib import import_module

_OPERATING_NAMES = {
    "DOMAIN_RPM", "domain_to_rpm", "estimate_shaft_freq", "identify_and_segment",
    "identify_condition", "rpm_to_domain", "sliding_windows",
}


def __getattr__(name):
    # DCS cleaning needs only numpy/pandas. Import scipy-based waveform
    # condition recognition only when its public functions are requested.
    if name in _OPERATING_NAMES:
        value = getattr(import_module(".operating_condition", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "clean_timeseries",
    "impute_timeseries",
    "align_process_table",
    "process_windows",
    "DOMAIN_RPM",
    "domain_to_rpm",
    "estimate_shaft_freq",
    "identify_and_segment",
    "identify_condition",
    "rpm_to_domain",
    "sliding_windows",
]
