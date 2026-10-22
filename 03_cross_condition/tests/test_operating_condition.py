"""工况识别与片段切分（规则版）单元测试。"""
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ))

from common_algorithms.data_processing import (  # noqa: E402
    domain_to_rpm,
    identify_and_segment,
    identify_condition,
    rpm_to_domain,
    sliding_windows,
)


def test_rpm_to_domain():
    assert rpm_to_domain(1797) == 0
    assert rpm_to_domain(1772) == 1
    assert rpm_to_domain(1750) == 2
    assert rpm_to_domain(1730) == 3
    # 转速有波动时取最近档位
    assert rpm_to_domain(1785) == 0
    assert rpm_to_domain(1740) == 2


def test_domain_to_rpm():
    assert domain_to_rpm(0) == 1797.0
    assert domain_to_rpm(3) == 1730.0


def test_identify_by_domain():
    assert identify_condition(domain=2) == 2


def test_identify_by_rpm():
    assert identify_condition(rpm=1772) == 1


def test_sliding_windows_shape():
    x = np.zeros(10000)
    w = sliding_windows(x, 1024, 0.5)
    assert w.shape[0] > 0
    assert w.shape[1] == 1024


def test_identify_and_segment():
    x = np.zeros(10000)
    windows, conditions = identify_and_segment(x, sample_rate=12000,
                                               window_len=1024, overlap=0.5,
                                               rpm=1797)
    assert windows.shape[1] == 1024
    assert (conditions == 0).all()


if __name__ == "__main__":
    for fn in [test_rpm_to_domain, test_domain_to_rpm, test_identify_by_domain,
               test_identify_by_rpm, test_sliding_windows_shape,
               test_identify_and_segment]:
        fn()
        print(f"{fn.__name__} OK")
