"""Generate only the three reviewed P0/P1 operators from shared sources.

Run from any directory. This does not generate model operators or load data.
Each output is a standalone DataFlow file; edit shared source, then rebuild.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('anomaly_detection', 'missing_value', 'process_timeseries')


def main():
    clock = ROOT / 'common_algorithms/data_processing/process_clock.py'
    for name in NAMES:
        source = ROOT / f'common_algorithms/data_processing/{name}.py'
        body = source.read_text(encoding='utf-8')
        body = '\n'.join(line for line in body.splitlines()
                         if not line.startswith('from __future__ import ')
                         and not line.startswith('from common_algorithms.data_processing.process_clock import '))
        code = 'from __future__ import annotations\n\n' + clock.read_text(encoding='utf-8') + '\n\n' + body + '\n'
        digest = hashlib.sha256(code.encode('utf-8')).hexdigest()
        header = (f'# Generated from common_algorithms/data_processing/process_clock.py and {source.relative_to(ROOT).as_posix()}\n'
                  f'# Content SHA256: {digest}\n'
                  '# Dependencies: numpy==2.3.5, pandas==3.0.1; verified with Python 3.12.\n')
        target = ROOT / f'dataflow_operators/data_processing/{name}.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(header + code, encoding='utf-8', newline='\n')
        print(target.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
