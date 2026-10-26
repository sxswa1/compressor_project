"""Compare a reproduced test run with the archived method-level metrics."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]


def compare(run,archive):
    specs={
        'regression_family_summary.csv':(['task','family','stratum'],['seeds','seed_mean','seed_min','seed_max']),
        'warning_event_summary.csv':(['source','stratum','method'],['scenes','backgrounds','paired_hits','paired_two_hour','no_paired_event','paired_late_under_2h','median_paired_lead']),
        'empirical_metrics.csv':(['method'],['valid_points','total_points','active_fraction','new_events']),
    }
    checks=[]
    for file,(keys,values) in specs.items():
        a=pd.read_csv(archive/file).set_index(keys).sort_index()
        b=pd.read_csv(run/file).set_index(keys).sort_index()
        assert a.index.is_unique and b.index.is_unique,file
        pd.testing.assert_index_equal(a.index,b.index)
        np.testing.assert_allclose(a[values],b[values],rtol=1e-7,atol=1e-8,equal_nan=True,err_msg=file)
        checks.append(dict(file=file,rows=len(a),columns=values))
    return dict(status='PASS',checks=checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'experiments/rcsd_v1/final_evaluation_pipeline_v1/reserved_results')
    parser.add_argument('--archive',type=Path,default=PROJECT/'results/2026-10-25/reserved_results')
    args=parser.parse_args();result=compare(args.run_dir,args.archive)
    (args.run_dir/'archive_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
