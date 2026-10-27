"""Fit all warning comparators with the recorded 150-epoch selection budget."""
from pathlib import Path
import sys
import time
import torch
from threadpoolctl import threadpool_limits
from portable import read,write

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'warning_matched_baselines_v1'))
import experiment as warning


def main():
    out=ROOT/'warning_lstm_budget_v1/results'
    if (out/'bundle.pkl').exists():raise FileExistsError('Warning bundle exists; use a new experiment copy')
    out.mkdir(parents=True,exist_ok=True)
    warning.OUT=out
    cfg=read(ROOT/'warning_matched_baselines_v1/plan.json')
    cfg['lstm']['max_epochs']=read(ROOT/'warning_lstm_budget_v1/plan.json')['max_epochs']
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    start=time.monotonic()
    with threadpool_limits(limits=2):
        raw,meta=warning.load_raw()
        warning.fit(raw,meta,cfg)
    write(out/'run_status.json',dict(status='TRAINED',seconds=time.monotonic()-start,config=cfg))


if __name__=='__main__':main()
