"""Predict warning, DI, four-level risk and RUL from one device CSV."""
from pathlib import Path
import argparse
import sys
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipelines.research_inference import predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[20261023,20261024,20261025],required=True)
    p.add_argument('--di-model',choices=['constant','ridge_current','ridge_history','trees_current','trees_history','gradient_boosting_history','lstm_window'],default='lstm_window')
    p.add_argument('--rul-model',choices=['median','ridge_history','trees_history','gradient_boosting_history','lstm_window'],default='trees_history')
    p.add_argument('--rul-variant',choices=['raw_original','residual_original','raw_augmented','residual_augmented'],default='residual_augmented')
    p.add_argument('--warning-method',choices=['change_ewma','cusum','v3_core_limits'],default='change_ewma')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Output already exists')
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        out=predict(pd.read_csv(args.input),args.seed,args.di_model,args.rul_model,args.rul_variant,args.warning_method)
    args.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)


if __name__=='__main__':main()
