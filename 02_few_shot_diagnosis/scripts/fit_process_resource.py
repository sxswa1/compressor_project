"""Fit a new physical-input RF/CNN resource from labelled support only."""
from pathlib import Path
import argparse,importlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--support-csv',type=Path,required=True)
    p.add_argument('--input-contract-json',type=Path,required=True)
    p.add_argument('--output-resource',type=Path,required=True)
    p.add_argument('--label-column',required=True)
    p.add_argument('--origin-record-column',required=True)
    p.add_argument('--support-source-kind',required=True,choices=['public_measured','physics_simulation','field_labelled'])
    p.add_argument('--method',default='random_forest',choices=['random_forest','cnn'])
    p.add_argument('--class-order-json',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=250)
    p.add_argument('--learning-rate',type=float,default=.002,help='CNN optimizer rate; RF does not use it')
    p.add_argument('--n-estimators',type=int,default=200)
    p.add_argument('--max-features',type=int,help='Support-only ANOVA selection of this many RF features')
    args=p.parse_args()
    contract=json.loads(args.input_contract_json.read_text(encoding='utf-8'))
    order=json.loads(args.class_order_json.read_text(encoding='utf-8')) if args.class_order_json else None
    module=importlib.import_module('02_few_shot_diagnosis.pipelines.process_fitting')
    torch.set_num_threads(2)
    report=module.fit_process_resource(pd.read_csv(args.support_csv,float_precision='round_trip'),args.output_resource,
        contract,args.label_column,args.origin_record_column,args.support_source_kind,args.method,
        seed=args.seed,steps=args.steps,n_estimators=args.n_estimators,max_features=args.max_features,class_order=order,
        learning_rate=args.learning_rate)
    print(json.dumps({'operation':report['operation'],'method':args.method,'class_counts':report['counts'],
        'query_records_used':0,'output_resource':str(args.output_resource)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
