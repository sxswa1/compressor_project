"""Turn compatible labelled support windows into a new fixed model resource."""
from pathlib import Path
import argparse,importlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-resource',type=Path,required=True)
    p.add_argument('--support-csv',type=Path,required=True)
    p.add_argument('--input-contract-json',type=Path,required=True,help='The exact channel/cadence/window contract JSON')
    p.add_argument('--output-resource',type=Path,required=True)
    p.add_argument('--label-column',required=True)
    p.add_argument('--origin-record-column',required=True,help='Original run/record ID; not an automatically generated sliding-window count')
    p.add_argument('--support-source-kind',required=True,choices=['public_measured','physics_simulation','field_labelled'])
    p.add_argument('--method',default='prototype',choices=['prototype','fine_tune','fine_tune_prototype'])
    p.add_argument('--class-order-json',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=250)
    p.add_argument('--learning-rate',type=float,default=.002,help='Fine-tuning rate; prototype-only construction does not use it')
    args=p.parse_args()
    module=importlib.import_module('02_few_shot_diagnosis.pipelines.process_adaptation')
    contract=json.loads(args.input_contract_json.read_text(encoding='utf-8'))
    order=json.loads(args.class_order_json.read_text(encoding='utf-8')) if args.class_order_json else None
    torch.set_num_threads(2)
    report=module.adapt_process_resource(pd.read_csv(args.support_csv,float_precision='round_trip'),args.source_resource,args.output_resource,
        contract,args.label_column,args.origin_record_column,args.support_source_kind,args.method,
        seed=args.seed,steps=args.steps,class_order=order,learning_rate=args.learning_rate)
    print(json.dumps({'operation':report['operation'],'class_counts':report['counts'],'query_records_used':0,
        'output_resource':str(args.output_resource)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
