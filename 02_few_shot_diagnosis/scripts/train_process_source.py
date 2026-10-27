"""Train a reusable source CNN/ProtoNet resource without accepting final queries."""
from pathlib import Path
import argparse,importlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-csv',type=Path,required=True)
    p.add_argument('--input-contract-json',type=Path,required=True)
    p.add_argument('--output-resource',type=Path,required=True)
    p.add_argument('--label-column',required=True)
    p.add_argument('--origin-record-column',required=True)
    p.add_argument('--window-id-column',default='window_id')
    p.add_argument('--time-column',default='timestamp')
    p.add_argument('--source-kind',choices=['public_measured','physics_simulation','field_labelled'],required=True)
    p.add_argument('--method',choices=['cnn','protonet'],default='protonet')
    p.add_argument('--class-order-json',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=250)
    p.add_argument('--episodes',type=int,default=350)
    p.add_argument('--episode-support',type=int,default=2)
    p.add_argument('--episode-query',type=int,default=3)
    p.add_argument('--episode-ways',type=int)
    p.add_argument('--episode-grouping',choices=['single_class_records','global_records'],default='single_class_records',
        help='Default rejects mixed-class records; global mode separates whole records across every selected episode class')
    p.add_argument('--learning-rate',type=float,default=.002)
    args=p.parse_args()
    contract=json.loads(args.input_contract_json.read_text(encoding='utf-8'))
    order=json.loads(args.class_order_json.read_text(encoding='utf-8')) if args.class_order_json else None
    # Identifiers and class names are metadata, never numeric sensor features.
    # Preserve leading zeroes rather than silently merging caller IDs.
    dtypes={name:str for name in [args.window_id_column,args.origin_record_column,args.label_column]}
    table=pd.read_csv(args.source_csv,float_precision='round_trip',dtype=dtypes)
    module=importlib.import_module('02_few_shot_diagnosis.pipelines.process_source_training')
    torch.set_num_threads(2)
    report=module.train_process_source(table,args.output_resource,contract,args.label_column,args.origin_record_column,
        args.source_kind,method=args.method,window_id_column=args.window_id_column,time_column=args.time_column,
        seed=args.seed,steps=args.steps,episodes=args.episodes,episode_support=args.episode_support,
        episode_query=args.episode_query,episode_ways=args.episode_ways,learning_rate=args.learning_rate,class_order=order,
        episode_grouping=args.episode_grouping)
    print(json.dumps({'operation':report['operation'],'method':args.method,'classes':report['classes'],
        'source_counts':report['counts'],'final_query_records_used':0,
        'episode_global_record_separation':report['episode_global_record_separation'],
        'output_resource':str(args.output_resource)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
