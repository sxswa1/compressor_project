"""Switch among frozen RCSD resources without fitting or accepting field schemas."""
from pathlib import Path
import argparse
import importlib.util
import json

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description='Known semi-synthetic RCSD states, not confirmed mechanical faults')
    p.add_argument('--study-dir',type=Path,default=ROOT/'02_few_shot_diagnosis/outputs/rcsd_diagnosis_v1')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-contract',type=Path,required=True,help='Caller-declared actual units/quantities/cadence; must match the trained resource')
    p.add_argument('--method',default='random_forest')
    p.add_argument('--k',type=int,default=5)
    args=p.parse_args()
    cases=json.loads((args.study_dir/'resource_manifest.json').read_text())['cases']
    key=f'k{0 if args.method=="source_cnn" else args.k}_{args.method}'
    matches=[c for c in cases if c['id']==key]
    if len(matches)!=1:raise ValueError('Choose a declared method/K from resource_manifest.json')
    case=matches[0];contract=json.loads(args.input_contract.read_text(encoding='utf-8'))
    packaged=args.study_dir/'operators/fault_diagnosis'
    path=(packaged if packaged.exists() else ROOT/'dataflow_operators/fault_diagnosis')/case['operator']
    spec=importlib.util.spec_from_file_location('rcsd_selected_operator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frame=pd.read_csv(args.input,float_precision='round_trip')
    result=module.execute(frame,[c['name'] for c in contract['channels']],str(args.study_dir/case['model']),contract)
    args.output.parent.mkdir(parents=True,exist_ok=True);result['out_1'].to_csv(args.output,index=False)
    print(json.dumps({'method':args.method,'k':case['k'],'windows':len(result['out_1']),
        'semi_synthetic_known_states_only':True,'training_performed':False},indent=2))


if __name__=='__main__':main()
