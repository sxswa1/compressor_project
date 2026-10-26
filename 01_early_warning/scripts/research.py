"""Build, fit and evaluate the RCSD warning / DI / risk / RUL experiment."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
REPO=PROJECT.parent
ROOT=PROJECT/'experiments/rcsd_v1'
sys.path.insert(0,str(ROOT))
from portable import digest,read,write,build_auxiliary,seal_assets,verify_assets


def run(relative,*args):
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(REPO),str(PROJECT),env.get('PYTHONPATH','')])
    env['PYTHONUTF8']='1'
    logs=PROJECT/'outputs/reproduction_logs';logs.mkdir(parents=True,exist_ok=True)
    logfile=logs/(relative.replace('/','_')+'.txt')
    command=[sys.executable,'-u',str(ROOT/relative),*map(str,args)]
    print('Running',relative,flush=True)
    with logfile.open('a',encoding='utf-8') as log:
        with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace') as proc:
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=proc.wait()
    if code:raise subprocess.CalledProcessError(code,command)


def prepare_tables():
    out=ROOT/'final_evaluation_protocol_v1/results';out.mkdir(parents=True,exist_ok=True)
    for name in ['model_asset_plan.csv','reserved_stress_blocks.csv']:
        shutil.copy2(PROJECT/'configs/rcsd_v1'/name,out/name)


def prepare(source):
    if source is None:raise ValueError('prepare needs --source pointing to the original RCSD workbook')
    expected=read(ROOT/'RCSD-1YD Data Set/dataset_v1/config.json')['source_sha256']
    if digest(source)!=expected:raise ValueError('Source workbook SHA-256 differs from the recorded RCSD release')
    for directory in ['dataset_v1','controlled_dataset_v2']:
        out=ROOT/'RCSD-1YD Data Set'/directory/'data'
        if out.exists() and any(out.iterdir()):raise FileExistsError(f'Data already exists: {out}')
    target=ROOT/'RCSD-1YD Data Set/Centrifugal compressor _2022.xlsx'
    shutil.copy2(source,target)
    prepare_tables()
    run('RCSD-1YD Data Set/dataset_v1/build_dataset.py')
    run('RCSD-1YD Data Set/dataset_v1/verify_dataset.py')
    run('RCSD-1YD Data Set/controlled_dataset_v2/build_dataset.py')
    run('RCSD-1YD Data Set/controlled_dataset_v2/verify_dataset.py')
    build_auxiliary()


def warning():
    if (ROOT/'warning_lstm_budget_v1/results/bundle.pkl').exists():
        raise FileExistsError('Warning bundle exists; use a new experiment copy')
    for v in [1,2,3]:run(f'early_warning_v{v}/train.py')
    run('train_warning.py')


def evaluate(split,output):
    verify_assets()
    if split=='test':
        directory=ROOT/'final_heldout_dataset_v1/data'
        if not directory.exists():run('final_evaluation_pipeline_v1/generate.py','--reserved')
        run('final_evaluation_pipeline_v1/verify_reserved_dataset.py')
    default=ROOT/'final_evaluation_pipeline_v1'/('reserved_results' if split=='test' else 'development_results')
    output=output.resolve() if output else default
    run('final_evaluation_pipeline_v1/evaluate.py','--split',split,'--output',output)
    run('final_evaluation_pipeline_v1/check_metrics.py','--output',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train-warning','train-regression','seal','evaluate'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.source)
    elif args.action=='train-warning':warning()
    elif args.action=='train-regression':
        prepare_tables();run('final_model_fit_v1/run_fit.py')
    elif args.action=='seal':seal_assets()
    elif args.action=='evaluate':evaluate(args.split,args.output)


if __name__=='__main__':main()
