"""Shared paths and frozen-input checks; no implicit dataset access."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
FIT=WORK/'final_model_fit_v1'
sys.path.insert(0,str(FIT))
from data import read,write,digest,import_file,raw_windows,tabular_features,features,risk_levels
from inference import load_model,predict_array,predict_scene
D1=WORK/'RCSD-1YD Data Set/dataset_v1/data'
D2=WORK/'RCSD-1YD Data Set/controlled_dataset_v2/data'
ST=WORK/'early_warning_v2/stress_test/results'
PROTOCOL=WORK/'final_evaluation_protocol_v1/protocol.json'

def check_frozen():
    from portable import verify_assets
    verify_assets(WORK)

def assert_release():
    check_frozen()


def warning_module():
    return import_file('frozen_warning_pipeline',WORK/'warning_matched_baselines_v1/experiment.py')
