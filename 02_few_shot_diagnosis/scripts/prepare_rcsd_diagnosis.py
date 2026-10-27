"""Prepare the fixed RCSD semi-synthetic diagnosis study without fitting models."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '02_few_shot_diagnosis'))
from pipelines.rcsd_semisynthetic import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT.parent / '公开数据集/Centrifugal compressor _2022.xlsx')
    p.add_argument('--config', type=Path, default=ROOT / '02_few_shot_diagnosis/configs/rcsd_semisynthetic_v1.json')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'data/RCSD_semisynthetic_diagnosis_v1')
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(prepare(args.source, config, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
