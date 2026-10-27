"""Restore study inputs from a user-obtained workbook without running failed trials.

No original or derived sensor observations are distributed in Git. The user
must obtain the RCSD workbook from its source under the applicable terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "02_few_shot_diagnosis"), str(Path(__file__).parent)]
from pipelines.rcsd_semisynthetic import prepare
from run_rcsd_optimization import extra_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "02_few_shot_diagnosis/configs/rcsd_semisynthetic_v1.json").read_text(encoding="utf-8"))
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("Wrong RCSD workbook SHA256")
    prepared = ROOT / "data/RCSD_semisynthetic_diagnosis_v1"
    if not prepared.exists():
        prepare(args.source, config, prepared)
    for name, digest in json.loads((prepared / "checksums.json").read_text(encoding="utf-8")).items():
        if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Previously prepared input changed: " + name)
    output = ROOT / "02_few_shot_diagnosis/outputs/rcsd_optimization_v2"
    if output.exists():
        raise FileExistsError("Keep existing outputs; use a clean checkout or move local outputs first")
    cfg = json.loads((ROOT / "02_few_shot_diagnosis/configs/rcsd_optimization_v2.json").read_text(encoding="utf-8"))
    x = np.load(prepared / "windows.npz", allow_pickle=False)["x"]
    manifest = pd.read_csv(prepared / "sample_manifest.csv")
    x, manifest = extra_queries(x, manifest, prepared, cfg, args.source)
    if len(x) != 520 or len(manifest) != 520:
        raise ValueError("Restored budget differs")
    output.mkdir(parents=True)
    np.savez_compressed(output / "all_windows.npz", x=x)
    manifest.to_csv(output / "sample_manifest.csv", index=False)
    columns = [c["name"] for c in json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))["input_contract"]["channels"]]
    for variant in manifest.loc[manifest.split.eq("query"), "variant"].unique():
        pieces = []
        for i, row in manifest[manifest.split.eq("query") & manifest.variant.eq(variant)].iterrows():
            part = pd.DataFrame(x[i].T, columns=columns)
            part.insert(0, "timestamp", pd.date_range(row.start_time, periods=20, freq="15min"))
            part.insert(0, "window_id", row.window_id)
            pieces.append(part)
        pd.concat(pieces, ignore_index=True).to_csv(output / f"query_{variant}.csv", index=False)
    # This only restores inputs; it never claims the rejected study was run.
    (output / "input_restoration.json").write_text(json.dumps({"windows": 520, "models_trained": 0,
        "failed_trial_results_reproduced": False, "source_sha256": config["source_sha256"]}, indent=2), encoding="utf-8")
    print(json.dumps({"restored_windows": 520, "query_windows": 352, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
