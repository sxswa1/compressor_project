"""Run a registered operator with shipped weights; no project imports needed.

Without --input this builds artificial numbers for interface demonstration,
not RCSD records and not labelled faults. No accuracy is calculated for them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESOURCE = ROOT / "02_few_shot_diagnosis/resources/rcsd_v2"


def artificial_input(contract):
    data = {}
    for i, channel in enumerate(contract["channels"]):
        quantity = channel["quantity"]
        baseline = {"temperature": 70., "pressure": 5., "differential_pressure": 1.,
                    "vibration": 20., "axial": 0., "speed": 7000.}.get(quantity, 1.)
        data[channel["name"]] = baseline + np.linspace(0., .01, contract["window_points"]) * (i + 1)
    frame = pd.DataFrame(data)
    frame.insert(0, "timestamp", pd.date_range("2026-01-01", periods=len(frame),
                                             freq=pd.Timedelta(seconds=contract["interval_seconds"])))
    frame.insert(0, "window_id", "artificial_interface_demo_not_measured")
    return frame


def load_operator(path):
    # Mirrors platform on_run: code text loaded without a local __file__.
    namespace = {"__name__": "dataflow_on_run"}
    exec(compile(path.read_text(encoding="utf-8"), "<dataflow_on_run>", "exec"), namespace)
    return namespace["execute"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Long table with complete windows")
    parser.add_argument("--input-contract", type=Path, help="Required with --input; caller's actual physical schema")
    parser.add_argument("--method", default="meta_adapted_prototype_seed_ensemble")
    parser.add_argument("--k", type=int, choices=[1, 5, 10], default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "02_few_shot_diagnosis/outputs/demo_predictions.csv")
    args = parser.parse_args()
    if args.input and not args.input_contract:
        parser.error("With external input, explicitly provide --input-contract; do not infer field units from RCSD")
    declared = args.input_contract or RESOURCE / "input_contract.json"
    contract = json.loads(declared.read_text(encoding="utf-8"))
    manifest = json.loads((RESOURCE / "resource_manifest.json").read_text(encoding="utf-8"))
    key = f"k{args.k}_{args.method}"
    matches = [item for item in manifest["cases"] if item["id"] == key]
    if len(matches) != 1:
        parser.error("Choose one method/K from resources/rcsd_v2/resource_manifest.json")
    case = matches[0]
    path = RESOURCE / case["model"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
        raise ValueError("Model checksum differs from reviewed resource")
    frame = pd.read_csv(args.input, float_precision="round_trip") if args.input else artificial_input(contract)
    execute = load_operator(ROOT / "dataflow_operators/fault_diagnosis" / case["operator"])
    result = execute(df=frame, value_columns=[c["name"] for c in contract["channels"]],
                     model_path=str(path), input_contract=contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["out_1"].to_csv(args.output, index=False)
    print(json.dumps({"method": args.method, "k": args.k, "windows": len(result["out_1"]),
                      "artificial_interface_input": args.input is None, "accuracy_claimed": False,
                      "training_performed": False, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
