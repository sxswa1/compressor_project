"""Check the exact uploaded on_run source, ten entries and 39 resources.

Optional --query-directory accepts four locally restored query_*.csv files;
it compares 13,728 predictions/scores with published frozen expectations.
The source-code load deliberately provides no __file__ or project imports.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "02_few_shot_diagnosis/resources/rcsd_v2"
sys.path.insert(0, str(ROOT / "02_few_shot_diagnosis/scripts"))
from run_rcsd_demo import artificial_input, load_operator


def check_ports(output):
    if set(output) != {"out_1", "out_2"} or not all(isinstance(v, pd.DataFrame) for v in output.values()):
        raise AssertionError("Expected out_1/out_2 DataFrames")


def reject(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid parameters/schema were silently accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-directory", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "02_few_shot_diagnosis/results/verification/upload_operator_checks.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    namespace = {}
    exec((ROOT / "dataflow_operators/registry.py").read_text(encoding="utf-8"), namespace)
    registry = namespace["OPERATORS"]
    if len(registry) != 10:
        raise AssertionError("Incomplete operator directory registration")
    operators = {}
    hashes = {}
    for name, entry in registry.items():
        path = ROOT / "dataflow_operators" / entry["path"]
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute"]
        if len(definitions) != 1:
            raise AssertionError("One unambiguous public execute required")
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("common_algorithms", "02_few_shot_diagnosis")):
                raise AssertionError("Operator depends on a local project module")
        execute = load_operator(path)
        sig = inspect.signature(execute)
        if next(iter(sig.parameters)) != "df" or "kwargs" not in sig.parameters or not execute.__doc__:
            raise AssertionError("Missing explicit table signature or documentation")
        operators[name] = execute
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    table = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=8, freq="6min"),
                          "device": "demo", "record": "a", "x": np.arange(8, dtype=float)})
    public = {
        "anomaly_detection": {"value_columns": ["x"], "group_columns": ["record"], "time_column": "timestamp"},
        "missing_value": {"value_columns": ["x"], "group_columns": ["record"], "time_column": "timestamp"},
        "process_timeseries": {"value_columns": ["x"], "group_columns": ["record"], "interval_seconds": 360.},
        "process_window_table": {"value_columns": ["x"], "group_columns": ["device", "record"],
                                  "interval_seconds": 360., "window_points": 4},
        "process_features": {"value_columns": ["x"], "window_id_column": "record", "interval_seconds": 360.},
    }
    for name, parameters in public.items():
        before = table.copy(deep=True)
        actual = operators[name](df=table, **parameters)
        check_ports(actual)
        empty = operators[name](df=table.iloc[:0], **parameters)
        check_ports(empty)
        if list(actual["out_1"]) != list(empty["out_1"]):
            raise AssertionError("Empty branch changed output schema: " + name)
        reject(lambda: operators[name](df=table.drop(columns="x"), **parameters))
        pd.testing.assert_frame_equal(before, table)
    contract = json.loads((RESOURCE / "input_contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESOURCE / "resource_manifest.json").read_text(encoding="utf-8"))
    if len(manifest["cases"]) != 39:
        raise AssertionError("Incomplete trained resources")
    columns = [c["name"] for c in contract["channels"]]
    example = artificial_input(contract)
    queries = None
    expected = None
    if args.query_directory:
        variants = ["main", "weak_lag_challenge", "mid_amplitude_long_lag", "very_weak_long_lag"]
        queries = pd.concat([pd.read_csv(args.query_directory / f"query_{v}.csv", float_precision="round_trip") for v in variants], ignore_index=True)
        expected = pd.read_csv(RESOURCE / "expected_resource_predictions.csv", float_precision="round_trip")
    matched = 0
    for case in manifest["cases"]:
        path = RESOURCE / case["model"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise AssertionError("Resource changed: " + case["id"])
        execute = operators[Path(case["operator"]).stem]
        params = {"value_columns": columns, "model_path": str(path), "input_contract": contract}
        before = example.copy(deep=True)
        check_ports(execute(df=example, **params))
        empty = execute(df=example.iloc[:0], **params)
        check_ports(empty)
        if not empty["out_1"].empty:
            raise AssertionError("Empty model input generated predictions")
        wrong = copy.deepcopy(contract)
        wrong["channels"][0]["unit"] = "incompatible-unit"
        reject(lambda: execute(df=example, **{**params, "input_contract": wrong}))
        broken = example.copy(deep=True)
        broken.loc[1, "timestamp"] = broken.loc[0, "timestamp"]
        reject(lambda: execute(df=broken, **params))
        pd.testing.assert_frame_equal(before, example)
        if queries is not None:
            actual = execute(df=queries, **params)["out_1"].set_index("window_id").sort_index()
            ref = expected[expected["case"].eq(case["id"])].set_index("window_id").sort_index()
            if len(actual) != 352 or actual.index.tolist() != ref.index.tolist():
                raise AssertionError("Query identities differ")
            if actual.predicted_label.tolist() != ref.predicted_label.tolist():
                raise AssertionError("Operator prediction differs: " + case["id"])
            np.testing.assert_allclose(actual.model_score, ref.model_score, rtol=0, atol=1e-6)
            matched += len(actual)
    report = {"operator_entries": 10, "source_loaded_without_file_variable": True,
              "standalone_project_imports": 0, "public_normal_empty_invalid_schema_checks": 5,
              "verified_model_resources": 39, "model_normal_empty_wrong_unit_duplicate_time_checks": 39,
              "predictions_and_scores_matched": matched, "query_replay_performed": queries is not None,
              "new_independent_fault_events": 0, "platform_database_registration_performed": False,
              "operator_file_sha256": hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
