"""Reproduce the nine original-support RF conditions without neural trials.

Reuses the exact frozen feature/augmentation arithmetic (levels=[1.0]);
no new observations, labels or query-based fitting are introduced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "02_few_shot_diagnosis"), str(Path(__file__).parent)]
from common_algorithms.models.process_cnn import ChannelStandardizer
from pipelines.rcsd_semisynthetic import support_tasks
from run_rcsd_support_optimization import train_target, scores_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "02_few_shot_diagnosis/outputs/rcsd_rf_replay")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve completed results; choose a new output directory")
    prepared = ROOT / "data/RCSD_semisynthetic_diagnosis_v1"
    initial = ROOT / "02_few_shot_diagnosis/outputs/rcsd_optimization_v2"
    preparation = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "02_few_shot_diagnosis/configs/rcsd_support_optimization_v2.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(initial / "sample_manifest.csv")
    x = np.load(initial / "all_windows.npz", allow_pickle=False)["x"]
    source, tasks = support_tasks(meta[meta.variant.isin(preparation["config"]["variants"])], preparation["config"])
    scaler = ChannelStandardizer().fit(x[source])
    sx = scaler.transform(x)
    y = meta.class_id.to_numpy(dtype=np.int64)
    g = meta.block_id.to_numpy()
    qi = np.flatnonzero(meta.split.eq("query"))
    contract = preparation["input_contract"]
    columns = [c["name"] for c in contract["channels"]]
    classes = preparation["config"]["class_names"]
    reference = pd.read_csv(ROOT / "02_few_shot_diagnosis/results/unsuccessful_trials/support_augmentation/metrics.csv")
    expected = pd.read_csv(ROOT / "02_few_shot_diagnosis/resources/rcsd_v2/expected_resource_predictions.csv", float_precision="round_trip")
    rows, matched = [], 0
    args.output_dir.mkdir(parents=True)
    for task in tasks:
        support = np.asarray(task["support_ids"])
        _, _, training, _ = train_target("random_forest", sx[support], y[support], g[support], [1.], 42, cfg, None, None)
        scores, forest, names = scores_target("random_forest", None, None, training, sx[qi], columns, contract["interval_seconds"], cfg)
        pred = forest.classes_[scores.argmax(axis=1)].astype(int)
        for variant in meta.loc[qi, "variant"].unique():
            positions = np.flatnonzero(meta.loc[qi, "variant"].to_numpy() == variant)
            truth, labels = y[qi[positions]], pred[positions]
            row = {"k": task["k"], "support_seed": task["support_seed"], "variant": variant,
                   "accuracy": float(accuracy_score(truth, labels)),
                   "macro_f1": float(f1_score(truth, labels, average="macro", zero_division=0)),
                   "confusion_matrix": json.dumps(confusion_matrix(truth, labels, labels=range(4)).tolist())}
            ref = reference[(reference.method == "random_forest_original_support") & (reference.k == task["k"])
                            & (reference.support_seed == task["support_seed"]) & (reference.variant == variant)]
            if len(ref) != 1:
                raise ValueError("Missing RF reference condition")
            for key in ["accuracy", "macro_f1"]:
                np.testing.assert_allclose(row[key], ref.iloc[0][key], rtol=0, atol=1e-12)
            if json.loads(row["confusion_matrix"]) != json.loads(ref.iloc[0].confusion_matrix):
                raise AssertionError("Reproduced RF confusion matrix differs")
            rows.append(row)
        if task["support_seed"] == 11:
            key = f"k{task['k']}_random_forest"
            actual = pd.DataFrame({"window_id": meta.loc[qi, "window_id"],
                                   "predicted_label": [classes[i] for i in pred], "model_score": scores.max(axis=1)}).set_index("window_id").sort_index()
            ref = expected[expected["case"].eq(key)].set_index("window_id").sort_index()
            if actual.predicted_label.tolist() != ref.predicted_label.tolist():
                raise AssertionError("Reproduced RF labels differ")
            np.testing.assert_allclose(actual.model_score, ref.model_score, rtol=0, atol=1e-12)
            matched += len(actual)
            joblib.dump({"model": forest, "feature_columns": names, "scaler": scaler.state(),
                         "process_contract": contract, "class_names": classes,
                         "provenance": {"semi_synthetic": True, "natural_fault_incidents": 0,
                                        "original_target_support_ids": support.tolist(), "query_fitting_rows": 0,
                                        "reproduced_from_frozen_original_support_implementation": True}},
                        args.output_dir / (key + ".joblib"))
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    report = {"metric_conditions_matched": len(rows), "fixed_resource_predictions_and_scores_matched": matched,
              "models_retrained": 9, "models_exported": 3, "source_label_windows": 124,
              "additional_observations": 0, "query_fitting_rows": 0}
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
