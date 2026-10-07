"""Compare candidate models with grouped cross-validation and log every run to MLflow.

For each candidate:

1. **5-fold grouped CV on train**: smishing PR-AUC, which is threshold-free and robust.
   Folds respect near-duplicate groups and are stratified by source and label.
2. **Fit on all of train, score validation**: recall at the false-alarm budget, overall and
   per source, plus macro F1.

A cross-source run trains on the classic corpus only, to measure how badly a model built on
2022 data misses modern scams. The test split is not used here.

Usage::

    uv run python -m scamsleuth.models.compare
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from scamsleuth.data.prepare import read_split
from scamsleuth.eval.metrics import POSITIVE, evaluate, threshold_at_fpr
from scamsleuth.models.baseline import CANDIDATES, logreg_baseline, score_matrix

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

EXPERIMENT = "scamsleuth-baselines"
TRACKING_URI = "sqlite:///mlflow.db"
MAX_FPR = 0.02


def grouped_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[Any]:
    strata = df["source"] + ":" + df["label"]
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(folds.split(df["text"], strata, df["group"]))


def cv_pr_auc(make_model: Callable[[], Any], df: pd.DataFrame, folds: list[Any]) -> list[float]:
    """Smishing PR-AUC on each held-out fold."""
    scores = []
    for fit_idx, held_idx in folds:
        fit, held = df.iloc[fit_idx], df.iloc[held_idx]
        model = make_model().fit(fit["text"], fit["label"])
        smishing = score_matrix(model, held["text"])[:, list(model.classes_).index(POSITIVE)]
        scores.append(float(average_precision_score(held["label"] == POSITIVE, smishing)))
    return scores


def validation_report(model: Any, val: pd.DataFrame, max_fpr: float) -> dict[str, Any]:
    """Overall and per-source metrics at the threshold that meets the false-alarm budget."""
    classes = list(model.classes_)
    scores = score_matrix(model, val["text"])
    threshold = threshold_at_fpr(val["label"], scores[:, classes.index(POSITIVE)], max_fpr)
    report = {"overall": evaluate(val["label"], scores, classes, threshold)}
    for source, rows in val.groupby("source"):
        mask = (val["source"] == source).to_numpy()
        report[str(source)] = evaluate(rows["label"], scores[mask], classes, threshold)
    return report


def _log(name: str, params: dict[str, Any], metrics: dict[str, float], report: Any) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
        mlflow.log_dict(report, "validation_report.json")


def run(data_dir: Path, out_path: Path, max_fpr: float = MAX_FPR) -> pd.DataFrame:
    train, val = read_split("train", data_dir), read_split("val", data_dir)
    folds = grouped_folds(train)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    rows = []
    experiments: list[tuple[str, Callable[[], Any], pd.DataFrame]] = [
        (name, make, train) for name, make in CANDIDATES.items()
    ]
    experiments.append(("logreg_trained_on_mendeley_only", logreg_baseline, train))
    for name, make_model, data in experiments:
        started = time.perf_counter()
        if name.endswith("mendeley_only"):
            data = data.loc[data["source"] == "mendeley"]
            cv = [float("nan")]
        else:
            cv = cv_pr_auc(make_model, data, folds)
        model = make_model().fit(data["text"], data["label"])
        report = validation_report(model, val, max_fpr)
        seconds = time.perf_counter() - started

        overall, imc, mendeley = report["overall"], report["imc25"], report["mendeley"]
        metrics = {
            "cv_pr_auc_mean": float(np.mean(cv)),
            "cv_pr_auc_std": float(np.std(cv)),
            "val_pr_auc": overall["pr_auc"],
            "val_recall": overall["recall"],
            "val_fpr": overall["fpr"],
            "val_macro_f1": overall["macro_f1"],
            "val_recall_imc25": imc["recall"],
            "val_recall_mendeley": mendeley["recall"],
            "seconds": seconds,
        }
        _log(name, {"model": name, "max_fpr": max_fpr, "train_rows": len(data)}, metrics, report)
        rows.append({"model": name, **metrics})
        print(f"{name:34} cv={metrics['cv_pr_auc_mean']:.3f} val_recall={overall['recall']:.3f}")

    table = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("reports/model_comparison.json"))
    parser.add_argument("--max-fpr", type=float, default=MAX_FPR)
    args = parser.parse_args(argv)
    print(run(args.data_dir, args.out, args.max_fpr).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
