"""Train the final baseline: tune, calibrate, pick a threshold, evaluate once on test.

Steps (all model selection uses train and validation only):

1. **Tune** the regularisation strength ``C`` of the two strongest families (logistic
   regression, linear SVM) by 5-fold grouped CV on train, scoring smishing PR-AUC. TF-IDF
   is fitted once per fold and shared by every ``C``, which keeps the search cheap.
2. **Calibrate** the winner with Platt (sigmoid) and isotonic calibration, using
   out-of-fold predictions from the same grouped folds. Keep the method with the lowest
   expected calibration error (ECE) on validation.
3. **Threshold**: the lowest score that keeps false alarms on validation ``ham`` at or
   below 2%.
4. **Evaluate once** on test, per source and per scam type, then on the two
   evaluation-only sets: genuine look-alikes (``hardneg``) and non-English reports
   (``multilingual``).

Usage::

    uv run python -m scamsleuth.models.train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.svm import LinearSVC

from scamsleuth.data.prepare import read_split
from scamsleuth.eval import plots
from scamsleuth.eval.metrics import (
    POSITIVE,
    evaluate,
    expected_calibration_error,
    rates_at_threshold,
    threshold_at_fpr,
)
from scamsleuth.models.baseline import prepare_texts, score_matrix, text_pipeline, tfidf_features
from scamsleuth.models.compare import MAX_FPR, TRACKING_URI, grouped_folds

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

SEARCH: dict[str, list[Any]] = {
    "logreg": [
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000) for c in (1, 3, 10, 30, 100)
    ],
    "linear_svm": [LinearSVC(C=c, class_weight="balanced") for c in (0.1, 0.3, 1.0, 3.0)],
}


def _positive_scores(clf: Any, features: Any) -> np.ndarray:
    column = list(clf.classes_).index(POSITIVE)
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(features))[:, column]
    return np.asarray(clf.decision_function(features))[:, column]


def tune(train: pd.DataFrame, folds: list[Any]) -> pd.DataFrame:
    """Grouped-CV smishing PR-AUC for every (family, C) in :data:`SEARCH`."""
    texts = np.array(prepare_texts(train["text"]), dtype=object)
    labels = train["label"].to_numpy()
    results: dict[tuple[str, float], list[float]] = {}
    for fit_idx, held_idx in folds:
        tfidf = tfidf_features().fit(texts[fit_idx])
        x_fit, x_held = tfidf.transform(texts[fit_idx]), tfidf.transform(texts[held_idx])
        for family, configs in SEARCH.items():
            for config in configs:
                clf = clone(config).fit(x_fit, labels[fit_idx])
                score = average_precision_score(
                    labels[held_idx] == POSITIVE, _positive_scores(clf, x_held)
                )
                results.setdefault((family, config.C), []).append(float(score))
    rows = [
        {"family": f, "C": c, "cv_pr_auc_mean": np.mean(s), "cv_pr_auc_std": np.std(s)}
        for (f, c), s in results.items()
    ]
    return pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False, ignore_index=True)


def calibrate(
    best: pd.Series, train: pd.DataFrame, val: pd.DataFrame, folds: list[Any]
) -> tuple[Any, pd.DataFrame, dict[str, np.ndarray]]:
    """Fit sigmoid and isotonic calibration; return the model with the lowest val ECE.

    Also returns the comparison table and each variant's validation probabilities.
    """
    base = next(c for c in SEARCH[best["family"]] if best["C"] == c.C)
    candidates: dict[str, Any] = {}
    if hasattr(base, "predict_proba"):
        candidates["uncalibrated"] = text_pipeline(clone(base))
    for method in ("sigmoid", "isotonic"):
        candidates[method] = CalibratedClassifierCV(
            text_pipeline(clone(base)), method=method, cv=folds, ensemble=False
        )

    rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, np.ndarray]] = {}
    y_val = (val["label"] == POSITIVE).to_numpy()
    for name, model in candidates.items():
        model.fit(train["text"], train["label"])
        prob = score_matrix(model, val["text"])[:, list(model.classes_).index(POSITIVE)]
        fitted[name] = (model, prob)
        rows.append(
            {
                "calibration": name,
                "val_ece": expected_calibration_error(y_val, prob),
                "val_brier": brier_score_loss(y_val, prob),
                "val_pr_auc": average_precision_score(y_val, prob),
            }
        )
    table = pd.DataFrame(rows).sort_values("val_ece", ignore_index=True)
    probs = {name: prob for name, (_, prob) in fitted.items()}
    return fitted[str(table["calibration"].iloc[0])][0], table, probs


def _recall_by(frame: pd.DataFrame, flagged: np.ndarray, column: str) -> dict[str, Any]:
    out = {}
    for key, rows in frame.assign(flagged=flagged).groupby(column):
        positives = rows["label"] == POSITIVE
        out[str(key)] = {
            "n": len(rows),
            "smishing_recall": float(rows.loc[positives, "flagged"].mean())
            if positives.any()
            else None,
            "flagged_share": float(rows["flagged"].mean()),
        }
    return out


def final_evaluation(model: Any, threshold: float, sets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    classes = list(model.classes_)
    report: dict[str, Any] = {"threshold": threshold, "max_fpr": MAX_FPR}
    for name, frame in sets.items():
        scores = score_matrix(model, frame["text"])
        smishing = scores[:, classes.index(POSITIVE)]
        flagged = smishing >= threshold
        entry: dict[str, Any] = {"overall": evaluate(frame["label"], scores, classes, threshold)}
        if name == "test":
            for source, rows in frame.groupby("source"):
                mask = (frame["source"] == source).to_numpy()
                entry[str(source)] = evaluate(rows["label"], scores[mask], classes, threshold)
            imc = (frame["source"] == "imc25").to_numpy()
            entry["by_scam_type"] = _recall_by(frame[imc], flagged[imc], "scam_type")
        else:
            entry["rates"] = rates_at_threshold(frame["label"], smishing, threshold)
            entry["by_language"] = _recall_by(frame, flagged, "language")
        report[name] = entry
    return report


def run(data_dir: Path, models_dir: Path, reports_dir: Path) -> dict[str, Any]:
    train, val, test = (read_split(s, data_dir) for s in ("train", "val", "test"))
    folds = grouped_folds(train)

    search = tune(train, folds)
    best = search.iloc[0]
    print(search.round(4).to_string(index=False))

    model, calibration, val_probs = calibrate(best, train, val, folds)
    print(calibration.round(4).to_string(index=False))

    val_smishing = score_matrix(model, val["text"])[:, list(model.classes_).index(POSITIVE)]
    threshold = threshold_at_fpr(val["label"], val_smishing, MAX_FPR)
    sets = {
        "test": test,
        "hardneg": read_split("hardneg", data_dir),
        "multilingual": read_split("multilingual", data_dir),
    }
    report = final_evaluation(model, threshold, sets)
    report["selection"] = {
        "family": str(best["family"]),
        "C": float(best["C"]),
        "calibration": str(calibration["calibration"].iloc[0]),
        "search": search.to_dict(orient="records"),
        "calibration_table": calibration.to_dict(orient="records"),
        "train_rows": len(train),
    }

    figures = reports_dir / "figures"
    test_smishing = score_matrix(model, test["text"])[:, list(model.classes_).index(POSITIVE)]
    plots.pr_curves(
        {"validation": (val["label"], val_smishing), "test": (test["label"], test_smishing)},
        figures / "pr_curve.png",
    )
    y_val = (val["label"] == POSITIVE).to_numpy()
    plots.reliability(
        {name: (y_val, prob) for name, prob in val_probs.items()}, figures / "reliability.png"
    )
    confusion = report["test"]["overall"]["confusion"]
    plots.confusion(confusion["matrix"], confusion["labels"], figures / "confusion_test.png")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "baseline.joblib")
    metadata = {"threshold": threshold, "classes": list(model.classes_), **report["selection"]}
    (models_dir / "baseline.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("scamsleuth-baselines")
    with mlflow.start_run(run_name=f"final_{best['family']}"):
        mlflow.log_params({k: v for k, v in metadata.items() if not isinstance(v, list)})
        overall = report["test"]["overall"]
        mlflow.log_metrics(
            {
                "test_recall": overall["recall"],
                "test_fpr": overall["fpr"],
                "test_pr_auc": overall["pr_auc"],
                "test_macro_f1": overall["macro_f1"],
                "hardneg_fpr": report["hardneg"]["rates"]["fpr"],
                "hardneg_recall": report["hardneg"]["rates"]["recall"],
                "multilingual_recall": report["multilingual"]["rates"]["recall"],
            }
        )
        mlflow.log_artifacts(str(figures), "figures")
        mlflow.log_artifact(str(reports_dir / "baseline_metrics.json"))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    report = run(args.data_dir, args.models_dir, args.reports_dir)
    overall = report["test"]["overall"]
    print(
        f"test: recall={overall['recall']:.3f} fpr={overall['fpr']:.3f} "
        f"pr_auc={overall['pr_auc']:.3f} macro_f1={overall['macro_f1']:.3f} | "
        f"hardneg fpr={report['hardneg']['rates']['fpr']:.3f} | "
        f"multilingual recall={report['multilingual']['rates']['recall']:.3f}"
    )


if __name__ == "__main__":
    main()
