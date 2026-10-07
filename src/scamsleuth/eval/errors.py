"""Collect a trained model's mistakes for manual error analysis.

Every message is defanged before it is written out, so a report never contains a
clickable scam link.

Usage::

    uv run python -m scamsleuth.eval.errors > errors.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scamsleuth.data.prepare import read_split
from scamsleuth.eval.metrics import NEGATIVE, POSITIVE, predict_with_threshold
from scamsleuth.features.entities import defang_text
from scamsleuth.models.baseline import score_matrix


def error_type(label: str, predicted: str) -> str:
    if label == predicted:
        return "correct"
    if label == POSITIVE:
        return "missed_scam"  # false negative: the costly error
    if predicted == POSITIVE:
        return "false_alarm" if label == NEGATIVE else "spam_as_smishing"
    return f"{label}_as_{predicted}"


def collect_errors(model: Any, threshold: float, frame: pd.DataFrame) -> pd.DataFrame:
    """All misclassified rows with their smishing score, sorted by how confident the error was."""
    classes = list(model.classes_)
    scores = score_matrix(model, frame["text"])
    predicted = predict_with_threshold(scores, classes, threshold)
    out = frame.assign(
        predicted=predicted,
        p_smishing=scores[:, classes.index(POSITIVE)],
        error=[error_type(y, p) for y, p in zip(frame["label"], predicted, strict=True)],
    )
    out = out.loc[out["error"] != "correct"].copy()
    out["text"] = out["text"].map(defang_text)
    confidence = np.where(out["label"] == POSITIVE, -out["p_smishing"], out["p_smishing"])
    return (
        out.assign(_c=confidence)
        .sort_values(["error", "_c"], ascending=[True, False])
        .drop(columns="_c")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    model = joblib.load(args.models_dir / "baseline.joblib")
    threshold = json.loads((args.models_dir / "baseline.json").read_text())["threshold"]
    errors = collect_errors(model, threshold, read_split(args.split, args.data_dir))
    columns = ["error", "label", "predicted", "p_smishing", "source", "scam_type", "text"]
    errors[columns].to_csv(sys.stdout, sep="\t", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
