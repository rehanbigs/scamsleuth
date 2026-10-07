"""Metrics for an imbalanced, cost-asymmetric problem.

The headline metric is **smishing recall at a fixed false-alarm rate**:

* *recall* is the share of ``smishing`` messages flagged as smishing;
* the *false-alarm rate* (FPR) is the share of legitimate ``ham`` messages flagged as
  smishing. Spam flagged as smishing is a much cheaper mistake, so it is not counted.

A model outputs a score, not a decision. :func:`threshold_at_fpr` picks the decision
threshold on one split (validation), and the rates are then reported on another (test).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

POSITIVE: Final = "smishing"
NEGATIVE: Final = "ham"


def _split_scores(y_true: ArrayLike, scores: ArrayLike) -> tuple[NDArray[Any], NDArray[Any]]:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    return s[y == POSITIVE], s[y == NEGATIVE]


def threshold_at_fpr(y_true: ArrayLike, scores: ArrayLike, max_fpr: float) -> float:
    """Lowest threshold whose false-alarm rate on ``ham`` is at most ``max_fpr``.

    A message is flagged when ``score >= threshold``. The lowest such threshold maximises
    recall subject to the false-alarm constraint.
    """
    _, negatives = _split_scores(y_true, scores)
    if negatives.size == 0:
        raise ValueError("need at least one ham message to set a false-alarm threshold")
    allowed = int(np.floor(max_fpr * negatives.size))
    ranked = np.sort(negatives)[::-1]
    if allowed >= ranked.size:
        return float(-np.inf)
    # Just above the (allowed+1)-th highest ham score, so at most `allowed` ham are flagged.
    return float(np.nextafter(ranked[allowed], np.inf))


def rates_at_threshold(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> dict[str, float]:
    """Recall on smishing, false-alarm rate on ham, and precision of the smishing flag."""
    y = np.asarray(y_true)
    flagged = np.asarray(scores, dtype=float) >= threshold
    positives, negatives = y == POSITIVE, y == NEGATIVE
    return {
        "recall": float(flagged[positives].mean()) if positives.any() else float("nan"),
        "fpr": float(flagged[negatives].mean()) if negatives.any() else float("nan"),
        "precision": float(positives[flagged].mean()) if flagged.any() else float("nan"),
    }


def recall_ci(
    y_true: ArrayLike,
    scores: ArrayLike,
    threshold: float,
    *,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for recall at a fixed threshold."""
    positives, _ = _split_scores(y_true, scores)
    if positives.size == 0:
        return float("nan"), float("nan")
    hits = (positives >= threshold).astype(float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(hits, size=(n_boot, hits.size), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def expected_calibration_error(y_binary: ArrayLike, prob: ArrayLike, n_bins: int = 15) -> float:
    """Average gap between predicted probability and observed frequency, weighted by bin size.

    A calibrated model that says "0.9" is right about 90% of the time; its ECE is near 0.
    """
    y = np.asarray(y_binary, dtype=float)
    p = np.asarray(prob, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def predict_with_threshold(
    score_matrix: ArrayLike, classes: Sequence[str], threshold: float
) -> NDArray[np.str_]:
    """Three-way decision: ``smishing`` if its score clears the threshold, else best other."""
    scores = np.asarray(score_matrix, dtype=float)
    labels = np.asarray(classes)
    positive = list(classes).index(POSITIVE)
    others = np.delete(np.arange(len(classes)), positive)
    fallback = labels[others][scores[:, others].argmax(axis=1)]
    return np.where(scores[:, positive] >= threshold, POSITIVE, fallback)


def evaluate(
    y_true: ArrayLike, score_matrix: ArrayLike, classes: Sequence[str], threshold: float
) -> dict[str, Any]:
    """Full report for one split at a given threshold."""
    y = np.asarray(y_true)
    scores = np.asarray(score_matrix, dtype=float)
    smishing = scores[:, list(classes).index(POSITIVE)]
    predicted = predict_with_threshold(scores, classes, threshold)
    labels = sorted(set(y) | set(classes))
    low, high = recall_ci(y, smishing, threshold)
    return {
        "n": int(y.size),
        "threshold": threshold,
        **rates_at_threshold(y, smishing, threshold),
        "recall_ci95": [low, high],
        "pr_auc": float(average_precision_score(y == POSITIVE, smishing))
        if (y == POSITIVE).any() and (y != POSITIVE).any()
        else float("nan"),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "confusion": {
            "labels": labels,
            "matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
        },
        "per_class": classification_report(
            y, predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }
