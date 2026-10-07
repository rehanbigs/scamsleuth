"""Figures for model reports: precision-recall curves, reliability diagrams, confusion matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.metrics import precision_recall_curve

from scamsleuth.eval.metrics import POSITIVE

plt.rcParams.update(
    {"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9}
)


def pr_curves(curves: Mapping[str, tuple[ArrayLike, ArrayLike]], path: Path) -> None:
    """One precision-recall curve per ``name -> (y_true, smishing_scores)``."""
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for name, (y_true, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(np.asarray(y_true) == POSITIVE, scores)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("recall (smishing)")
    ax.set_ylabel("precision")
    ax.set_ylim(0.5, 1.01)
    ax.legend(loc="lower left")
    ax.set_title("Precision-recall")
    _save(fig, path)


def reliability(
    curves: Mapping[str, tuple[ArrayLike, ArrayLike]], path: Path, n_bins: int = 10
) -> None:
    """Predicted probability vs observed frequency, per ``name -> (y_binary, prob)``."""
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect")
    edges = np.linspace(0, 1, n_bins + 1)
    for name, (y_binary, prob) in curves.items():
        y, p = np.asarray(y_binary, dtype=float), np.asarray(prob, dtype=float)
        bins = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        filled = [b for b in range(n_bins) if (bins == b).any()]
        ax.plot(
            [p[bins == b].mean() for b in filled],
            [y[bins == b].mean() for b in filled],
            marker="o",
            markersize=3,
            label=name,
        )
    ax.set_xlabel("predicted P(smishing)")
    ax.set_ylabel("observed share smishing")
    ax.legend(loc="upper left")
    ax.set_title("Reliability diagram")
    _save(fig, path)


def confusion(matrix: Sequence[Sequence[int]], labels: Sequence[str], path: Path) -> None:
    data = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(data, cmap="Blues")
    for (i, j), value in np.ndenumerate(data):
        color = "white" if value > data.max() / 2 else "black"
        ax.text(j, i, f"{value:,}", ha="center", va="center", color=color)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (test)")
    _save(fig, path)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
