"""De-duplication and leakage-safe train/validation/test splitting.

Two kinds of duplicate are handled differently:

* **Exact duplicates** (identical after normalisation) are collapsed to one row.
  When copies disagree on the label, the majority label wins; ties go to the most
  severe class, because missing a scam costs more than a false alarm.
* **Near duplicates** (the same template with small edits, e.g. a different phone
  number) are kept, but grouped so that a whole group lands in a single split.
  Otherwise the model is tested on messages it has effectively already seen.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

from scamsleuth.data.load import LABELS

SPLITS: Final = ("train", "val", "test")

_SEVERITY: Final = {label: rank for rank, label in enumerate(LABELS)}
_NON_WORD = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonical form used to detect duplicates: case, punctuation and spacing removed."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("�", "")
    return _SPACE.sub(" ", _NON_WORD.sub(" ", text)).strip()


@dataclass(frozen=True)
class DedupeReport:
    rows_in: int
    rows_out: int
    duplicate_groups: int
    conflicting_groups: int


def _resolve_label(labels: pd.Series) -> str:
    counts = labels.value_counts()
    tied = counts[counts == counts.max()].index
    return str(max(tied, key=lambda label: _SEVERITY[label]))


def dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, DedupeReport]:
    """Collapse rows whose normalised text is identical, resolving label conflicts."""
    norm = df["text"].map(normalize_text)
    grouped = df.groupby(norm, sort=False)

    sizes = grouped.size()
    n_labels = grouped["label"].nunique()
    resolved = grouped["label"].agg(_resolve_label)

    out = df.loc[~norm.duplicated()].copy()
    out["label"] = norm.loc[out.index].map(resolved)
    report = DedupeReport(
        rows_in=len(df),
        rows_out=len(out),
        duplicate_groups=int((sizes > 1).sum()),
        conflicting_groups=int((n_labels > 1).sum()),
    )
    return out.reset_index(drop=True), report


def _shingles(text: str, k: int = 5) -> set[bytes]:
    norm = normalize_text(text)
    if len(norm) <= k:
        return {norm.encode()}
    return {norm[i : i + k].encode() for i in range(len(norm) - k + 1)}


def near_duplicate_groups(
    texts: pd.Series, *, threshold: float = 0.8, num_perm: int = 128, seed: int = 0
) -> np.ndarray:
    """Assign a group id to each text; texts with estimated Jaccard >= threshold share one.

    Uses MinHash LSH over character 5-gram shingles, then merges candidate pairs with
    union-find so that groups are transitive.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures: list[MinHash] = []
    for i, text in enumerate(texts):
        signature = MinHash(num_perm=num_perm, seed=seed)
        signature.update_batch(list(_shingles(text)))
        lsh.insert(i, signature)
        signatures.append(signature)

    parent = list(range(len(signatures)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, signature in enumerate(signatures):
        for j in lsh.query(signature):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[max(root_i, root_j)] = min(root_i, root_j)

    return np.array([find(i) for i in range(len(signatures))])


def assign_splits(
    labels: pd.Series, groups: np.ndarray, *, n_folds: int = 7, seed: int = 42
) -> pd.Series:
    """Stratified, group-aware split: one fold for test, one for validation, rest train.

    With the default 7 folds this gives roughly 71% / 14% / 14%.
    """
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split = pd.Series("train", index=labels.index, dtype=object)
    for fold, (_, held_out) in enumerate(folds.split(np.zeros(len(labels)), labels, groups)):
        if fold == 0:
            split.iloc[held_out] = "test"
        elif fold == 1:
            split.iloc[held_out] = "val"
    return split
