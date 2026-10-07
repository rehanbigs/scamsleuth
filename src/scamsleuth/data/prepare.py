"""Build the processed dataset: download, validate, combine, de-duplicate, group and split.

Outputs (under ``data/processed``):

* ``train`` / ``val`` / ``test``: English messages from the Mendeley corpus and the IMC 2025
  smishing reports, split by near-duplicate group and stratified by source and label.
* ``multilingual``: non-English IMC 2025 reports, for evaluation only.
* ``hardneg``: synthetic scams and genuine look-alikes, for evaluation only.

Usage::

    uv run python -m scamsleuth.data.prepare
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import pandas as pd

from scamsleuth.data.external import download_hardneg, download_imc25, load_hardneg, load_imc25
from scamsleuth.data.load import download_raw, load_raw
from scamsleuth.data.split import SPLITS, assign_splits, dedupe_exact, near_duplicate_groups

EVAL_SETS: Final = ("multilingual", "hardneg")
COLUMNS: Final = ["text", "label", "source", "scam_type", "language"]


def combine(mendeley: pd.DataFrame, imc25: pd.DataFrame) -> pd.DataFrame:
    """Stack the English training sources into one frame with a ``source`` column."""
    parts = [
        mendeley.assign(source="mendeley", scam_type="", language="english"),
        imc25.loc[imc25["language"] == "english"].assign(source="imc25"),
    ]
    return pd.concat([part[COLUMNS] for part in parts], ignore_index=True)


def _counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    table = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    return {str(k): {str(c): int(v) for c, v in row.items()} for k, row in table.iterrows()}


def build(
    mendeley_csv: Path, imc25_csv: Path, *, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the combined frame (with ``group`` and ``split`` columns) and a summary."""
    combined = combine(load_raw(mendeley_csv), load_imc25(imc25_csv))
    df, report = dedupe_exact(combined)
    df["group"] = near_duplicate_groups(df["text"])
    strata = df["source"] + ":" + df["label"]
    df["split"] = assign_splits(strata, df["group"].to_numpy(), seed=seed)

    group_sizes = df["group"].value_counts()
    summary: dict[str, Any] = {
        "seed": seed,
        "rows_before_dedupe": _counts(combined),
        "dedupe": asdict(report),
        "near_duplicate_groups": int((group_sizes > 1).sum()),
        "rows_in_near_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
        "splits": {split: _counts(df.loc[df["split"] == split]) for split in SPLITS},
    }
    return df, summary


def build_eval_sets(imc25_csv: Path, hardneg_csv: Path) -> dict[str, pd.DataFrame]:
    """Evaluation-only sets that never touch training."""
    imc = load_imc25(imc25_csv)
    multilingual = imc.loc[imc["language"] != "english"].assign(source="imc25")
    hardneg = load_hardneg(hardneg_csv).assign(source="hardneg")
    return {
        "multilingual": dedupe_exact(multilingual[COLUMNS])[0],
        "hardneg": dedupe_exact(hardneg[COLUMNS])[0],
    }


def read_split(split: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load one processed split or evaluation set written by :func:`main`."""
    if split not in SPLITS + EVAL_SETS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS + EVAL_SETS}")
    return pd.read_parquet(data_dir / "processed" / f"{split}.parquet")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    raw_dir = args.data_dir / "raw"
    imc25_csv = download_imc25(raw_dir)
    df, summary = build(download_raw(raw_dir), imc25_csv, seed=args.seed)
    eval_sets = build_eval_sets(imc25_csv, download_hardneg(raw_dir))
    summary["eval_sets"] = {name: _counts(frame) for name, frame in eval_sets.items()}

    out_dir = args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        df.loc[df["split"] == split].drop(columns="split").to_parquet(
            out_dir / f"{split}.parquet", index=False
        )
    for name, frame in eval_sets.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
