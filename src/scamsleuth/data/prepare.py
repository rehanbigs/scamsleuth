"""Build the processed dataset: download, validate, de-duplicate, group and split.

Usage::

    uv run python -m scamsleuth.data.prepare            # writes to data/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from scamsleuth.data.load import download_raw, load_raw
from scamsleuth.data.split import SPLITS, assign_splits, dedupe_exact, near_duplicate_groups


def build(raw_csv: Path, *, seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the processed frame (with ``group`` and ``split`` columns) and a summary."""
    df, report = dedupe_exact(load_raw(raw_csv))
    df["group"] = near_duplicate_groups(df["text"])
    df["split"] = assign_splits(df["label"], df["group"].to_numpy(), seed=seed)

    group_sizes = df["group"].value_counts()
    summary: dict[str, Any] = {
        "seed": seed,
        "dedupe": asdict(report),
        "near_duplicate_groups": int((group_sizes > 1).sum()),
        "rows_in_near_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
        "splits": {
            split: df.loc[df["split"] == split, "label"].value_counts().to_dict()
            for split in SPLITS
        },
    }
    return df, summary


def read_split(split: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load one processed split written by :func:`main`."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return pd.read_parquet(data_dir / "processed" / f"{split}.parquet")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    raw_csv = download_raw(args.data_dir / "raw")
    df, summary = build(raw_csv, seed=args.seed)

    out_dir = args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        df.loc[df["split"] == split].drop(columns="split").to_parquet(
            out_dir / f"{split}.parquet", index=False
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
