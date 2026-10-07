from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scamsleuth.data import prepare, split


def test_normalize_text_ignores_case_punctuation_spacing_and_mojibake() -> None:
    assert split.normalize_text("\tWIN a £100 prize!!  Call  now.") == "win a 100 prize call now"
    assert split.normalize_text("Win a �100 prize, call now") == "win a 100 prize call now"


def test_dedupe_exact_majority_label_wins() -> None:
    df = pd.DataFrame(
        {
            "text": ["Free prize! Call now", "free prize call now", "FREE PRIZE, call now", "hi"],
            "label": ["spam", "smishing", "smishing", "ham"],
        }
    )
    out, report = split.dedupe_exact(df)

    assert list(out["label"]) == ["smishing", "ham"]
    assert report.rows_in == 4
    assert report.rows_out == 2
    assert report.duplicate_groups == 1
    assert report.conflicting_groups == 1


def test_dedupe_exact_tie_goes_to_most_severe_label() -> None:
    df = pd.DataFrame(
        {"text": ["Claim your refund", "claim your refund."], "label": ["spam", "smishing"]}
    )
    out, _ = split.dedupe_exact(df)
    assert list(out["label"]) == ["smishing"]


def test_near_duplicate_groups_links_template_variants_only() -> None:
    template = (
        "URGENT! You have won a 1 week FREE membership in our £100,000 Jackpot! Txt CLAIM to {}"
    )
    texts = pd.Series(
        [
            template.format(81010),
            template.format(81011),
            "Running late, see you at the station around six",
        ]
    )
    groups = split.near_duplicate_groups(texts)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def _synthetic(n_groups: int = 140) -> tuple[pd.Series, np.ndarray]:
    rng = np.random.default_rng(0)
    group_labels = rng.choice(["ham", "spam", "smishing"], size=n_groups, p=[0.8, 0.1, 0.1])
    sizes = rng.integers(1, 4, size=n_groups)
    groups = np.repeat(np.arange(n_groups), sizes)
    return pd.Series(np.repeat(group_labels, sizes)), groups


def test_assign_splits_keeps_groups_together_and_is_deterministic() -> None:
    labels, groups = _synthetic()
    splits = split.assign_splits(labels, groups, seed=1)

    per_group = pd.Series(splits.to_numpy()).groupby(groups).nunique()
    assert (per_group == 1).all()
    assert set(splits) == {"train", "val", "test"}
    assert splits.equals(split.assign_splits(labels, groups, seed=1))


def test_assign_splits_preserves_class_balance() -> None:
    labels, groups = _synthetic()
    splits = split.assign_splits(labels, groups)
    overall = labels.value_counts(normalize=True)
    for name in ("train", "val", "test"):
        share = labels[splits == name].value_counts(normalize=True)
        assert abs(share["ham"] - overall["ham"]) < 0.1


def test_build_has_no_text_shared_across_splits(tmp_path: Path) -> None:
    rows = [
        f"ham,Hello friend number {i} how was the {w} today,No,No,No"
        for i, w in enumerate(["match", "exam", "trip", "movie", "class", "party", "game"] * 6)
    ]
    rows += [
        f"Smishing,Your parcel {i} is held pay fee at http://x{i}.top,yes,No,No" for i in range(14)
    ]
    rows += [f"spam,Mega sale {i} on shoes and bags this weekend only,No,No,No" for i in range(14)]
    csv = tmp_path / "raw.csv"
    csv.write_text("LABEL,TEXT,URL,EMAIL,PHONE\n" + "\n".join(rows) + "\n", encoding="utf-8")

    df, summary = prepare.build(csv)

    assert summary["near_duplicate_groups"] >= 1
    assert (df.groupby("group")["split"].nunique() == 1).all()
    assert sum(sum(counts.values()) for counts in summary["splits"].values()) == len(df)


def test_read_split_round_trips_and_rejects_unknown_split(tmp_path: Path) -> None:
    (tmp_path / "processed").mkdir()
    frame = pd.DataFrame({"text": ["hi"], "label": ["ham"]})
    frame.to_parquet(tmp_path / "processed" / "val.parquet", index=False)

    assert prepare.read_split("val", tmp_path).equals(frame)
    with pytest.raises(ValueError, match="unknown split"):
        prepare.read_split("dev", tmp_path)
