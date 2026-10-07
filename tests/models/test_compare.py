from pathlib import Path

import pandas as pd
import pytest

from scamsleuth.data.prepare import read_split
from scamsleuth.models import compare
from scamsleuth.models.baseline import CANDIDATES, logreg_baseline

SCAMS = [
    "Your parcel {i} is held, pay the fee at http://parcel{i}.top",
    "URGENT account {i} locked, verify now at bit.ly/sec{i}",
    "Tax refund {i} waiting, claim at http://gov-refund{i}.xyz",
]
SPAM = [
    "Mega sale {i} on shoes this weekend, txt SHOES to 84025",
    "Free ringtone {i}, txt TONE to 87066",
]
HAM = [
    "See you at {i} near the station",
    "Can you pick up milk {i} on the way?",
    "Running {i} min late",
]


def _frame(n: int, offset: int) -> pd.DataFrame:
    rows = []
    for i in range(offset, offset + n):
        rows += [
            (t.format(i=i), "smishing", "imc25" if k else "mendeley") for k, t in enumerate(SCAMS)
        ]
        rows += [(t.format(i=i), "spam", "mendeley") for t in SPAM]
        rows += [(t.format(i=i), "ham", "mendeley") for t in HAM]
    df = pd.DataFrame(rows, columns=["text", "label", "source"])
    df["group"] = range(len(df))
    return df


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    _frame(12, 0).to_parquet(processed / "train.parquet", index=False)
    _frame(6, 100).to_parquet(processed / "val.parquet", index=False)
    return tmp_path


def test_validation_report_has_overall_and_per_source(data_dir: Path) -> None:
    train, val = read_split("train", data_dir), read_split("val", data_dir)
    model = logreg_baseline().fit(train["text"], train["label"])
    report = compare.validation_report(model, val, max_fpr=0.1)

    assert set(report) == {"overall", "imc25", "mendeley"}
    assert report["overall"]["fpr"] <= 0.1
    assert report["overall"]["threshold"] == report["imc25"]["threshold"]


def test_run_logs_every_candidate(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compare, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(
        compare,
        "CANDIDATES",
        {"logreg": logreg_baseline, "nb": CANDIDATES["naive_bayes"]},
    )
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts relative to the working directory
    out = tmp_path / "comparison.json"

    table = compare.run(data_dir, out, max_fpr=0.1)

    assert list(table["model"]) == ["logreg", "nb", "logreg_trained_on_mendeley_only"]
    assert table["cv_pr_auc_mean"].iloc[:2].between(0, 1).all()
    assert out.exists()
