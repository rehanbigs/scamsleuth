import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from scamsleuth.models import train

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
        rows += [(t.format(i=i), "smishing", "imc25", "banking") for t in SCAMS]
        rows += [(t.format(i=i), "spam", "mendeley", "") for t in SPAM]
        rows += [(t.format(i=i), "ham", "mendeley", "") for t in HAM]
    df = pd.DataFrame(rows, columns=["text", "label", "source", "scam_type"])
    df["language"] = "english"
    df["group"] = range(len(df))
    return df


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    _frame(15, 0).to_parquet(processed / "train.parquet", index=False)
    for name, offset in (("val", 100), ("test", 200), ("hardneg", 300), ("multilingual", 400)):
        _frame(5, offset).to_parquet(processed / f"{name}.parquet", index=False)

    monkeypatch.setattr(
        train,
        "SEARCH",
        {
            "logreg": [
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000) for c in (1, 10)
            ],
            "linear_svm": [LinearSVC(C=1.0, class_weight="balanced")],
        },
    )
    monkeypatch.setattr(train, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data", tmp_path / "models", tmp_path / "reports"


def test_run_selects_calibrates_and_reports(dirs: tuple[Path, Path, Path]) -> None:
    data_dir, models_dir, reports_dir = dirs
    report = train.run(data_dir, models_dir, reports_dir)

    selection = report["selection"]
    assert selection["family"] in {"logreg", "linear_svm"}
    assert selection["calibration"] in {"uncalibrated", "sigmoid", "isotonic"}
    assert len(selection["search"]) == 3

    assert report["test"]["overall"]["fpr"] <= 0.05
    assert set(report["test"]["by_scam_type"]) == {"banking"}
    assert {"rates", "by_language"} <= set(report["hardneg"])

    for figure in ("pr_curve.png", "reliability.png", "confusion_test.png"):
        assert (reports_dir / "figures" / figure).stat().st_size > 0
    metadata = json.loads((models_dir / "baseline.json").read_text())
    assert metadata["classes"] == ["ham", "smishing", "spam"]
    assert (models_dir / "baseline.joblib").exists()
