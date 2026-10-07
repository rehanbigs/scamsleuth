from pathlib import Path

import pandas as pd
import pytest

from scamsleuth.data import prepare


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    words = ["match", "exam", "trip", "movie", "class", "party", "game"]
    rows = [f"ham,Hello friend {i} how was the {w} today,No,No,No" for i, w in enumerate(words * 6)]
    rows += [
        f"Smishing,Your parcel {i} is held pay at http://x{i}.top,yes,No,No" for i in range(14)
    ]
    rows += [f"spam,Mega sale {i} on shoes and bags this weekend,No,No,No" for i in range(14)]
    mendeley = tmp_path / "mendeley.csv"
    mendeley.write_text("LABEL,TEXT,URL,EMAIL,PHONE\n" + "\n".join(rows) + "\n", encoding="utf-8")

    imc = pd.DataFrame(
        {
            "text": [f"Bank alert {i}: account {i} suspended, verify <URL> now" for i in range(14)]
            + ["Su cuenta esta bloqueada, verifique aqui <URL> ahora mismo"],
            "scam_type": ["banking"] * 15,
            "language": ["English"] * 14 + ["Spanish"],
        }
    )
    imc_path = tmp_path / "imc.csv"
    imc.to_csv(imc_path, index=False)

    hardneg = pd.DataFrame(
        {
            "text": ["Your OTP is 4411, do not share", "KYC blocked, click http://x.xyz"],
            "label": ["safe", "scam"],
            "scam_type": ["none", "fake_kyc"],
            "language": ["en", "hinglish"],
        }
    )
    hardneg_path = tmp_path / "hardneg.csv"
    hardneg.to_csv(hardneg_path, index=False)
    return mendeley, imc_path, hardneg_path


def test_build_combines_sources_without_leakage(tmp_path: Path) -> None:
    mendeley, imc, _ = _write_sources(tmp_path)
    df, summary = prepare.build(mendeley, imc)

    assert set(df["source"]) == {"mendeley", "imc25"}
    assert (df["language"] == "english").all()
    assert (df.groupby("group")["split"].nunique() == 1).all()
    assert summary["rows_before_dedupe"]["imc25"]["smishing"] == 14
    total = sum(
        n for split in summary["splits"].values() for c in split.values() for n in c.values()
    )
    assert total == len(df)


def test_eval_sets_hold_non_english_and_synthetic_rows(tmp_path: Path) -> None:
    _, imc, hardneg = _write_sources(tmp_path)
    sets = prepare.build_eval_sets(imc, hardneg)

    assert list(sets["multilingual"]["language"]) == ["spanish"]
    assert list(sets["hardneg"]["label"]) == ["ham", "smishing"]
    assert list(sets["hardneg"].columns) == prepare.COLUMNS


def test_read_split_round_trips_and_rejects_unknown_split(tmp_path: Path) -> None:
    (tmp_path / "processed").mkdir()
    frame = pd.DataFrame({"text": ["hi"], "label": ["ham"]})
    for name in ("val", "hardneg"):
        frame.to_parquet(tmp_path / "processed" / f"{name}.parquet", index=False)

    assert prepare.read_split("val", tmp_path).equals(frame)
    assert prepare.read_split("hardneg", tmp_path).equals(frame)
    with pytest.raises(ValueError, match="unknown split"):
        prepare.read_split("dev", tmp_path)
