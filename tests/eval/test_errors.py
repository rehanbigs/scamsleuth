import pandas as pd
import pytest

from scamsleuth.eval import errors
from scamsleuth.models.baseline import logreg_baseline


@pytest.mark.parametrize(
    ("label", "predicted", "expected"),
    [
        ("smishing", "smishing", "correct"),
        ("smishing", "ham", "missed_scam"),
        ("ham", "smishing", "false_alarm"),
        ("spam", "smishing", "spam_as_smishing"),
        ("spam", "ham", "spam_as_ham"),
    ],
)
def test_error_type(label: str, predicted: str, expected: str) -> None:
    assert errors.error_type(label, predicted) == expected


def test_collect_errors_defangs_and_keeps_only_mistakes() -> None:
    train = pd.DataFrame(
        {
            "text": [
                "Pay the fee at http://parcel.top now",
                "See you at six",
                "Big sale txt SALE to 84025",
            ]
            * 5,
            "label": ["smishing", "ham", "spam"] * 5,
        }
    )
    model = logreg_baseline().fit(train["text"], train["label"])
    frame = pd.DataFrame(
        {
            "text": ["Verify your account at http://evil.top", "Lunch at http://cafe.com/menu?"],
            "label": ["ham", "ham"],  # deliberately mislabelled to force false alarms
        }
    )
    out = errors.collect_errors(model, threshold=0.0, frame=frame)

    assert set(out["error"]) == {"false_alarm"}
    assert out["text"].str.contains("hxxp://").all()
    assert not out["text"].str.contains("http://").any()
