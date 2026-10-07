import numpy as np
import pytest
from sklearn.svm import LinearSVC

from scamsleuth.models import baseline

TEXTS = [
    "Your parcel is on hold, pay the fee at http://parcel.top/pay",
    "URGENT your bank account is locked, verify at bit.ly/secure-login",
    "Claim your tax refund now at hxxp://refund-gov[.]xyz",
    "FREE ringtones every week txt TONE to 87066 150p/msg",
    "Win a £500 voucher, text WIN to 81010 now",
    "Mega sale on shoes this weekend only, txt SHOES to 84025",
    "See you at 6 near the station",
    "Can you pick up milk on the way home?",
    "Running late, start the meeting without me",
]
LABELS = ["smishing"] * 3 + ["spam"] * 3 + ["ham"] * 3


def test_prepare_texts_masks_and_lowercases() -> None:
    assert baseline.prepare_texts(["Pay at http://x.top NOW"]) == ["pay at  xxurl  now"]


def test_logreg_baseline_fits_and_scores_smishing_highest() -> None:
    model = baseline.logreg_baseline().fit(TEXTS, LABELS)
    scores = baseline.smishing_scores(model, TEXTS)

    assert scores.shape == (len(TEXTS),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[:3].min() > scores[3:].max()


def test_char_ngrams_generalise_to_obfuscated_spelling() -> None:
    model = baseline.logreg_baseline().fit(TEXTS, LABELS)
    plain, obfuscated, benign = baseline.smishing_scores(
        model,
        [
            "verify your bank account at http://x.top",
            "v3rify your b@nk acc0unt at http://x.top",
            "see you at the station",
        ],
    )
    assert obfuscated > benign
    assert obfuscated == pytest.approx(plain, abs=0.35)


def test_smishing_scores_falls_back_to_decision_function() -> None:
    model = baseline.text_pipeline(LinearSVC()).fit(TEXTS, LABELS)
    scores = baseline.smishing_scores(model, TEXTS)
    assert scores[:3].min() > scores[6:].max()


@pytest.mark.parametrize("name", sorted(baseline.CANDIDATES))
def test_every_candidate_fits_and_ranks_scams_above_chat(name: str) -> None:
    texts = TEXTS * 10  # enough rows for stacking's 5-fold CV and LightGBM's leaf minimum
    labels = LABELS * 10
    model = baseline.CANDIDATES[name]().fit(texts, labels)
    scores = baseline.smishing_scores(model, TEXTS)

    assert scores.shape == (len(TEXTS),)
    assert baseline.score_matrix(model, TEXTS).shape == (len(TEXTS), 3)
    assert scores[:3].mean() > scores[6:].mean()
