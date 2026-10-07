import numpy as np
import pytest

from scamsleuth.eval import metrics

Y = np.array(["ham"] * 100 + ["smishing"] * 10 + ["spam"] * 5)


def _scores() -> np.ndarray:
    ham = np.linspace(0.0, 0.5, 100)  # highest ham scores: 0.5, 0.4949...
    smishing = np.linspace(0.3, 1.0, 10)
    spam = np.full(5, 0.9)
    return np.concatenate([ham, smishing, spam])


def test_threshold_at_fpr_respects_constraint_and_maximises_recall() -> None:
    scores = _scores()
    threshold = metrics.threshold_at_fpr(Y, scores, max_fpr=0.02)
    rates = metrics.rates_at_threshold(Y, scores, threshold)

    assert rates["fpr"] <= 0.02
    assert rates["fpr"] == pytest.approx(0.02)  # uses the full allowance
    lower = metrics.rates_at_threshold(Y, scores, np.nextafter(threshold, -np.inf))
    assert lower["fpr"] > 0.02  # any lower threshold breaks the constraint


def test_threshold_at_fpr_handles_ties_and_edge_cases() -> None:
    y = np.array(["ham"] * 4 + ["smishing"])
    tied = np.array([0.5, 0.5, 0.5, 0.1, 0.9])
    threshold = metrics.threshold_at_fpr(y, tied, max_fpr=0.25)
    assert metrics.rates_at_threshold(y, tied, threshold)["fpr"] == 0.0  # cannot flag 1 of 3 ties

    assert metrics.threshold_at_fpr(y, tied, max_fpr=1.0) == -np.inf
    with pytest.raises(ValueError, match="at least one ham"):
        metrics.threshold_at_fpr(np.array(["smishing"]), np.array([0.5]), 0.02)


def test_rates_ignore_spam_in_false_alarm_rate() -> None:
    scores = _scores()
    rates = metrics.rates_at_threshold(Y, scores, 0.85)
    assert rates["fpr"] == 0.0  # all spam is flagged, but spam is not a false alarm
    assert rates["recall"] == pytest.approx(0.2)  # 0.922 and 1.0 clear 0.85
    assert rates["precision"] == pytest.approx(2 / 7)


def test_recall_ci_brackets_point_estimate() -> None:
    scores = _scores()
    low, high = metrics.recall_ci(Y, scores, 0.6)
    point = metrics.rates_at_threshold(Y, scores, 0.6)["recall"]
    assert low <= point <= high
    assert high - low > 0


def test_expected_calibration_error() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    calibrated = rng.uniform(size=p.size) < p
    assert metrics.expected_calibration_error(calibrated, p) < 0.02
    assert metrics.expected_calibration_error(calibrated, np.full(p.size, 0.99)) > 0.4


def test_predict_with_threshold_falls_back_to_best_other_class() -> None:
    classes = ["ham", "smishing", "spam"]
    scores = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6], [0.5, 0.45, 0.05]])
    assert list(metrics.predict_with_threshold(scores, classes, 0.4)) == [
        "ham",
        "spam",
        "smishing",
    ]


def test_evaluate_reports_all_fields() -> None:
    classes = ["ham", "smishing", "spam"]
    y = np.array(["ham", "ham", "smishing", "spam"])
    scores = np.array([[0.9, 0.05, 0.05], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    report = metrics.evaluate(y, scores, classes, threshold=0.5)

    assert report["recall"] == 1.0
    assert report["fpr"] == 0.0
    assert report["pr_auc"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["confusion"]["matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
