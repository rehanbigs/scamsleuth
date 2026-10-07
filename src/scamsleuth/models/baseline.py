"""Classical text classifiers: TF-IDF features with linear models.

Two complementary TF-IDF views are combined:

* **word 1-2 grams** capture scam phrasing ("claim your", "prize guaranteed");
* **character 2-5 grams** (within word boundaries) survive obfuscation such as
  ``fr€e`` or ``acc0unt`` and misspellings, which break word features.

Text is first canonicalised and has its URLs, phone numbers, emails, dates and short
codes replaced by placeholder tokens (see :func:`~scamsleuth.features.entities.mask_entities`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Final

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from scamsleuth.features.entities import mask_entities
from scamsleuth.features.lexical import lexical_features

POSITIVE = "smishing"


def prepare_texts(texts: Iterable[str]) -> list[str]:
    """Canonicalise, mask entities and lower-case each message."""
    return [mask_entities(text).lower() for text in texts]


def tfidf_features(*, min_df: int = 2, char_ngrams: tuple[int, int] = (2, 5)) -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df, sublinear_tf=True
                ),
            ),
        ]
    )


def text_pipeline(classifier: ClassifierMixin, **tfidf_kwargs: Any) -> Pipeline:
    """``prepare -> TF-IDF (word + char) -> classifier``."""
    return Pipeline(
        [
            ("prepare", FunctionTransformer(prepare_texts)),
            ("tfidf", tfidf_features(**tfidf_kwargs)),
            ("clf", classifier),
        ]
    )


def logreg_baseline(c: float = 10.0) -> Pipeline:
    """TF-IDF + logistic regression with class weights balanced against the imbalance."""
    return text_pipeline(LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def naive_bayes(alpha: float = 0.1) -> Pipeline:
    """TF-IDF + Complement Naive Bayes, the classic spam-filter reference point."""
    return text_pipeline(ComplementNB(alpha=alpha))


def linear_svm(c: float = 0.5) -> Pipeline:
    """TF-IDF + linear SVM. Outputs margins, not probabilities, until calibrated."""
    return text_pipeline(LinearSVC(C=c, class_weight="balanced"))


def lexical_lightgbm() -> Pipeline:
    """Gradient-boosted trees on hand-crafted red-flag features (no word features)."""
    return Pipeline(
        [
            ("features", FunctionTransformer(lexical_features)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    class_weight="balanced",
                    random_state=0,
                    verbose=-1,
                ),
            ),
        ]
    )


def stacked() -> StackingClassifier:
    """Blend the text model and the red-flag model with a logistic-regression meta-model.

    The meta-model is trained on out-of-fold probabilities, so it learns how far to trust
    each base model on messages that model has not seen.
    """
    return StackingClassifier(
        estimators=[("text", logreg_baseline()), ("lexical", lexical_lightgbm())],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000),
        stack_method="predict_proba",
        cv=5,
    )


CANDIDATES: Final[dict[str, Callable[[], Any]]] = {
    "naive_bayes": naive_bayes,
    "logreg": logreg_baseline,
    "linear_svm": linear_svm,
    "lightgbm_lexical": lexical_lightgbm,
    "stack_logreg_lightgbm": stacked,
}


def score_matrix(model: Any, texts: Iterable[str]) -> np.ndarray:
    """Per-class scores, columns ordered as ``model.classes_``."""
    texts = list(texts)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(texts))
    return np.asarray(model.decision_function(texts))


def smishing_scores(model: Any, texts: Iterable[str]) -> np.ndarray:
    """Score for the ``smishing`` class: a probability if available, else a margin."""
    return score_matrix(model, texts)[:, list(model.classes_).index(POSITIVE)]
