"""Classical text classifiers: TF-IDF features with linear models.

Two complementary TF-IDF views are combined:

* **word 1-2 grams** capture scam phrasing ("claim your", "prize guaranteed");
* **character 2-5 grams** (within word boundaries) survive obfuscation such as
  ``fr€e`` or ``acc0unt`` and misspellings, which break word features.

Text is first canonicalised and has its URLs, phone numbers, emails, dates and short
codes replaced by placeholder tokens (see :func:`~scamsleuth.features.entities.mask_entities`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from scamsleuth.features.entities import mask_entities

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


def smishing_scores(model: Any, texts: Iterable[str]) -> np.ndarray:
    """Score for the ``smishing`` class: a probability if available, else a margin."""
    texts = list(texts)
    column = list(model.classes_).index(POSITIVE)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(texts))[:, column]
    return np.asarray(model.decision_function(texts))[:, column]
