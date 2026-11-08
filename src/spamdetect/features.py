"""Feature extraction transformers. Everything is fitted inside ``Pipeline.fit``."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .preprocessing import PLACEHOLDERS, Preprocessor, preprocess

META_NAMES = ["n_chars", "digit_ratio", "upper_ratio", "n_exclaim", *[f"n_{p}" for p in PLACEHOLDERS]]


class MetaFeatures(BaseEstimator, TransformerMixin):
    """Simple stylistic / count features computed from the raw text."""

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        rows = []
        for t in X:
            n = max(len(t), 1)
            tokens = preprocess(t).split()
            rows.append([
                len(t),
                sum(c.isdigit() for c in t) / n,
                sum(c.isupper() for c in t) / n,
                t.count("!"),
                *[tokens.count(p) for p in PLACEHOLDERS],
            ])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(META_NAMES, dtype=object)


def build_features(cfg: dict[str, Any], min_df: int = 1) -> FeatureUnion:
    """Build the word (+ optional char, meta) TF-IDF FeatureUnion from ``cfg``."""
    pre = Preprocessor(**cfg["preprocess"])
    f = cfg["features"]
    parts: list[tuple[str, Any]] = [
        ("word", TfidfVectorizer(
            ngram_range=tuple(f["word_ngram"]), sublinear_tf=True, min_df=min_df,
            preprocessor=pre, token_pattern=r"(?u)\b\w+\b")),
    ]
    if f.get("use_char", True):
        parts.append(("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=tuple(f["char_ngram"]), sublinear_tf=True,
            min_df=2, preprocessor=pre)))
    if f.get("use_meta", False):
        parts.append(("meta", Pipeline([("m", MetaFeatures()), ("s", MaxAbsScaler())])))
    return FeatureUnion(parts)
