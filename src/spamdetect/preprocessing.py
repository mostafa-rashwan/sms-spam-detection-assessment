"""The single, deterministic text-cleaning function used for training and inference."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_CURRENCY = re.compile(r"[£$€₹¥]\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?p\b")
_PHONE = re.compile(r"\+?\d{5,}|\+?\d{2,5}[\s-]\d{3,}(?:[\s-]\d{2,})*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_SPACE = re.compile(r"\s+")

PLACEHOLDERS = ("urltoken", "emailtoken", "phonetoken", "currencytoken", "numbertoken")


def preprocess(text: str) -> str:
    """Normalise ``text``: NFKC, lowercase, replace URLs/emails/phones/currency/numbers by tokens."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _EMAIL.sub(" emailtoken ", text)
    text = _URL.sub(" urltoken ", text)
    text = _CURRENCY.sub(" currencytoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    text = _NUMBER.sub(" numbertoken ", text)
    return _SPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class Preprocessor:
    """Picklable callable: ``preprocess`` plus optional stop-word removal / Porter stemming."""

    remove_stopwords: bool = False
    stem: bool = False

    def __call__(self, text: str) -> str:
        out = preprocess(text)
        if not (self.remove_stopwords or self.stem):
            return out
        words = out.split()
        if self.remove_stopwords:
            words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        if self.stem:
            from nltk.stem import PorterStemmer  # optional dependency, no corpora needed

            stemmer = PorterStemmer()
            words = [w if w in PLACEHOLDERS else stemmer.stem(w) for w in words]
        return " ".join(words)
