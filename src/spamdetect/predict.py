"""Inference API, explanation and CLI."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

log = logging.getLogger(__name__)


class SpamClassifier:
    """Loads the saved pipeline + metadata and scores messages."""

    def __init__(self, pipeline, metadata: dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold: float = float(metadata["threshold"])
        self.max_length: int = int(metadata.get("max_length", 1000))

    @classmethod
    def load(cls, model_dir: str | Path = "models") -> SpamClassifier:
        """Load ``spam_pipeline.joblib`` and ``metadata.json`` from ``model_dir``."""
        model_dir = Path(model_dir)
        pipe_path, meta_path = model_dir / "spam_pipeline.joblib", model_dir / "metadata.json"
        if not pipe_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Model artefacts not found in {model_dir}. Train first with "
                "`python -m spamdetect.train --config config.yaml`."
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        trained = metadata.get("sklearn_version")
        if trained and trained != sklearn.__version__:
            msg = (f"Model was trained with scikit-learn {trained} but {sklearn.__version__} "
                   "is installed; predictions may differ or fail.")
            log.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return cls(joblib.load(pipe_path), metadata)

    def _validate(self, text: Any) -> str:
        if not isinstance(text, str):
            raise ValueError(f"text must be a str, got {type(text).__name__}")
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")
        if len(text) > self.max_length:
            raise ValueError(f"text is {len(text)} chars; maximum is {self.max_length}")
        return text

    def predict_many(self, texts: list[str]) -> list[dict[str, Any]]:
        """Score a list of messages (all are validated first; invalid input raises ValueError)."""
        clean = [self._validate(t) for t in texts]
        if not clean:
            return []
        probs = self.pipeline.predict_proba(np.asarray(clean, dtype=object))[:, 1]
        return [{"label": "spam" if p >= self.threshold else "ham",
                 "spam_probability": float(p), "threshold": self.threshold} for p in probs]

    def predict(self, text: str) -> dict[str, Any]:
        """Return ``{"label", "spam_probability", "threshold"}`` for one message."""
        return self.predict_many([text])[0]

    def explain(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Top n-grams pushing the message towards spam (linear / NB models; else empty)."""
        text = self._validate(text)
        clf = self.pipeline.named_steps["clf"]
        weights = _spam_weights(clf)
        if weights is None:
            return []
        x = self.pipeline.named_steps["features"].transform(np.asarray([text], dtype=object))
        x = x.tocsr()
        names = self.pipeline.named_steps["features"].get_feature_names_out()
        contrib = x.multiply(weights).tocsr()
        idx, vals = contrib.indices, contrib.data
        order = np.argsort(-np.abs(vals))[:top_k]
        return [(str(names[idx[i]]), float(vals[i])) for i in order]


def _spam_weights(clf) -> np.ndarray | None:
    """Per-feature log-odds style weight towards the spam class, or None if unsupported."""
    if hasattr(clf, "coef_"):
        return np.ravel(clf.coef_)
    if hasattr(clf, "feature_log_prob_"):
        return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    if hasattr(clf, "calibrated_classifiers_"):
        coefs = [np.ravel(c.estimator.coef_) for c in clf.calibrated_classifiers_]
        return np.mean(coefs, axis=0)
    return None


def main() -> None:
    """CLI: ``python -m spamdetect.predict "message"`` or ``--file messages.csv``."""
    ap = argparse.ArgumentParser(description="Classify SMS messages")
    ap.add_argument("message", nargs="?", help="a single message")
    ap.add_argument("--file", help="CSV with a 'text' column (or first column)")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()
    if not args.message and not args.file:
        ap.error("provide a message or --file")
    clf = SpamClassifier.load(args.model_dir)
    if args.file:
        df = pd.read_csv(args.file)
        texts = df["text"].astype(str).tolist() if "text" in df else df.iloc[:, 0].astype(str).tolist()
    else:
        texts = [args.message]
    for t, r in zip(texts, clf.predict_many(texts), strict=True):
        print(json.dumps({"text": t, **r}, ensure_ascii=False))


if __name__ == "__main__":
    main()
