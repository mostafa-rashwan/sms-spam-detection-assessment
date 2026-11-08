"""Candidate estimators, tuning grids and pipeline assembly."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_features

# Ordered simplest -> most complex (used for tie-breaking).
SIMPLICITY_ORDER = ["multinomial_nb", "complement_nb", "logreg", "linear_svm", "random_forest"]


def make_estimator(name: str, seed: int):
    """Return an unfitted classifier for ``name``."""
    if name == "multinomial_nb":
        return MultinomialNB()
    if name == "complement_nb":
        return ComplementNB()
    if name == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    if name == "linear_svm":
        return CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=seed), cv=3, method="sigmoid")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name: str, cfg: dict[str, Any], params: dict[str, Any] | None = None) -> Pipeline:
    """Assemble ``features -> clf`` and apply optional ``params`` (grid-style keys)."""
    pipe = Pipeline([
        ("features", build_features(cfg)),
        ("clf", make_estimator(name, cfg["seed"])),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


def make_grid(name: str, cfg: dict[str, Any]) -> dict[str, list]:
    """Hyper-parameter grid for ``name`` (min_df of the word vectoriser is always tuned)."""
    g = cfg["grids"]
    grid: dict[str, list] = {"features__word__min_df": list(g["min_df"])}
    if name in ("multinomial_nb", "complement_nb"):
        grid["clf__alpha"] = list(g[name]["alpha"])
    elif name == "logreg":
        grid["clf__C"] = list(g["logreg"]["C"])
    elif name == "linear_svm":
        grid["clf__estimator__C"] = list(g["linear_svm"]["C"])
    elif name == "random_forest":
        grid["clf__n_estimators"] = list(g["random_forest"]["n_estimators"])
    else:
        raise ValueError(f"Unknown model: {name}")
    return grid
