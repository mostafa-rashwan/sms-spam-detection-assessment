"""Training orchestration: tuning, OOF threshold policy, model selection, single test evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import evaluate as ev
from .logging_utils import setup_logging
from .models import SIMPLICITY_ORDER, make_grid, make_pipeline

log = logging.getLogger(__name__)
TIE_TOLERANCE = 0.005  # recall differences below this are treated as ties -> prefer simpler model


def _oof(pipe: Pipeline, texts: np.ndarray, y: np.ndarray, skf: StratifiedKFold):
    """Out-of-fold spam scores plus the fold index of every row."""
    scores = np.zeros(len(y))
    folds = np.zeros(len(y), dtype=int)
    for k, (tr, va) in enumerate(skf.split(texts, y)):
        m = clone(pipe).fit(texts[tr], y[tr])
        scores[va] = m.predict_proba(texts[va])[:, 1]
        folds[va] = k
    return scores, folds


def _cv_report(y, scores, folds, max_fpr):
    """Pooled-OOF threshold under the FPR cap, applied per fold for mean +- std."""
    thr, _ = ev.threshold_at_fpr(y, scores, max_fpr)
    per_fold = [ev.compute_metrics(y[folds == k], scores[folds == k], thr)
                for k in np.unique(folds)]
    return {"threshold": thr, "pooled": ev.compute_metrics(y, scores, thr),
            "folds": ev.summarize_folds(per_fold)}


def _baseline_pipeline(seed: int) -> Pipeline:
    """Repo-B-style baseline: alpha-only text, stopwords + Porter stem, TF-IDF(3000), MultinomialNB."""
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stemmer = PorterStemmer()

    def prep(t: str) -> str:
        words = re.sub(r"[^a-zA-Z]", " ", t).lower().split()
        return " ".join(stemmer.stem(w) for w in words if w not in ENGLISH_STOP_WORDS)

    return Pipeline([
        ("features", TfidfVectorizer(max_features=3000, preprocessor=prep)),
        ("clf", MultinomialNB()),
    ])


def _tune(name, cfg, texts, y, skf):
    grid = make_grid(name, cfg)
    search = GridSearchCV(make_pipeline(name, cfg), grid, scoring="average_precision",
                          cv=skf, n_jobs=-1, refit=False)
    search.fit(texts, y)
    return search.best_params_, float(search.best_score_)


def run_training(cfg: dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame,
                 data_checksum: str = "unknown") -> dict[str, Any]:
    """Select a model + threshold on the training split, then evaluate once on the test split."""
    cfg = copy.deepcopy(cfg)
    seed, max_fpr = cfg["seed"], cfg["max_ham_fpr"]
    X_tr, y_tr = train_df["text"].to_numpy(), train_df["label"].to_numpy()
    X_te, y_te = test_df["text"].to_numpy(), test_df["label"].to_numpy()
    skf = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)

    # --- 1. tune + OOF evaluation of every candidate (training split only) ---
    cand: dict[str, dict[str, Any]] = {}
    for name in cfg["candidates"]:
        params, ap = _tune(name, cfg, X_tr, y_tr, skf)
        scores, folds = _oof(make_pipeline(name, cfg, params), X_tr, y_tr, skf)
        rep = _cv_report(y_tr, scores, folds, max_fpr)
        cand[name] = {"params": params, "grid_pr_auc": ap, "cv": rep}
        log.info("%s: OOF recall=%.4f fpr=%.4f params=%s", name,
                 rep["pooled"]["recall"], rep["pooled"]["ham_fpr"], params)

    # --- 2. selection: best OOF recall under the cap; near-ties -> simplest ---
    ok = {n: c for n, c in cand.items() if c["cv"]["pooled"]["ham_fpr"] <= max_fpr}
    pool = ok or cand
    best_recall = max(c["cv"]["pooled"]["recall"] for c in pool.values())
    near = [n for n, c in pool.items() if best_recall - c["cv"]["pooled"]["recall"] <= TIE_TOLERANCE]
    chosen = min(near, key=SIMPLICITY_ORDER.index)
    log.info("Selected model: %s", chosen)

    # --- 3. ablation of preprocessing / feature options on the chosen model ---
    ablation: list[dict[str, Any]] = []
    if cfg.get("ablation", True):
        base_rep = cand[chosen]["cv"]["pooled"]["recall"]
        toggles = [("preprocess", "remove_stopwords", True), ("preprocess", "stem", True),
                   ("features", "use_char", False), ("features", "use_meta", True)]
        for section, key, val in toggles:
            trial = copy.deepcopy(cfg)
            trial[section][key] = val
            params = {k: v for k, v in cand[chosen]["params"].items()
                      if not (k.startswith("features__char") and not trial["features"]["use_char"])}
            try:
                s, f = _oof(make_pipeline(chosen, trial, params), X_tr, y_tr, skf)
            except ImportError as exc:
                ablation.append({"change": f"{section}.{key}={val}", "skipped": str(exc)})
                continue
            rep = _cv_report(y_tr, s, f, max_fpr)
            gain = rep["pooled"]["recall"] - base_rep
            kept = gain > TIE_TOLERANCE and rep["pooled"]["ham_fpr"] <= max_fpr
            ablation.append({"change": f"{section}.{key}={val}", "recall": rep["pooled"]["recall"],
                             "ham_fpr": rep["pooled"]["ham_fpr"], "gain": gain, "kept": kept})
            if kept:
                cfg[section][key] = val
                base_rep = rep["pooled"]["recall"]
                cand[chosen]["cv"] = rep

    # --- 4. final OOF threshold (on possibly ablated config), refit, single test evaluation ---
    final = make_pipeline(chosen, cfg, cand[chosen]["params"])
    scores, folds = _oof(final, X_tr, y_tr, skf)
    cv_final = _cv_report(y_tr, scores, folds, max_fpr)
    threshold = cv_final["threshold"]
    final.fit(X_tr, y_tr)
    test_scores = final.predict_proba(X_te)[:, 1]
    test_m = ev.compute_metrics(y_te, test_scores, threshold)

    # --- 5. baseline (Repo-B-style) on the same split ---
    baseline: dict[str, Any] = {}
    try:
        bp = _baseline_pipeline(seed)
        b_oof, b_folds = _oof(bp, X_tr, y_tr, skf)
        b_thr = ev.threshold_at_fpr(y_tr, b_oof, max_fpr)[0]
        bp.fit(X_tr, y_tr)
        b_scores = bp.predict_proba(X_te)[:, 1]
        baseline = {
            "description": "TF-IDF(max_features=3000, unigrams, alpha-only, stopwords, Porter) + "
                           "MultinomialNB",
            "test_at_0.5": ev.compute_metrics(y_te, b_scores, 0.5),
            "test_at_fpr_cap_threshold": ev.compute_metrics(y_te, b_scores, b_thr),
        }
    except ImportError as exc:
        baseline = {"skipped": f"nltk not installed: {exc}"}

    metrics = {
        "config": {"seed": seed, "max_ham_fpr": max_fpr, "cv_folds": cfg["cv_folds"],
                   "preprocess": cfg["preprocess"], "features": cfg["features"]},
        "selected_model": chosen,
        "selected_params": cand[chosen]["params"],
        "threshold": threshold,
        "candidates": {n: {"params": c["params"], "cv_pooled": c["cv"]["pooled"],
                           "cv_folds": c["cv"]["folds"]} for n, c in cand.items()},
        "ablation": ablation,
        "cv": {"pooled": cv_final["pooled"], "folds": cv_final["folds"]},
        "test": test_m,
        "baseline": baseline,
        "split": {"train_n": len(y_tr), "test_n": len(y_te)},
    }

    # --- 6. artefacts ---
    models_dir, reports_dir = Path(cfg["paths"]["models_dir"]), Path(cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, models_dir / "spam_pipeline.joblib")
    metadata = {
        "model_name": chosen, "params": {k: str(v) for k, v in cand[chosen]["params"].items()},
        "threshold": threshold, "max_ham_fpr": max_fpr,
        "max_length": cfg["inference"]["max_length"],
        "cv_metrics": cv_final["pooled"], "test_metrics": test_m,
        "dataset": "UCI SMS Spam Collection v.1 (id 228), de-duplicated",
        "data_checksum_sha256": data_checksum,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        ev.plot_confusion_matrix(test_m, reports_dir / "confusion_matrix.png")
        ev.plot_pr_curve(y_te, test_scores, test_m, reports_dir / "pr_curve.png")
    except ImportError:
        log.warning("matplotlib missing; plots skipped")
    ev.error_table(X_te, y_te, test_scores, threshold).to_csv(
        reports_dir / "errors.csv", index=False)
    return metrics


def main() -> None:
    """CLI: ``python -m spamdetect.train --config config.yaml``."""
    from .config import load_config
    from .data import prepare_data

    setup_logging()
    ap = argparse.ArgumentParser(description="Train the SMS spam classifier")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_df, test_df, manifest = prepare_data(cfg)
    m = run_training(cfg, train_df, test_df, manifest["checksum_sha256"])
    print(json.dumps({"selected_model": m["selected_model"], "threshold": m["threshold"],
                      "test": m["test"]}, indent=2))


if __name__ == "__main__":
    main()
