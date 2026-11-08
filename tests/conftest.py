"""Synthetic, offline fixtures (never the real dataset)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

SPAM = [
    "WINNER!! You have won a £1000 prize. Call 09061701461 now to claim",
    "Free entry in 2 a weekly comp to win FA Cup tickets. Text FA to 87121 now",
    "URGENT! Your mobile number has won £2000 bonus caller prize. Call 09050000000",
    "Congratulations! Claim your free $500 gift card at http://win.example.com today",
    "You have been selected for a FREE holiday. Reply WIN to 80062 to claim your prize",
    "Final notice: claim your £150 reward now, call 08001234567 or visit www.prize.example",
    "Get free ringtones now! Text TONE to 83333, only 150p per week",
    "Cash prize £5000 waiting. Txt CLAIM to 81010 now. T&Cs apply",
]
HAM = [
    "Ok see you at the station at 6, bring the umbrella",
    "Can you pick up some milk on your way home please",
    "Sorry I missed your call, I was in a meeting. Will ring you later",
    "Happy birthday! Hope you have a lovely day with the family",
    "Lunch tomorrow? I was thinking that new place near the office",
    "Don't forget mum's dinner on Sunday, I will cook the pasta",
    "Thanks for yesterday, it was great catching up with you all",
    "I'll be late tonight, the train is delayed again, start without me",
]


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows += [("spam", f"{t} #{i}") for t in SPAM]
        rows += [("ham", f"{t} #{i}") for t in HAM]
    return pd.DataFrame(rows, columns=["label", "text"])


@pytest.fixture()
def tiny_cfg(tmp_path):
    return {
        "seed": 42, "test_size": 0.25, "cv_folds": 3, "max_ham_fpr": 0.05,
        "paths": {k: str(tmp_path / k) for k in
                  ["raw_dir", "processed_dir", "models_dir", "reports_dir"]},
        "dataset": {"url": "", "member": "SMSSpamCollection", "expected_rows": 0},
        "preprocess": {"remove_stopwords": False, "stem": False},
        "features": {"word_ngram": [1, 2], "char_ngram": [2, 4], "use_char": True,
                     "use_meta": False},
        "inference": {"max_length": 1000},
        "grids": {"min_df": [1], "multinomial_nb": {"alpha": [0.1, 1.0]},
                  "complement_nb": {"alpha": [0.1]}, "logreg": {"C": [10.0]},
                  "linear_svm": {"C": [1.0]}, "random_forest": {"n_estimators": [20]}},
        "candidates": ["multinomial_nb", "logreg"],
        "ablation": False,
    }


@pytest.fixture()
def cfg_copy(tiny_cfg):
    return copy.deepcopy(tiny_cfg)


@pytest.fixture()
def trained(tiny_cfg, raw_df):
    from spamdetect import data, train

    df, _ = data.clean(raw_df)
    tr, te = data.split(df, tiny_cfg["test_size"], tiny_cfg["seed"])
    metrics = train.run_training(tiny_cfg, tr.reset_index(drop=True), te.reset_index(drop=True),
                                 "testsum")
    return tiny_cfg, metrics, tr, te
