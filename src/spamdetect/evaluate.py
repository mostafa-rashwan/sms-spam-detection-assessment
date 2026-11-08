"""Metrics, threshold policy, plots and error analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> tuple[float, float]:
    """Return ``(threshold, recall)`` maximising spam recall with ham FPR <= ``max_fpr``.

    A message is flagged spam when ``score >= threshold``. Among equal-recall thresholds the
    highest (most conservative) one is kept. If no score satisfies the cap, the threshold is
    placed just above the maximum score (recall 0).
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    n_neg, n_pos = max(int((y == 0).sum()), 1), max(int((y == 1).sum()), 1)
    best_t, best_r = float(np.nextafter(scores.max(), np.inf)), 0.0
    for t in np.unique(scores)[::-1]:
        pred = scores >= t
        fpr = (pred & (y == 0)).sum() / n_neg
        if fpr <= max_fpr:
            rec = (pred & (y == 1)).sum() / n_pos
            if rec > best_r:
                best_t, best_r = float(t), float(rec)
    return best_t, best_r


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, F1, PR-AUC, ROC-AUC, ham FPR and confusion counts at ``threshold``."""
    y = np.asarray(y)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(y, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "ham_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and std of each numeric metric over folds."""
    keys = ["precision", "recall", "f1", "pr_auc", "roc_auc", "ham_fpr"]
    return {k: {"mean": float(np.mean([m[k] for m in fold_metrics])),
                "std": float(np.std([m[k] for m in fold_metrics]))} for k in keys}


def plot_confusion_matrix(m: dict[str, float], path: str | Path) -> None:
    """Save a confusion-matrix image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test confusion matrix (thr={m['threshold']:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr_curve(y: np.ndarray, scores: np.ndarray, m: dict[str, float], path: str | Path) -> None:
    """Save the test precision-recall curve with the operating point marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p, r, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, label=f"PR-AUC={m['pr_auc']:.3f}")
    ax.scatter([m["recall"]], [m["precision"]], color="red", zorder=3, label="operating point")
    ax.set_xlabel("Recall (spam)")
    ax.set_ylabel("Precision (spam)")
    ax.set_title("Test precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def error_table(texts, y: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """False positives and false negatives with their text and score."""
    y = np.asarray(y)
    pred = (np.asarray(scores) >= threshold).astype(int)
    df = pd.DataFrame({"text": list(texts), "label": y, "pred": pred, "spam_probability": scores})
    df = df[df["label"] != df["pred"]].copy()
    df["error"] = np.where(df["pred"] == 1, "false_positive", "false_negative")
    return df.sort_values(["error", "spam_probability"]).reset_index(drop=True)
