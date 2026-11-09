"""Account classification metrics. PR-AUC means Average Precision, not trapezoidal AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def validate_vectors(y, scores, ids=None):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) == 0 or len(y) != len(scores):
        raise ValueError("Labels and scores must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    if ids is not None:
        ids = np.asarray(ids)
        if ids.ndim != 1 or len(ids) != len(y) or len(np.unique(ids)) != len(ids):
            raise ValueError("Expected aligned unique account IDs")
    return y.astype(int), scores


def select_threshold(y, scores):
    """Maximize validation F1; choose highest threshold on a tie (fewer alerts)."""
    y, scores = validate_vectors(y, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes in validation")
    p, r, t = precision_recall_curve(y, scores)
    f = np.divide(
        2 * p[:-1] * r[:-1], p[:-1] + r[:-1], out=np.zeros(len(t)), where=(p[:-1] + r[:-1]) > 0
    )
    return float(t[np.flatnonzero(np.isclose(f, f.max(), rtol=0, atol=1e-12))[-1]])


def evaluate(y, scores, threshold, ids, ks=(100, 500)):
    y, scores = validate_vectors(y, scores, ids)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    order = np.lexsort((np.asarray(ids), -scores))
    positives = int(y.sum())
    top = {}
    for k in dict.fromkeys([*ks, positives]):
        if not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError("K must be a nonnegative integer")
        effective = min(int(k), len(y))
        hits = int(y[order[:effective]].sum())
        top[str(k)] = {
            "requested_k": int(k),
            "effective_k": effective,
            "precision": hits / effective if effective else None,
            "recall": hits / positives if positives else None,
        }
    both = len(np.unique(y)) == 2
    pred = scores >= threshold
    return {
        "n": len(y),
        "positives": positives,
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y, scores)) if positives else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "top_k": top,
        "top_k_positive_count": top[str(positives)],
        "undefined": {
            **(
                {"pr_auc": "no positive labels", "recall_at_k": "no positive labels"}
                if not positives
                else {}
            ),
            **({"roc_auc": "requires both classes"} if not both else {}),
        },
    }
