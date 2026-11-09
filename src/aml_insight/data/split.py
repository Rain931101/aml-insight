"""Stable account splits, shared with A and C by a persisted ID manifest."""

import numpy as np
from sklearn.model_selection import train_test_split

SEEDS = [0, 1, 2, 3, 4]


def stratified_split(ids, y, seed):
    ids, y = np.asarray(ids), np.asarray(y)
    if ids.ndim != 1 or y.ndim != 1 or len(ids) != len(y):
        raise ValueError("Expected aligned ID and label vectors")
    if len(np.unique(ids)) != len(ids) or not np.isin(y, [0, 1]).all():
        raise ValueError("Unique IDs and binary labels required")
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes required")
    # Sort first so row order in an A-delivered file cannot change account membership.
    order = np.argsort(ids)
    ids, y = ids[order], y[order]
    idx = np.arange(len(ids))
    train, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    result = {"train": train, "validation": val, "test": test}
    for part in result.values():
        if len(np.unique(y[part])) != 2 or abs(float(y[part].mean() - y.mean())) >= 0.005:
            raise ValueError("Cannot meet two-class split and <0.005 prevalence tolerance")
    return {name: sorted(ids[part].tolist()) for name, part in result.items()}
