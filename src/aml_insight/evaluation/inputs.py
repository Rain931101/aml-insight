"""Strict interfaces for member A features and member C account predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

FORBIDDEN = {"isFraud", "fraudStep", "nodeid", "tx_id"}


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Expected .csv or .parquet input")


def validate_ids(values):
    if values.isna().any() or values.duplicated().any():
        raise ValueError("Account IDs must be non-null and unique")
    if not pd.api.types.is_integer_dtype(values):
        raise ValueError("Account IDs must be integers")


def load_features(path, spec, setting, drop_groups=()):
    df = read_table(path)
    if not {"nodeid", "isFraud"} <= set(df):
        raise ValueError("Feature table requires nodeid and isFraud")
    validate_ids(df.nodeid)
    if not df.isFraud.isin([0, 1]).all():
        raise ValueError("Labels must be binary with no missing values")
    if setting not in ("S1", "S2"):
        raise ValueError("Setting must be S1 or S2")
    if "fraudStep" in df:
        raise ValueError("forbidden leakage feature fraudStep")
    groups = spec.get("groups", {})
    if not groups or "amount" not in groups:
        raise ValueError("Features must be classified in groups including amount (may be empty)")
    declared = [c for cols in groups.values() for c in cols]
    if FORBIDDEN.intersection(declared) or "fraudStep" in df:
        raise ValueError("forbidden leakage or identifier feature")
    if len(declared) != len(set(declared)):
        raise ValueError("Feature groups must be disjoint")
    actual = set(df) - {"nodeid", "isFraud"}
    if set(declared) != actual:
        raise ValueError("Every feature must be classified exactly once in groups")
    if not set(drop_groups) <= set(groups):
        raise ValueError("Unknown ablation group")
    if any(not pd.api.types.is_numeric_dtype(df[c]) for c in declared):
        raise ValueError("Features must be numeric")
    if not np.isfinite(df[declared].to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite; imputation must be train-fitted upstream")
    excluded = set(drop_groups) | ({"amount"} if setting == "S2" else set())
    columns = [c for group, cols in groups.items() if group not in excluded for c in cols]
    if not columns:
        raise ValueError("No features remain")
    return df.set_index("nodeid").sort_index(), columns


def align_predictions(path, expected_ids):
    df = read_table(path)
    if set(df) != {"nodeid", "score"}:
        raise ValueError("Predictions require exactly nodeid,score")
    validate_ids(df.nodeid)
    if set(df.nodeid) != set(expected_ids):
        raise ValueError("Prediction IDs must exactly match the split manifest")
    scores = df.set_index("nodeid").loc[expected_ids, "score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0,1]")
    return scores
