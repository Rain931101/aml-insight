"""Leakage-safe account behavior features for AMLSim."""

from __future__ import annotations

import pandas as pd

from aml_insight.data.loader import LEAKAGE_COLUMNS

AMOUNT_FEATURES = (
    "init_balance", "in_amount_sum", "in_amount_mean", "out_amount_sum",
    "out_amount_mean", "net_flow", "min_amount",
)
BEHAVIOR_FEATURES = ("in_tx_count", "out_tx_count")
TEMPORAL_FEATURES = (
    "active_step_count", "mean_interarrival_steps", "median_interarrival_steps",
)
STRUCTURAL_FEATURES = (
    "in_degree", "out_degree", "neighbor_count", "local_density",
    "two_hop_neighbor_count",
)
FEATURE_GROUPS = {
    "amount": AMOUNT_FEATURES,
    "behavior": BEHAVIOR_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "structure": STRUCTURAL_FEATURES,
}
NON_AMOUNT_FEATURES = (*BEHAVIOR_FEATURES, *TEMPORAL_FEATURES, *STRUCTURAL_FEATURES[:2])


def build_account_features(
    nodes: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    min_step: int | None = None,
    max_step: int | None = None,
    include_labels: bool = False,
) -> pd.DataFrame:
    """Build features from transactions within the inclusive step window.

    Amount/balance features are isolated from behavior, temporal, and structure
    groups for S1/S2. Labels are excluded by default and never enter calculations.
    """
    if min_step is not None and max_step is not None and min_step > max_step:
        raise ValueError("min_step must be <= max_step")
    tx = transactions
    if min_step is not None:
        tx = tx.loc[tx.time >= min_step]
    if max_step is not None:
        tx = tx.loc[tx.time <= max_step]
    tx = tx.copy()
    columns = ["nodeid", *NON_AMOUNT_FEATURES, *AMOUNT_FEATURES]
    out_tx = tx.groupby("sourceNodeId", sort=False).agg(
        out_tx_count=("tx_id", "size"), out_amount_sum=("value", "sum"),
        out_amount_mean=("value", "mean"), out_degree=("targetNodeId", "nunique"),
    )
    in_tx = tx.groupby("targetNodeId", sort=False).agg(
        in_tx_count=("tx_id", "size"), in_amount_sum=("value", "sum"),
        in_amount_mean=("value", "mean"), in_degree=("sourceNodeId", "nunique"),
    )
    base = nodes[["nodeid", "init_balance"]].set_index("nodeid")
    base = base.join(out_tx.rename_axis("nodeid")).join(in_tx.rename_axis("nodeid"))
    count_cols = ["out_tx_count", "out_degree", "in_tx_count", "in_degree"]
    base[count_cols] = base[count_cols].fillna(0).astype("int64")
    amount_cols = ["out_amount_sum", "out_amount_mean", "in_amount_sum", "in_amount_mean"]
    base[amount_cols] = base[amount_cols].fillna(0.0)
    base["net_flow"] = base["in_amount_sum"] - base["out_amount_sum"]
    incident_amounts = pd.concat([
        tx[["sourceNodeId", "value"]].rename(columns={"sourceNodeId": "nodeid"}),
        tx[["targetNodeId", "value"]].rename(columns={"targetNodeId": "nodeid"}),
    ], ignore_index=True)
    base["min_amount"] = incident_amounts.groupby("nodeid").value.min().reindex(base.index).fillna(0.0)

    # Inter-arrival intervals are calculated separately for incoming and outgoing
    # events, then combined as event-to-event intervals per account.
    event_rows = pd.concat([
        tx[["sourceNodeId", "time"]].rename(columns={"sourceNodeId": "nodeid"}),
        tx[["targetNodeId", "time"]].rename(columns={"targetNodeId": "nodeid"}),
    ], ignore_index=True).sort_values(["nodeid", "time"], kind="stable")
    event_rows["interval"] = event_rows.groupby("nodeid", sort=False).time.diff()
    interval_summary = event_rows.dropna(subset=["interval"]).groupby("nodeid").interval.agg(["mean", "median"]).rename(columns={
        "mean": "mean_interarrival_steps", "median": "median_interarrival_steps",
    })
    base = base.join(interval_summary)
    base[["mean_interarrival_steps", "median_interarrival_steps"]] = base[
        ["mean_interarrival_steps", "median_interarrival_steps"]
    ].fillna(0.0)
    base["active_step_count"] = event_rows.groupby("nodeid").time.nunique().reindex(base.index).fillna(0).astype("int64")
    result = base.reset_index()[columns]
    if include_labels:
        result = result.merge(nodes[["nodeid", "isFraud"]], on="nodeid", validate="one_to_one")
    feature_columns = [c for c in result if c not in {"nodeid", "isFraud"}]
    forbidden = set(LEAKAGE_COLUMNS) & set(feature_columns)
    if forbidden:
        raise AssertionError(f"leakage columns entered features: {sorted(forbidden)}")
    return result


def select_feature_set(features: pd.DataFrame, setting: str = "S1") -> pd.DataFrame:
    """Select model inputs: S1 includes amount/balance; S2 excludes that group."""
    if setting not in {"S1", "S2"}:
        raise ValueError("setting must be 'S1' or 'S2'")
    keep = ["nodeid", *NON_AMOUNT_FEATURES, *(c for c in STRUCTURAL_FEATURES[2:] if c in features)]
    if setting == "S1":
        keep += list(AMOUNT_FEATURES)
    if "isFraud" in features:
        keep.append("isFraud")
    return features[keep].copy()
