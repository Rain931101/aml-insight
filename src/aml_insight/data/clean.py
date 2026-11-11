"""Validated, non-destructive cleaning for AMLSim transaction data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aml_insight.data.loader import AMLSimDataset


@dataclass(frozen=True)
class CleaningSummary:
    input_transactions: int
    output_transactions: int
    exact_duplicate_rows: int
    self_loop_rows: int
    accounts_without_transactions: int


def clean_dataset(
    dataset: AMLSimDataset,
    *,
    drop_self_loops: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningSummary]:
    """Return validated node/transaction copies and a row-level quality summary.

    Duplicate transaction rows and self-loops are retained by default. They are
    simulator events, not accidental duplicate records for this project. Graph
    construction later aggregates events by directed account pair while storing
    their multiplicity and amount/time summaries.
    """
    nodes = dataset.nodes.copy()
    tx = dataset.transactions.copy()
    required_nodes = {"nodeid", "isFraud", "init_balance", "fraudStep"}
    required_tx = {"tx_id", "sourceNodeId", "targetNodeId", "value", "time"}
    if missing := required_nodes - set(nodes.columns):
        raise ValueError(f"nodes missing required columns: {sorted(missing)}")
    if missing := required_tx - set(tx.columns):
        raise ValueError(f"transactions missing required columns: {sorted(missing)}")
    if nodes.nodeid.duplicated().any():
        raise ValueError("nodeid must be unique")
    account_ids = set(nodes.nodeid)
    if not tx.sourceNodeId.isin(account_ids).all() or not tx.targetNodeId.isin(account_ids).all():
        raise ValueError("transaction endpoint is missing from nodes")
    if (tx.value <= 0).any():
        raise ValueError("transaction values must be positive")
    duplicate_mask = tx.duplicated(["sourceNodeId", "targetNodeId", "value", "time"], keep="first")
    loop_mask = tx.sourceNodeId.eq(tx.targetNodeId)
    summary = CleaningSummary(
        input_transactions=len(tx),
        output_transactions=len(tx) - (int(loop_mask.sum()) if drop_self_loops else 0),
        exact_duplicate_rows=int(duplicate_mask.sum()),
        self_loop_rows=int(loop_mask.sum()),
        accounts_without_transactions=len(account_ids - set(tx.sourceNodeId) - set(tx.targetNodeId)),
    )
    if drop_self_loops:
        tx = tx.loc[~loop_mask].copy()
    return nodes.reset_index(drop=True), tx.reset_index(drop=True), summary
