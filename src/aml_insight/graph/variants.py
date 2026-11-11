"""Reproducible graph ablations for E2."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from aml_insight.features.account import AMOUNT_FEATURES


def remove_amount_features(features: pd.DataFrame) -> pd.DataFrame:
    """Feature ablation for S2; labels and identifiers are retained separately."""
    return features.drop(columns=[c for c in AMOUNT_FEATURES if c in features], errors="ignore").copy()


def remove_node_features(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop an explicit feature subset without modifying the input frame."""
    protected = {"nodeid", "isFraud", "fraudStep"}
    if protected & set(columns):
        raise ValueError("cannot remove identifiers/labels through feature ablation")
    return features.drop(columns=columns, errors="raise").copy()


def remove_edges(edge_table: pd.DataFrame, *, fraction: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Randomly remove a fraction of aggregated account-pair edges, reproducibly."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    return edge_table.sample(frac=1 - fraction, random_state=seed).sort_values(
        ["sourceNodeId", "targetNodeId"]
    ).reset_index(drop=True)


def degree_preserving_rewire(edge_table: pd.DataFrame, *, swaps: int | None = None,
                             seed: int = 0) -> pd.DataFrame:
    """Directed double-edge swaps preserving every node's in/out degree.

    Rewired edges retain a deterministic neutral payload (unit count and missing
    amount/time summaries); callers should treat this as a structural ablation.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edge_table[["sourceNodeId", "targetNodeId"]].itertuples(index=False, name=None))
    original_nodes = set(graph.nodes)
    original_in = dict(graph.in_degree())
    original_out = dict(graph.out_degree())
    requested = swaps if swaps is not None else max(1, graph.number_of_edges() // 2)
    if requested < 0:
        raise ValueError("swaps must be non-negative")
    if requested:
        nx.directed_edge_swap(graph, nswap=requested, max_tries=max(requested * 50, 100), seed=seed)
    if set(graph.nodes) != original_nodes or dict(graph.in_degree()) != original_in or dict(graph.out_degree()) != original_out:
        raise AssertionError("directed rewiring did not preserve node degrees")
    return pd.DataFrame([
        {"sourceNodeId": u, "targetNodeId": v, "transaction_count": 1,
         "amount_sum": 0.0, "amount_mean": 0.0, "earliest_time": 0, "latest_time": 0}
        for u, v in sorted(graph.edges())
    ])
