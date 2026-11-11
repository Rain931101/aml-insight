"""Account graph construction and platform neighborhood queries."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass
class GraphBundle:
    graph: nx.DiGraph
    edge_table: pd.DataFrame
    node_table: pd.DataFrame


def build_graph(nodes: pd.DataFrame, transactions: pd.DataFrame, *,
                min_step: int | None = None, max_step: int | None = None) -> GraphBundle:
    """Build a directed account graph, aggregating events without losing counts.

    ``min_step`` and ``max_step`` define inclusive temporal windows while keeping
    the full account universe as nodes.
    """
    if min_step is not None and max_step is not None and min_step > max_step:
        raise ValueError("min_step must be <= max_step")
    graph = nx.DiGraph()
    node_table = nodes.copy()
    for row in node_table.itertuples(index=False):
        attrs = row._asdict()
        node_id = attrs.pop("nodeid")
        graph.add_node(int(node_id), **attrs)
    tx = transactions
    if min_step is not None:
        tx = tx.loc[tx.time >= min_step]
    if max_step is not None:
        tx = tx.loc[tx.time <= max_step]
    edge_table = tx.groupby(["sourceNodeId", "targetNodeId"], as_index=False).agg(
        transaction_count=("tx_id", "size"), amount_sum=("value", "sum"),
        amount_mean=("value", "mean"), earliest_time=("time", "min"), latest_time=("time", "max"),
    )
    for row in edge_table.itertuples(index=False):
        source, target = int(row.sourceNodeId), int(row.targetNodeId)
        graph.add_edge(source, target, transaction_count=int(row.transaction_count),
                       amount_sum=float(row.amount_sum), amount_mean=float(row.amount_mean),
                       earliest_time=int(row.earliest_time), latest_time=int(row.latest_time))
    return GraphBundle(graph, edge_table, node_table)


def add_neighborhood_features(bundle: GraphBundle) -> pd.DataFrame:
    """Return neighbor count, undirected local density and exclusive 2-hop count."""
    g = bundle.graph
    undirected = g.to_undirected(as_view=True)
    rows = []
    for node in g.nodes:
        neighbors = set(g.predecessors(node)) | set(g.successors(node))
        two_hop = set()
        for neighbor in neighbors:
            two_hop.update(g.predecessors(neighbor))
            two_hop.update(g.successors(neighbor))
        two_hop.discard(node)
        two_hop.difference_update(neighbors)
        rows.append({
            "nodeid": node,
            "neighbor_count": len(neighbors),
            "local_density": float(nx.density(undirected.subgraph(neighbors))) if len(neighbors) > 1 else 0.0,
            "two_hop_neighbor_count": len(two_hop),
        })
    return pd.DataFrame(rows)


def to_pyg(
    bundle: GraphBundle,
    feature_frame: pd.DataFrame,
    *,
    setting: str = "S1",
    feature_columns: list[str] | None = None,
):
    """Convert to torch_geometric.data.Data with stable 0..N-1 account indexing.

    One PyG edge is emitted per aggregated directed account pair. S1 edge
    features include amount sum; S2 keeps only count and time boundaries.
    """
    import torch
    from torch_geometric.data import Data

    if setting not in {"S1", "S2"}:
        raise ValueError("setting must be 'S1' or 'S2'")
    nodes = bundle.node_table.sort_values("nodeid").reset_index(drop=True)
    features = feature_frame.set_index("nodeid").reindex(nodes.nodeid)
    cols = feature_columns or [c for c in features.columns if c != "isFraud"]
    if "isFraud" in cols or "fraudStep" in cols:
        raise ValueError("labels and fraudStep must not be model features")
    if not features.index.equals(pd.Index(nodes.nodeid)):
        raise ValueError("feature rows must cover graph nodes in nodeid order")
    index = {int(node): i for i, node in enumerate(nodes.nodeid)}
    edges = bundle.edge_table
    edge_index = torch.tensor([
        [index[int(x)] for x in edges.sourceNodeId],
        [index[int(x)] for x in edges.targetNodeId],
    ], dtype=torch.long)
    edge_attr_names = ["transaction_count", "earliest_time", "latest_time"]
    if setting == "S1":
        edge_attr_names.insert(1, "amount_sum")
    edge_attr = torch.tensor(edges[edge_attr_names].to_numpy(), dtype=torch.float)
    x = torch.tensor(features[cols].fillna(0).to_numpy(), dtype=torch.float)
    y = torch.tensor(nodes.isFraud.to_numpy(), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                node_id=torch.tensor(nodes.nodeid.to_numpy(), dtype=torch.long),
                feature_names=cols, edge_attr_names=edge_attr_names)


def account_transactions(transactions: pd.DataFrame, account_id: int, *, limit: int = 1000) -> pd.DataFrame:
    """Transactions touching an account, newest step first."""
    selected = transactions.loc[
        transactions.sourceNodeId.eq(account_id) | transactions.targetNodeId.eq(account_id)
    ].sort_values(["time", "tx_id"], ascending=False)
    return selected.head(limit).reset_index(drop=True)


def neighborhood(bundle: GraphBundle, account_id: int, *, hops: int = 1) -> nx.DiGraph:
    """Return a directed induced neighborhood of one or two hops."""
    if account_id not in bundle.graph:
        raise KeyError(f"unknown account_id: {account_id}")
    if hops not in {1, 2}:
        raise ValueError("hops must be 1 or 2")
    undirected = bundle.graph.to_undirected(as_view=True)
    reached = {account_id}
    frontier = {account_id}
    for _ in range(hops):
        frontier = set().union(*(set(undirected.neighbors(n)) for n in frontier)) - reached
        reached.update(frontier)
    return bundle.graph.subgraph(reached).copy()
