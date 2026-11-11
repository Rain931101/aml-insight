"""Small deterministic checks for member A's data and graph contracts."""

import pandas as pd
import pytest

from aml_insight.data.clean import clean_dataset
from aml_insight.data.loader import AMLSimDataset
from aml_insight.features.account import FEATURE_GROUPS, build_account_features, select_feature_set
from aml_insight.graph.build import add_neighborhood_features, build_graph, neighborhood, to_pyg
from aml_insight.graph.variants import degree_preserving_rewire, remove_edges


@pytest.fixture
def mini():
    nodes = pd.DataFrame({"nodeid": [0, 1, 2, 3], "isFraud": [0, 1, 0, 0],
                          "init_balance": [10., 20., 30., 40.], "fraudStep": [-1, 10, -1, -1]})
    tx = pd.DataFrame({"tx_id": [0, 1, 2, 3, 4], "sourceNodeId": [0, 0, 1, 1, 0],
                       "targetNodeId": [1, 1, 2, 1, 1], "value": [5., 5., 7., 4., 5.], "time": [1, 2, 3, 4, 1]})
    return AMLSimDataset("mini", nodes, tx, "")


def test_clean_preserves_duplicates_loops_and_isolates(mini):
    _, tx, summary = clean_dataset(mini)
    assert len(tx) == 5
    assert summary.exact_duplicate_rows == 1
    assert summary.self_loop_rows == 1
    assert summary.accounts_without_transactions == 1


def test_temporal_cutoff_and_amount_sets(mini):
    f = build_account_features(mini.nodes, mini.transactions, max_step=2, include_labels=True)
    assert f.loc[f.nodeid == 1, "in_tx_count"].iloc[0] == 3
    assert f.loc[f.nodeid == 1, "in_amount_sum"].iloc[0] == 15
    assert "out_amount_sum" in select_feature_set(f, "S1")
    assert "out_amount_sum" not in select_feature_set(f, "S2")
    assert "init_balance" in select_feature_set(f, "S1")
    assert "init_balance" not in select_feature_set(f, "S2")
    assert "fraudStep" not in f.columns
    assert set().union(*map(set, FEATURE_GROUPS.values())) == {
        "init_balance", "in_amount_sum", "in_amount_mean", "out_amount_sum", "out_amount_mean",
        "net_flow", "in_tx_count", "out_tx_count", "active_step_count", "mean_interarrival_steps",
        "median_interarrival_steps", "in_degree", "out_degree", "neighbor_count", "local_density",
        "two_hop_neighbor_count",
    }
    late = build_account_features(mini.nodes, mini.transactions, min_step=3, max_step=4)
    assert late.loc[late.nodeid == 2, "in_tx_count"].iloc[0] == 1
    assert late.loc[late.nodeid == 1, "in_tx_count"].iloc[0] == 1
    with pytest.raises(ValueError, match="min_step"):
        build_account_features(mini.nodes, mini.transactions, min_step=4, max_step=3)


def test_graph_aggregates_events_and_includes_isolates(mini):
    bundle = build_graph(mini.nodes, mini.transactions)
    assert bundle.graph.number_of_nodes() == 4
    assert bundle.graph[0][1]["transaction_count"] == 3
    assert bundle.edge_table.transaction_count.sum() == len(mini.transactions)
    assert add_neighborhood_features(bundle).loc[lambda x: x.nodeid == 3, "neighbor_count"].iloc[0] == 0
    assert len(neighborhood(bundle, 0, hops=2)) == 3
    early = build_graph(mini.nodes, mini.transactions, max_step=2)
    assert early.edge_table.transaction_count.sum() == 3
    assert early.edge_table.latest_time.max() <= 2
    assert early.graph.number_of_nodes() == 4
    late = build_graph(mini.nodes, mini.transactions, min_step=3, max_step=4)
    assert late.edge_table.transaction_count.sum() == 2
    assert late.edge_table.earliest_time.min() >= 3


def test_s2_pyg_edge_attributes_exclude_amounts(mini):
    bundle = build_graph(mini.nodes, mini.transactions)
    features = build_account_features(mini.nodes, mini.transactions, include_labels=True)
    features = features.merge(add_neighborhood_features(bundle), on="nodeid")
    s2 = select_feature_set(features, "S2")
    graph = to_pyg(
        bundle, s2, setting="S2",
        feature_columns=[c for c in s2 if c not in {"nodeid", "isFraud"}],
    )
    assert graph.feature_names
    assert graph.edge_attr_names == ["transaction_count", "earliest_time", "latest_time"]
    assert graph.edge_attr.shape[1] == 3
    s1 = select_feature_set(features, "S1")
    full_graph = to_pyg(
        bundle, s1, setting="S1",
        feature_columns=[c for c in s1 if c not in {"nodeid", "isFraud"}],
    )
    assert full_graph.edge_attr_names == [
        "transaction_count", "amount_sum", "earliest_time", "latest_time"
    ]
    assert full_graph.edge_attr.shape[1] == 4


def test_variants_are_reproducible(mini):
    graph = build_graph(mini.nodes, mini.transactions)
    a = remove_edges(graph.edge_table, fraction=0.25, seed=12)
    b = remove_edges(graph.edge_table, fraction=0.25, seed=12)
    pd.testing.assert_frame_equal(a, b)
    cycle = pd.DataFrame({"sourceNodeId": [0, 1, 2, 3], "targetNodeId": [1, 2, 3, 0],
                          "transaction_count": [1] * 4, "amount_sum": [2.] * 4,
                          "amount_mean": [2.] * 4, "earliest_time": [1] * 4,
                          "latest_time": [1] * 4})
    rewired = degree_preserving_rewire(cycle, swaps=1, seed=3)
    assert len(rewired) == len(cycle)
