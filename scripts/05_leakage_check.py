"""Fail-fast checks for feature leakage and time-window boundaries."""

from __future__ import annotations

import json

from aml_insight.data.loader import PROJECT_ROOT, load_dataset
from aml_insight.features.account import FEATURE_GROUPS, build_account_features
from aml_insight.graph.build import add_neighborhood_features, build_graph


def main() -> None:
    ds = load_dataset("20K_fanin200cycle200")
    features = build_account_features(
        ds.nodes, ds.transactions, min_step=1, max_step=89, include_labels=False
    )
    feature_columns = set(features.columns) - {"nodeid"}
    forbidden = {"isFraud", "fraudStep"}
    assert not (feature_columns & forbidden), f"forbidden columns: {feature_columns & forbidden}"
    assert features.nodeid.is_unique and len(features) == len(ds.nodes)
    assert ds.transactions.loc[ds.transactions.time > 89].shape[0] > 0
    altered = ds.transactions.copy()
    future = altered.time > 89
    altered.loc[future, "value"] = altered.loc[future, "value"] * 1000
    altered.loc[future, "time"] += 500
    again = build_account_features(
        ds.nodes, altered, min_step=1, max_step=89, include_labels=False
    )
    assert features.equals(again), "future transactions changed T=89 features"
    early_graph = build_graph(ds.nodes, ds.transactions, min_step=1, max_step=89)
    altered_early_graph = build_graph(ds.nodes, altered, min_step=1, max_step=89)
    assert early_graph.edge_table.equals(altered_early_graph.edge_table), "future transactions changed T=89 edges"
    topology = add_neighborhood_features(early_graph)
    altered_topology = add_neighborhood_features(altered_early_graph)
    assert topology.equals(altered_topology), "future transactions changed T=89 graph features"
    window_features = features.merge(topology, on="nodeid", validate="one_to_one")
    declared_features = set().union(*map(set, FEATURE_GROUPS.values()))
    actual_features = set(window_features.columns) - {"nodeid"}
    assert declared_features == actual_features, "feature groups must cover every model feature exactly once"
    result = {"dataset": ds.name, "cutoff": 89, "accounts": len(features),
              "window": [1, 89], "feature_columns": sorted(actual_features), "forbidden_columns_absent": True,
              "future_transaction_invariance": True, "graph_cutoff_invariance": True,
              "graph_feature_invariance": True, "checked_by": "scripts/05_leakage_check.py"}
    path = PROJECT_ROOT / "reports" / "week2" / "leakage_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
