"""Build leakage-safe account features and graph artifacts for the main dataset."""

from __future__ import annotations

import json
import subprocess

from aml_insight.data.clean import clean_dataset
from aml_insight.data.loader import PROJECT_ROOT, load_dataset
from aml_insight.features.account import FEATURE_GROUPS, build_account_features, select_feature_set
from aml_insight.graph.build import add_neighborhood_features, build_graph, to_pyg


def main() -> None:
    dataset = load_dataset("20K_fanin200cycle200")
    nodes, tx, cleaning = clean_dataset(dataset)
    def build_window_features(min_step: int | None, max_step: int | None):
        account_features = build_account_features(
            nodes, tx, min_step=min_step, max_step=max_step, include_labels=True
        )
        window_graph = build_graph(nodes, tx, min_step=min_step, max_step=max_step)
        topology = add_neighborhood_features(window_graph)
        return account_features.merge(topology, on="nodeid", validate="one_to_one"), window_graph

    features, graph = build_window_features(None, None)
    time_train_features, time_train_graph = build_window_features(1, 89)
    time_test_features, time_test_graph = build_window_features(90, 149)
    out = PROJECT_ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out / "features.parquet", index=False)
    select_feature_set(features, "S1").to_parquet(out / "features_s1.parquet", index=False)
    select_feature_set(features, "S2").to_parquet(out / "features_s2.parquet", index=False)
    graph.edge_table.to_parquet(out / "edges.parquet", index=False)
    graph.node_table.to_parquet(out / "nodes.parquet", index=False)
    for frame, aliases in (
        (time_train_features, ("time_train_features.parquet", "account_features_steps1_89.parquet")),
        (time_test_features, ("time_test_features.parquet", "account_features_steps90_149.parquet")),
    ):
        for filename in aliases:
            frame.to_parquet(out / filename, index=False)
    feature_groups = {
        "amount": list(FEATURE_GROUPS["amount"]),
        "behavior": list(FEATURE_GROUPS["behavior"]),
        "temporal": list(FEATURE_GROUPS["temporal"]),
        "structure": list(FEATURE_GROUPS["structure"]),
    }
    (out / "feature_groups.json").write_text(json.dumps(feature_groups, indent=2), encoding="utf-8")
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip() or None
    windows = {
        "dataset": dataset.name,
        "data_commit": (PROJECT_ROOT / "data/raw/AMLSIM_COMMIT.txt").read_text().strip(),
        "code_commit": code_commit,
        "train": {
            "min_step": 1, "max_step": 89, "accounts": len(time_train_features),
            "transactions": int(tx.time.between(1, 89).sum()),
            "aggregated_edges": time_train_graph.graph.number_of_edges(),
            "files": ["time_train_features.parquet", "account_features_steps1_89.parquet"],
        },
        "test": {
            "min_step": 90, "max_step": 149, "accounts": len(time_test_features),
            "transactions": int(tx.time.between(90, 149).sum()),
            "aggregated_edges": time_test_graph.graph.number_of_edges(),
            "files": ["time_test_features.parquet", "account_features_steps90_149.parquet"],
        },
        "window_semantics": "inclusive; features and graph edges use only transactions within each window",
    }
    (out / "time_windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")
    import torch
    for setting in ("S1", "S2"):
        setting_features = select_feature_set(features, setting)
        pyg = to_pyg(graph, setting_features, setting=setting, feature_columns=[
            c for c in setting_features.columns if c not in {"nodeid", "isFraud", "fraudStep"}
        ])
        torch.save(pyg, out / f"graph_{setting.lower()}.pt")
        if setting == "S1":
            torch.save(pyg, out / "graph.pt")  # backward-compatible S1 name
    summary = {
        "dataset": dataset.name, "source_commit": (PROJECT_ROOT / "data/raw/AMLSIM_COMMIT.txt").read_text().strip(),
        "code_commit": code_commit,
        "accounts": graph.graph.number_of_nodes(), "aggregated_directed_edges": graph.graph.number_of_edges(),
        "transactions": len(tx), "self_loops_retained": cleaning.self_loop_rows,
        "duplicate_rows_retained": cleaning.exact_duplicate_rows,
        "accounts_without_transactions": cleaning.accounts_without_transactions,
        "s1_features": [c for c in select_feature_set(features, "S1") if c not in {"nodeid", "isFraud"}],
        "s2_features": [c for c in select_feature_set(features, "S2") if c not in {"nodeid", "isFraud"}],
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
