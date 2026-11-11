"""Materialize reproducible E2 graph ablations under ignored processed data."""

from __future__ import annotations

import json

from aml_insight.data.loader import PROJECT_ROOT, load_dataset
from aml_insight.features.account import AMOUNT_FEATURES
from aml_insight.graph.build import build_graph
from aml_insight.graph.variants import degree_preserving_rewire, remove_edges, remove_node_features


def main() -> None:
    ds = load_dataset("20K_fanin200cycle200")
    bundle = build_graph(ds.nodes, ds.transactions)
    out = PROJECT_ROOT / "data" / "processed" / "e2_variants"
    out.mkdir(parents=True, exist_ok=True)
    edges = bundle.edge_table
    variants = {
        "drop_20pct_edges_seed0": remove_edges(edges, fraction=.2, seed=0),
        "degree_preserving_rewire_seed0": degree_preserving_rewire(edges, swaps=max(1, len(edges) // 2), seed=0),
    }
    manifest = {"dataset": ds.name, "variants": {}}
    for name, frame in variants.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
        manifest["variants"][name] = {"rows": len(frame), "seed": 0}
    features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    if features_path.exists():
        import pandas as pd
        features = pd.read_parquet(features_path)
        amount_free = remove_node_features(features, list(AMOUNT_FEATURES))
        amount_free.to_parquet(out / "features_without_amounts.parquet", index=False)
        manifest["variants"]["features_without_amounts"] = {
            "rows": len(amount_free), "columns": len(amount_free.columns), "seed": None
        }
        structure_free = remove_node_features(
            features, ["neighbor_count", "local_density", "two_hop_neighbor_count"]
        )
        structure_free.to_parquet(out / "features_without_graph_structure.parquet", index=False)
        manifest["variants"]["features_without_graph_structure"] = {
            "rows": len(structure_free), "columns": len(structure_free.columns), "seed": None
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
