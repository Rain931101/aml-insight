"""Build immutable, separated inputs; no label-based topology construction."""

import time

import networkx as nx
import numpy as np
import pandas as pd
from core import INPUT, ROOT, sha, write


def array_edges(graph):
    a = np.array(sorted(graph.edges()), dtype=np.int64).T
    return np.concatenate([a, a[::-1]], axis=1) if a.size else np.empty((2, 0), dtype=np.int64)


def prepare():
    INPUT.mkdir(parents=True, exist_ok=False)
    source = ROOT / "data/processed"
    for key, file in [
        ("full", "features.parquet"),
        ("early", "time_train_features.parquet"),
        ("late", "time_test_features.parquet"),
    ]:
        f = pd.read_parquet(source / file).sort_values("nodeid").reset_index(drop=True)
        if not np.array_equal(f.nodeid, np.arange(20000)):
            raise ValueError("Unexpected node universe")
        f.to_parquet(INPUT / f"{key}.parquet", index=False)
    tx = pd.read_csv(ROOT / "data/raw/20K_fanin200cycle200/transactions.csv")
    graph = nx.Graph()
    graph.add_nodes_from(range(20000))
    graph.add_edges_from(tx[["sourceNodeId", "targetNodeId"]].itertuples(index=False, name=None))
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    original = set(graph.edges())
    meta = []
    np.save(INPUT / "full.npy", array_edges(graph))
    np.save(INPUT / "no_edges.npy", np.empty((2, 0), dtype=np.int64))
    for part, mask in [("early", tx.time <= 89), ("late", tx.time >= 90)]:
        g = nx.Graph()
        g.add_nodes_from(range(20000))
        g.add_edges_from(
            tx.loc[mask, ["sourceNodeId", "targetNodeId"]].itertuples(index=False, name=None)
        )
        g.remove_edges_from(list(nx.selfloop_edges(g)))
        np.save(INPUT / f"{part}.npy", array_edges(g))
    for seed in range(5):
        start = time.perf_counter()
        rewired = graph.copy()
        nx.double_edge_swap(
            rewired, nswap=2 * len(original), max_tries=100 * len(original), seed=seed
        )
        if dict(rewired.degree()) != dict(graph.degree()):
            raise AssertionError("degree changed")
        edges = set(rewired.edges())
        common = len(original & edges)
        np.save(INPUT / f"rewire-{seed}.npy", array_edges(rewired))
        rng = np.random.default_rng(seed)
        pairs = sorted(original)
        choose = rng.choice(len(pairs), int(np.floor(len(pairs) * 0.5)), replace=False)
        dropped = nx.Graph()
        dropped.add_nodes_from(range(20000))
        dropped.add_edges_from([pairs[i] for i in choose])
        np.save(INPUT / f"drop50-{seed}.npy", array_edges(dropped))
        meta.append(
            {
                "seed": seed,
                "original_undirected_pairs": len(original),
                "rewired_pairs": len(edges),
                "swaps": 2 * len(original),
                "all_node_degrees_preserved": True,
                "edge_overlap_fraction": common / len(original),
                "dropped_pairs": len(choose),
                "seconds": time.perf_counter() - start,
            }
        )
    write(INPUT / "graph_checks.json", meta)
    write(
        INPUT / "inputs.json",
        {
            "files": {f.name: sha(f) for f in sorted(INPUT.iterdir()) if f.is_file()},
            "sources": {
                str(source / f): sha(source / f)
                for f in [
                    "features.parquet",
                    "time_train_features.parquet",
                    "time_test_features.parquet",
                ]
            },
        },
    )
    print(meta)


if __name__ == "__main__":
    prepare()
