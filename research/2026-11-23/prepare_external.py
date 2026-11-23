"""Elliptic local-feature temporal split; unknown labels remain unknown, not normal."""

import numpy as np
import pandas as pd
from core import INPUT, ROOT, sha, write


def prepare():
    raw = ROOT / "data/external/elliptic"
    path = INPUT / "elliptic"
    path.mkdir(exist_ok=False)
    features = pd.read_csv(raw / "elliptic_txs_features.csv", header=None, usecols=range(95))
    features.columns = ["nodeid", "time", *[f"local_{i}" for i in range(93)]]
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv", dtype={"class": str})
    frame = features.merge(classes, left_on="nodeid", right_on="txId", validate="one_to_one").drop(
        columns="txId"
    )
    frame["label"] = frame["class"].map({"1": 1, "2": 0}).fillna(-1).astype(int)
    if not frame.nodeid.is_unique:
        raise ValueError("Duplicate tx IDs")
    if (
        frame.isna().any().any()
        or not np.isfinite(frame[[f"local_{i}" for i in range(93)]].to_numpy()).all()
    ):
        raise ValueError("Invalid external features")
    frame.to_parquet(path / "features.parquet", index=False)
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")
    positions = pd.Series(np.arange(len(frame)), index=frame.nodeid)
    a = positions.reindex(edges.txId1).to_numpy()
    b = positions.reindex(edges.txId2).to_numpy()
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing external endpoint")
    a = a.astype(int)
    b = b.astype(int)
    times = frame.time.to_numpy()
    for name, horizon in [("train", 29), ("validation", 34), ("test", 49)]:
        take = (times[a] <= horizon) & (times[b] <= horizon) & (a != b)
        e = np.vstack([a[take], b[take]])
        e = np.unique(np.concatenate([e, e[::-1]], axis=1), axis=1)
        np.save(path / f"{name}.npy", e)
    masks = {
        "train": frame.time.le(29) & frame.label.ge(0),
        "validation": frame.time.between(30, 34) & frame.label.ge(0),
        "test": frame.time.between(35, 49) & frame.label.ge(0),
    }
    idx = {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()}
    for name, rows in idx.items():
        if frame.iloc[rows].label.nunique() != 2:
            raise ValueError(f"Single class {name}")
    write(path / "split.json", idx)
    write(
        path / "receipt.json",
        {
            "source": str(raw / "receipt.json"),
            "raw_sha256": {f.name: sha(f) for f in raw.glob("*.csv")},
            "counts": {
                name: {"known": len(rows), "positive": int(frame.iloc[rows].label.sum())}
                for name, rows in idx.items()
            },
            "unknown_nodes": int((frame.label == -1).sum()),
            "nodes": len(frame),
            "directed_edges": len(edges),
            "features": "93 local features; excludes time and 72 published aggregated features",
            "limitation": "Published features may already have vendor preprocessing; train-only rescaling cannot prove original preprocessing was train-only.",
            "files": {f.name: sha(f) for f in path.iterdir() if f.is_file()},
        },
    )
    print("Prepared independent-source temporal method replication; no model test metrics computed")


if __name__ == "__main__":
    prepare()
