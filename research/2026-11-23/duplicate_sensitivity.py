"""Post-hoc robustness of fixed models to ambiguous duplicate event tuples.

Not a new final test; no retraining, no new threshold, no choice of preferred cleaning.
"""

import numpy as np
import pandas as pd
import torch
from core import INPUT, REPORT, ROOT, RUNS, load, scores, write
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from aml_insight.features.account import build_account_features
from aml_insight.models.graphsage import GraphSAGE


def run():
    out = REPORT / "duplicate_sensitivity"
    out.mkdir(exist_ok=False)
    nodes = pd.read_csv(ROOT / "data/raw/20K_fanin200cycle200/nodes.csv")
    tx = pd.read_csv(ROOT / "data/raw/20K_fanin200cycle200/transactions.csv").reset_index(
        names="tx_id"
    )
    unique = tx.drop_duplicates(["sourceNodeId", "targetNodeId", "value", "time"])
    before = set(tx[["sourceNodeId", "targetNodeId"]].itertuples(index=False, name=None))
    after = set(unique[["sourceNodeId", "targetNodeId"]].itertuples(index=False, name=None))
    assert before == after
    f = pd.read_parquet(INPUT / "full.parquet").set_index("nodeid")
    changed = build_account_features(nodes, unique, include_labels=True).set_index("nodeid")
    # Three expensive neighborhood summaries are unchanged because pair topology is identical.
    for col in ["neighbor_count", "local_density", "two_hop_neighbor_count"]:
        changed[col] = f[col]
    edges = np.load(INPUT / "full.npy")
    records = []
    for seed in range(5):
        g = RUNS / f"gnn-full-seed{seed}"
        t = RUNS / f"tabular-full-seed{seed}"
        if not (g / "COMPLETE.json").exists() or not (t / "COMPLETE.json").exists():
            raise ValueError("Baselines incomplete")
        ids = load(g / "manifest.json")["test"]
        labels = f.loc[ids, "isFraud"].to_numpy()
        for kind, directory in [("graphsage", g), ("xgboost", t)]:
            cols = load(directory / "columns.json")
            base = pd.read_csv(directory / "predictions.csv").set_index("nodeid").loc[ids]
            threshold = load(directory / "selection.json")["threshold"]
            if kind == "xgboost":
                m = XGBClassifier()
                m.load_model(t / "xgboost.ubj")
                p = m.predict_proba(changed.loc[ids, cols])[:, 1]
            else:
                prep = np.load(g / "preprocess.npz")
                raw = changed[cols].to_numpy()
                raw = np.sign(raw) * np.log1p(abs(raw))
                x = ((raw - prep["mean"]) / prep["std"]).astype("float32")
                state = torch.load(g / "model.pt", weights_only=True, map_location="cpu")
                m = GraphSAGE(state["in_dim"], state["params"]["hidden"], state["dropout"])
                m.load_state_dict(state["state_dict"])
                p = scores(m, x, edges)[f.index.get_indexer(ids)]
            original = base[f"{kind}_score"].to_numpy()
            pred = p >= threshold
            records.append(
                {
                    "seed": seed,
                    "model": kind,
                    "n_test": len(ids),
                    "original_ap": float(average_precision_score(labels, original)),
                    "deduplicated_input_ap": float(average_precision_score(labels, p)),
                    "mean_abs_score_change": float(np.mean(abs(original - p))),
                    "decision_flips": int((pred != base[f"{kind}_pred"].to_numpy()).sum()),
                }
            )
    pd.DataFrame(records).to_csv(out / "results.csv", index=False)
    write(
        out / "receipt.json",
        {
            "removed_duplicate_rows": len(tx) - len(unique),
            "account_pair_topology_unchanged": True,
            "status": "post_hoc_fixed_model_input_sensitivity",
            "limitations": "Not evidence duplicates are errors; primary event multiplicity preserved; no threshold retuning",
        },
    )
    print("Duplicate sensitivity saved")


if __name__ == "__main__":
    run()
