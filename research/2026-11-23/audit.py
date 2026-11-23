"""Read-only audit of all three official samples; preserves event multiplicity."""

import hashlib
import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research_extension_2026-11-23"
COLS = ["sourceNodeId", "targetNodeId", "value", "time"]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit():
    OUT.mkdir(exist_ok=True)
    frames, records = {}, []
    for folder in sorted((ROOT / "data/raw").glob("20K*")):
        n = pd.read_csv(folder / "nodes.csv")
        t = pd.read_csv(folder / "transactions.csv")
        ids = set(n.nodeid)
        labels = n.set_index("nodeid").isFraud
        small = t[t.value < 100]
        both = (small.sourceNodeId.map(labels) == 1) & (small.targetNodeId.map(labels) == 1)
        g = nx.Graph()
        g.add_nodes_from(ids)
        g.add_edges_from(t[COLS[:2]].itertuples(index=False, name=None))
        components = sorted(map(len, nx.connected_components(g)), reverse=True)
        inactive = ids - set(t.sourceNodeId) - set(t.targetNodeId)
        record = {
            "dataset": folder.name,
            "accounts": len(n),
            "transactions": len(t),
            "positive_accounts": int(n.isFraud.sum()),
            "positive_rate": float(n.isFraud.mean()),
            "account_ids_unique": bool(n.nodeid.is_unique),
            "binary_labels": bool(n.isFraud.isin([0, 1]).all()),
            "missing_values": int(n.isna().sum().sum() + t.isna().sum().sum()),
            "finite_amounts": bool(np.isfinite(t.value).all()),
            "missing_endpoints": int((~t.sourceNodeId.isin(ids) | ~t.targetNodeId.isin(ids)).sum()),
            "duplicate_event_rows": int(t.duplicated(COLS).sum()),
            "self_transfer_rows": int((t.sourceNodeId == t.targetNodeId).sum()),
            "time_min": int(t.time.min()),
            "time_max": int(t.time.max()),
            "inactive_accounts": len(inactive),
            "inactive_positives": int(labels.reindex(list(inactive)).sum()),
            "small_transactions": len(small),
            "small_both_positive": int(both.sum()),
            "components": len(components),
            "largest_component": components[0],
            "sha256": {f: digest(folder / f) for f in ["nodes.csv", "transactions.csv"]},
        }
        frames[folder.name] = (n, t.drop_duplicates(COLS))
        records.append(record)
    overlaps = []
    for a, b in itertools.combinations(frames, 2):
        na, ta = frames[a]
        nb, tb = frames[b]
        joint = na[["nodeid", "isFraud"]].merge(nb[["nodeid", "isFraud"]], on="nodeid")
        shared = len(ta.merge(tb, on=COLS))
        overlaps.append(
            {
                "a": a,
                "b": b,
                "shared_distinct_event_rows": shared,
                "distinct_a": len(ta),
                "distinct_b": len(tb),
                "jaccard_events": shared / (len(ta) + len(tb) - shared),
                "same_numeric_nodeids": len(joint),
                "label_disagreements_for_same_numeric_id": int(
                    (joint.isFraud_x != joint.isFraud_y).sum()
                ),
            }
        )
    payload = {
        "samples": records,
        "overlaps": overlaps,
        "interpretation": [
            "Repeated numeric node IDs do not establish real entity identity across samples.",
            "Exact shared event tuples establish sample dependence; not independent external validation.",
            "Duplicate tuples may be legitimate repeated simulated events; preserve primary analysis and run sensitivity only.",
            "Static account labels cannot establish individual transaction criminality or onset time.",
            "Connected-component holdout feasibility must be checked; do not split a giant component and call it independent.",
        ],
    }
    (OUT / "data_audit.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(overlaps).to_csv(OUT / "sample_overlap.csv", index=False)
    pd.json_normalize(records).to_csv(OUT / "data_quality.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    audit()
