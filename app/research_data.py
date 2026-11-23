"""Pure read-only calculations for the research workbench; no model imports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).parent / "assets/research"
MODELS = {"graphsage": "GraphSAGE", "xgboost": "XGBoost", "rule_based": "Rule-based"}
COLORS = {"graphsage": "#8477ef", "xgboost": "#2fa793", "rule_based": "#c69752"}


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_bundle():
    receipt = read_json(ASSETS / "receipt.json")
    for name, expected in receipt["files"].items():
        path = ASSETS / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"研究資料完整性檢查失敗：{name}")
    return receipt


def run_path(setting, seed):
    if setting not in {"S1", "S2"} or seed not in range(5):
        raise ValueError("Invalid experiment selection")
    return ASSETS / f"E1-{setting}-seed{seed}"


def results():
    rows = []
    for path in sorted(ASSETS.glob("E1-*/metrics.json")):
        run = read_json(path)
        for model, metrics in run["models"].items():
            rows.append(
                {
                    "setting": run["setting"],
                    "seed": run["seed"],
                    "model": model,
                    **{k: metrics[k] for k in ["pr_auc", "roc_auc", "f1", "threshold"]},
                    "precision_100": metrics["top_k"]["100"]["precision"],
                    "recall_500": metrics["top_k"]["500"]["recall"],
                }
            )
    return pd.DataFrame(rows)


def predictions(setting, seed):
    return pd.read_csv(run_path(setting, seed) / "predictions.csv")


def precision_recall(labels, scores):
    """Threshold PR points with ties grouped (not a rank-order interpolation)."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    s = np.asarray(scores)[order]
    y = np.asarray(labels)[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    return np.r_[0, tp / y.sum()], np.r_[1, tp / (ends + 1)]


def cohort(nodes, tx, start, end):
    window = tx[tx.time.between(start, end)].copy()
    labels = nodes.set_index("nodeid").isFraud
    window["source_label"] = window.sourceNodeId.map(labels)
    window["target_label"] = window.targetNodeId.map(labels)
    return window


def neighborhood(tx, node, limit=40):
    incident = tx[(tx.sourceNodeId == node) | (tx.targetNodeId == node)]
    edges = (
        incident.groupby(["sourceNodeId", "targetNodeId"])
        .agg(count=("value", "size"), amount=("value", "sum"))
        .reset_index()
    )
    edges = edges.sort_values(
        ["count", "sourceNodeId", "targetNodeId"], ascending=[False, True, True]
    ).head(limit)
    return incident, edges
