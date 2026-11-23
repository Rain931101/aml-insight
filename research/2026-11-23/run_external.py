"""New-source Elliptic temporal comparison. Raw/derived rows stay in ignored local data."""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
import torch
from core import (
    INPUT,
    finish,
    fit_gnn,
    load,
    scale,
    scores,
    sha,
    snapshot_code,
    verify_protocol,
    write,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

from aml_insight.evaluation.metrics import select_threshold
from aml_insight.models.xgb import fit_xgb


def run(seed):
    verify_protocol()
    start = time.perf_counter()
    root = INPUT / "elliptic"
    receipt = load(root / "receipt.json")
    for name, h in receipt["files"].items():
        if sha(root / name) != h:
            raise ValueError(f"Changed external input {name}")
    out = root / "runs" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    f = pd.read_parquet(root / "features.parquet")
    idx = {k: np.array(v) for k, v in load(root / "split.json").items()}
    cols = [f"local_{i}" for i in range(93)]
    raw = f[cols].to_numpy()
    y = f.label.to_numpy()
    x, _, mean, std = scale(raw, raw, idx["train"], signed_log=False)
    np.savez(out / "preprocess.npz", mean=mean, std=std)
    x_train = x.copy()
    x_train[f.time > 29] = 0
    x_val = x.copy()
    x_val[f.time > 34] = 0
    e = {k: np.load(root / f"{k}.npy") for k in ["train", "validation", "test"]}
    # All validation candidates see only their respective observation horizon.
    best, gnn_records = fit_gnn(
        x_train,
        e["train"],
        y,
        idx["train"],
        idx["validation"],
        seed=seed,
        x_val=x_val,
        edges_val=e["validation"],
    )
    models = {"graphsage": best["model"]}
    val_scores = {"graphsage": scores(best["model"], x_val, e["validation"])[idx["validation"]]}
    selection = {
        "graphsage": {"params": best["params"], "epoch": best["epoch"], "candidates": gnn_records}
    }
    torch.save(
        {
            "state_dict": best["model"].state_dict(),
            "in_dim": 93,
            "params": best["params"],
            "dropout": 0.5,
        },
        out / "graphsage.pt",
    )
    grids = {
        "xgboost": list(
            ParameterGrid(
                {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1], "n_estimators": [100, 300]}
            )
        ),
        "random_forest": [{"max_depth": d} for d in [None, 12]],
        "logistic": [{"C": c} for c in [0.1, 1.0, 10.0]],
    }
    for kind, grid in grids.items():
        best_ap = -1
        records = []
        xx = x if kind == "logistic" else raw
        for params in grid:
            if kind == "xgboost":
                m = fit_xgb(xx[idx["train"]], y[idx["train"]], params, seed)
            elif kind == "random_forest":
                m = RandomForestClassifier(
                    n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1, **params
                )
                m.fit(xx[idx["train"]], y[idx["train"]])
            else:
                m = LogisticRegression(
                    class_weight="balanced", max_iter=3000, random_state=seed, **params
                )
                m.fit(xx[idx["train"]], y[idx["train"]])
            v = m.predict_proba(xx[idx["validation"]])[:, 1]
            ap = float(average_precision_score(y[idx["validation"]], v))
            records.append({"params": params, "validation_ap": ap})
            if ap > best_ap:
                best_ap = ap
                models[kind] = m
                val_scores[kind] = v
                selection[kind] = {"params": params, "validation_ap": ap}
        selection[kind]["candidates"] = records
        joblib.dump(models[kind], out / f"{kind}.joblib")
    thresholds = {k: select_threshold(y[idx["validation"]], v) for k, v in val_scores.items()}
    write(
        out / "selection.json",
        {"models": selection, "thresholds": thresholds, "test_metrics_not_consulted": True},
    )
    pd.DataFrame({"nodeid": f.nodeid.iloc[idx["validation"]], **val_scores}).to_csv(
        out / "validation.csv", index=False
    )
    score_map = {}
    for kind, m in models.items():
        score_map[kind] = (
            scores(m, x, e["test"])[idx["test"]]
            if kind == "graphsage"
            else m.predict_proba((x if kind == "logistic" else raw)[idx["test"]])[:, 1]
        )
    finish(
        out,
        f.nodeid.to_numpy()[idx["test"]],
        y[idx["test"]],
        score_map,
        thresholds,
        {
            "dataset": "Elliptic",
            "seed": seed,
            "split": "time1-29/30-34/35-49",
            "status": "new_source_method_replication_not_banking_account_transfer",
            "unknown_not_negative": True,
            "published_preprocessing_unverifiable": True,
            "raw_receipt_sha": sha(root / "receipt.json"),
            "seconds_training_inference": time.perf_counter() - start,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int, choices=range(5))
    args = p.parse_args()
    run(args.seed)
