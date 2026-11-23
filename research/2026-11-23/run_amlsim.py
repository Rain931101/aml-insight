"""One exploration job, exclusive outputs. Select with validation, then seal test once."""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from core import (
    INPUT,
    PLAN,
    finish,
    fit_gnn,
    load,
    new_run,
    scale,
    scores,
    sha,
    write,
)
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

from aml_insight.data.split import stratified_split
from aml_insight.evaluation.metrics import select_threshold
from aml_insight.features.account import FEATURE_GROUPS
from aml_insight.models.xgb import fit_xgb


def inputs(variant, seed):
    for name, h in load(INPUT / "inputs.json")["files"].items():
        if sha(INPUT / name) != h:
            raise ValueError(f"Input changed {name}")
    early = variant == "early_late"
    f = pd.read_parquet(INPUT / ("early.parquet" if early else "full.parquet")).set_index("nodeid")
    late = pd.read_parquet(INPUT / "late.parquet").set_index("nodeid") if early else f
    columns = [
        *FEATURE_GROUPS["behavior"],
        *FEATURE_GROUPS["temporal"],
        *FEATURE_GROUPS["structure"],
    ]
    if variant.startswith("no_") and variant not in ["no_edges"]:
        group = {"no_structure": "structure", "no_behavior": "behavior", "no_temporal": "temporal"}[
            variant
        ]
        columns = [c for c in columns if c not in FEATURE_GROUPS[group]]
    manifest = stratified_split(f.index.to_numpy(), f.isFraud.to_numpy(), seed)
    idx = {k: f.index.get_indexer(v) for k, v in manifest.items()}
    return f, late, columns, manifest, idx


def gnn(variant, seed):
    start = time.perf_counter()
    f, late, cols, manifest, idx = inputs(variant, seed)
    out = new_run(f"gnn-{variant}-seed{seed}")
    x, xt, mean, std = scale(f[cols], late[cols], idx["train"])
    np.savez(out / "preprocess.npz", mean=mean, std=std)
    edge_name = (
        f"{variant}-{seed}"
        if variant in ["rewire", "drop50"]
        else "early"
        if variant == "early_late"
        else "no_edges"
        if variant == "no_edges"
        else "full"
    )
    e = np.load(INPUT / f"{edge_name}.npy")
    et = np.load(INPUT / "late.npy") if variant == "early_late" else e
    write(out / "manifest.json", manifest)
    write(out / "columns.json", cols)
    best, records = fit_gnn(x, e, f.isFraud.to_numpy(), idx["train"], idx["validation"], seed=seed)
    val = scores(best["model"], x, e)[idx["validation"]]
    threshold = select_threshold(f.isFraud.to_numpy()[idx["validation"]], val)
    write(
        out / "selection.json",
        {
            "params": best["params"],
            "epoch": best["epoch"],
            "threshold": threshold,
            "candidates": records,
        },
    )
    torch.save(
        {
            "state_dict": best["model"].state_dict(),
            "in_dim": len(cols),
            "params": best["params"],
            "dropout": 0.5,
        },
        out / "model.pt",
    )
    pd.DataFrame({"nodeid": manifest["validation"], "score": val}).to_csv(
        out / "validation.csv", index=False
    )
    s = scores(best["model"], xt, et)[idx["test"]]
    finish(
        out,
        manifest["test"],
        f.isFraud.to_numpy()[idx["test"]],
        {"graphsage": s},
        {"graphsage": threshold},
        {
            "dataset": "AMLSim",
            "setting": "S2",
            "variant": variant,
            "seed": seed,
            "status": "exploratory_seen_test",
            "epochs_per_candidate": PLAN["gnn_epochs"],
            "edge_input": edge_name,
            "input_receipt_sha": sha(INPUT / "inputs.json"),
            "seconds_training_inference": time.perf_counter() - start,
        },
    )


def tabular(variant, seed):
    start = time.perf_counter()
    f, late, cols, manifest, idx = inputs(variant, seed)
    out = new_run(f"tabular-{variant}-seed{seed}")
    write(out / "manifest.json", manifest)
    write(out / "columns.json", cols)
    best = None
    records = []
    grid = {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1], "n_estimators": [100, 300]}
    for params in ParameterGrid(grid):
        m = fit_xgb(f[cols].iloc[idx["train"]], f.isFraud.iloc[idx["train"]], params, seed)
        v = m.predict_proba(f[cols].iloc[idx["validation"]])[:, 1]
        ap = float(average_precision_score(f.isFraud.iloc[idx["validation"]], v))
        records.append({"params": params, "validation_ap": ap})
        if best is None or ap > best["ap"]:
            best = {"model": m, "ap": ap, "params": params, "v": v}
    m = best["model"]
    threshold = select_threshold(f.isFraud.iloc[idx["validation"]], best["v"])
    write(
        out / "selection.json",
        {"params": best["params"], "threshold": threshold, "candidates": records},
    )
    m.save_model(out / "xgboost.ubj")
    score_map = {"xgboost": m.predict_proba(late[cols].iloc[idx["test"]])[:, 1]}
    thresholds = {"xgboost": threshold}
    if variant in ["full", "early_late"]:
        rulecols = ["in_tx_count", "out_tx_count", "in_degree"]
        quantiles = f[rulecols].iloc[idx["train"]].quantile(0.95)

        def rule(frame):
            return (frame[rulecols] >= quantiles).mean(axis=1).to_numpy()

        rv = rule(f.iloc[idx["validation"]])
        rt = select_threshold(f.isFraud.iloc[idx["validation"]], rv)
        score_map["rule_train_quantile"] = rule(late.iloc[idx["test"]])
        thresholds["rule_train_quantile"] = rt
        write(out / "rule.json", {"train_quantiles": quantiles.to_dict(), "val_f1_threshold": rt})
    finish(
        out,
        manifest["test"],
        f.isFraud.to_numpy()[idx["test"]],
        score_map,
        thresholds,
        {
            "dataset": "AMLSim",
            "setting": "S2",
            "variant": variant,
            "seed": seed,
            "status": "exploratory_seen_test",
            "input_receipt_sha": sha(INPUT / "inputs.json"),
            "seconds_training_inference": time.perf_counter() - start,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["gnn", "tabular"])
    p.add_argument("variant")
    p.add_argument("seed", type=int)
    a = p.parse_args()
    if a.seed not in PLAN["seeds"]:
        raise ValueError("unplanned seed")
    allowed = PLAN["variants"] if a.model == "gnn" else PLAN["xgb_variants"]
    if a.variant not in allowed:
        raise ValueError("unplanned variant")
    (gnn if a.model == "gnn" else tabular)(a.variant, a.seed)
