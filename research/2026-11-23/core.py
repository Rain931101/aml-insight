"""Frozen experiment helpers. No imports or actions alter prior sealed experiments."""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from aml_insight.evaluation.metrics import evaluate
from aml_insight.models.graphsage import GraphSAGE

HERE = Path(__file__).resolve().parent
INPUT = ROOT / "data/processed/research_extension_20261123"
RUNS = ROOT / "experiments/research-extension-20261123"
REPORT = ROOT / "reports/research_extension_2026-11-23"
PLAN = json.loads((HERE / "plan.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def verify_protocol():
    for name, h in load(HERE / "protocol_lock.json")["files"].items():
        if sha(HERE / name) != h:
            raise ValueError("Protocol changed after freezing")


def snapshot_code(out):
    files = [*HERE.glob("*.py"), *HERE.glob("*.json"), HERE / "protocol.md"]
    files += list((ROOT / "src/aml_insight").rglob("*.py"))
    write(out / "source_hashes.json", {str(f.relative_to(ROOT)): sha(f) for f in files})


def scale(train, test, train_rows, signed_log=True):
    a = np.array(train, dtype=np.float64)
    b = np.array(test, dtype=np.float64)
    if signed_log:
        a = np.sign(a) * np.log1p(abs(a))
        b = np.sign(b) * np.log1p(abs(b))
    mean = a[train_rows].mean(0)
    std = a[train_rows].std(0)
    std[std == 0] = 1
    return ((a - mean) / std).astype("float32"), ((b - mean) / std).astype("float32"), mean, std


def fit_gnn(x, edges, y, train_idx, val_idx, *, seed, x_val=None, edges_val=None):
    import copy
    import time

    torch.set_num_threads(1)
    x = torch.tensor(x)
    edges = torch.tensor(edges, dtype=torch.long)
    xv = x if x_val is None else torch.tensor(x_val)
    ev = edges if edges_val is None else torch.tensor(edges_val, dtype=torch.long)
    yy = torch.tensor(y, dtype=torch.float32)
    ti = torch.tensor(train_idx)
    vi = torch.tensor(val_idx)
    weight = torch.tensor(float((yy[ti] == 0).sum() / (yy[ti] == 1).sum()))
    candidates = []
    winner = None
    for params in PLAN["gnn_grid"]:
        torch.manual_seed(seed)
        model = GraphSAGE(x.shape[1], params["hidden"], 0.5)
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=0.0005)
        best_ap = -1
        best_state = None
        best_epoch = None
        curve = []
        start = time.perf_counter()
        for epoch in range(1, PLAN["gnn_epochs"] + 1):
            model.train()
            opt.zero_grad()
            logits = model(x, edges)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[ti], yy[ti], pos_weight=weight
            )
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(xv, ev)[vi]).numpy()
            ap = float(average_precision_score(np.asarray(y)[val_idx], scores))
            curve.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_ap": ap})
            if ap > best_ap:
                best_ap = ap
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        candidates.append(
            {
                "params": params,
                "selected_epoch": best_epoch,
                "validation_ap": best_ap,
                "seconds": time.perf_counter() - start,
                "curve": curve,
            }
        )
        if winner is None or best_ap > winner["validation_ap"]:
            model.load_state_dict(best_state)
            winner = {
                "model": model,
                "params": params,
                "epoch": best_epoch,
                "validation_ap": best_ap,
            }
    return winner, candidates


def scores(model, x, edge):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x), torch.tensor(edge, dtype=torch.long))).numpy()


def finish(out, ids, y, score_map, thresholds, metadata):
    # Exclusive seal is acquired before computing any test metric.
    with (out / "TEST_STARTED.json").open("x") as f:
        json.dump(metadata, f, indent=2)
    table = pd.DataFrame({"nodeid": ids, "label": y})
    metrics = {}
    for name, score in score_map.items():
        ks = tuple(
            sorted({100, 500, *[max(1, int(np.ceil(len(y) * p))) for p in [0.01, 0.05, 0.10]]})
        )
        metrics[name] = evaluate(y, score, thresholds[name], ids, ks=ks)
        table[f"{name}_score"] = score
        table[f"{name}_pred"] = (score >= thresholds[name]).astype(int)
    table.to_csv(out / "predictions.csv", index=False)
    write(out / "metrics.json", metrics)
    write(out / "COMPLETE.json", metadata)
    write(
        out / "result_hashes.json",
        {f.name: sha(f) for f in out.iterdir() if f.is_file() and f.name != "result_hashes.json"},
    )
    print(
        json.dumps(
            {"run": out.name, "AP": {m: r["pr_auc"] for m, r in metrics.items()}},
            ensure_ascii=False,
        ),
        flush=True,
    )


def new_run(name):
    verify_protocol()
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=False)
    snapshot_code(out)
    return out
