"""True GNNExplainer on saved follow-up checkpoints; no refitting or threshold tuning."""

import itertools
import math
import time

import numpy as np
import pandas as pd
import shap
import torch
from core import INPUT, PLAN, REPORT, RUNS, load, sha, write
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph
from xgboost import XGBClassifier

from aml_insight.models.graphsage import GraphSAGE


class CenteredModel(torch.nn.Module):
    """Center logits at the validation-selected cutoff; does not refit the model."""

    def __init__(self, base, threshold):
        super().__init__()
        self.base = base
        self.offset = float(np.log(threshold / (1 - threshold)))

    def forward(self, x, edge_index):
        return self.base(x, edge_index) - self.offset


def pair_scores(e, mask):
    pairs = {}
    for (u, v), score in zip(e.T, mask):
        pair = tuple(sorted((int(u), int(v))))
        pairs.setdefault(pair, []).append(float(score))
    return {p: float(np.mean(v)) for p, v in pairs.items()}


def run():
    torch.set_num_threads(1)
    gdir = RUNS / "gnn-full-seed0"
    tdir = RUNS / "tabular-full-seed0"
    if not (gdir / "COMPLETE.json").exists() or not (tdir / "COMPLETE.json").exists():
        raise ValueError("Both frozen follow-up baseline runs must complete first")
    out = REPORT / "gnn_explanations"
    out.mkdir(exist_ok=False)
    pred = pd.read_csv(tdir / "predictions.csv").set_index("nodeid")
    txgb = load(tdir / "selection.json")["threshold"]
    cases = []
    for name, y, p in [("TP", 1, 1), ("FP", 0, 1), ("FN", 1, 0), ("TN", 0, 0)]:
        group = pred[(pred.label == y) & (pred.xgboost_pred == p)].copy()
        group["distance"] = abs(group.xgboost_score - txgb)
        for node in group.reset_index().sort_values(["distance", "nodeid"]).head(2).nodeid:
            cases.append({"case": name, "nodeid": int(node)})
    write(
        out / "case_selection.json",
        {
            "rule": "two nearest validation cutoff per XGB TP/FP/FN/TN; tie nodeid",
            "cases": cases,
            "missing_categories": sorted({"TP", "FP", "FN", "TN"} - {c["case"] for c in cases}),
        },
    )
    f = pd.read_parquet(INPUT / "full.parquet").set_index("nodeid")
    cols = load(gdir / "columns.json")
    manifest = load(gdir / "manifest.json")
    prep = np.load(gdir / "preprocess.npz")
    raw = f[cols].to_numpy()
    logged = np.sign(raw) * np.log1p(abs(raw))
    x = torch.tensor(((logged - prep["mean"]) / prep["std"]).astype("float32"))
    e = torch.tensor(np.load(INPUT / "full.npy"), dtype=torch.long)
    ckpt = torch.load(gdir / "model.pt", map_location="cpu", weights_only=True)
    model = GraphSAGE(ckpt["in_dim"], ckpt["params"]["hidden"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = load(gdir / "selection.json")["threshold"]
    centered = CenteredModel(model, threshold)
    centered.eval()
    with torch.no_grad():
        full_logits = model(x, e)
        full_scores = torch.sigmoid(full_logits).numpy()
    xm = XGBClassifier()
    xm.load_model(tdir / "xgboost.ubj")
    xcols = load(tdir / "columns.json")
    bg = f.loc[manifest["train"], xcols].sample(n=100, random_state=0)
    case_features = f.loc[[c["nodeid"] for c in cases], xcols]
    sx = shap.TreeExplainer(
        xm, data=bg, model_output="probability", feature_perturbation="interventional"
    )(case_features)
    pd.DataFrame(sx.values, index=case_features.index, columns=xcols).to_csv(
        out / "paired_xgb_shap.csv", index_label="nodeid"
    )
    audit = []
    sets = {}
    latencies = []
    for case in cases:
        node = case["nodeid"]
        position = f.index.get_loc(node)
        subset, es, mapping, _ = k_hop_subgraph(
            position, 2, e, relabel_nodes=True, num_nodes=len(f)
        )
        xs = x[subset]
        target = int(mapping[0])
        directory = out / str(node)
        directory.mkdir()
        with torch.no_grad():
            sublogit = float(model(xs, es)[target])
            full = float(full_logits[position])
        if abs(sublogit - full) > 1e-5:
            raise ValueError("Two-hop target logit does not match full graph")
        predicted = int(full_scores[position] >= threshold)
        2 * predicted - 1
        orig = float(full_scores[position] if predicted else 1 - full_scores[position])
        for seed in [0, 1, 2]:
            torch.manual_seed(seed)
            start = time.perf_counter()
            explainer = Explainer(
                model=centered,
                algorithm=GNNExplainer(epochs=PLAN["explanation_epochs"]),
                explanation_type="model",
                node_mask_type=None,
                edge_mask_type="object",
                model_config={
                    "mode": "binary_classification",
                    "task_level": "node",
                    "return_type": "raw",
                },
            )
            explanation = explainer(xs, es, index=target)
            edge_mask = explanation.edge_mask.detach().numpy()
            paired = pair_scores(es.numpy(), edge_mask)
            ordered = sorted(paired, key=lambda pair: (-paired[pair], pair))
            k = max(1, math.ceil(0.2 * len(ordered))) if ordered else 0
            top = set(ordered[:k])
            sets[node, seed] = top

            def confidence_drop(
                removed, es=es, xs=xs, target=target, orig=orig, predicted=predicted
            ):
                if model.training:
                    raise ValueError("Dropout must remain disabled during explanation fidelity")
                keep = torch.tensor(
                    [tuple(sorted((int(u), int(v)))) not in removed for u, v in es.T]
                )
                with torch.no_grad():
                    p = float(torch.sigmoid(model(xs, es[:, keep])[target]))
                return orig - (p if predicted else 1 - p)

            top_drop = confidence_drop(top) if k else None
            random = []
            for rseed in range(10):
                rng = np.random.default_rng(rseed)
                idx = rng.choice(len(ordered), k, replace=False)
                random.append(confidence_drop({ordered[i] for i in idx}) if k else 0.0)
            pd.DataFrame(
                {
                    "source_nodeid": f.index.to_numpy()[subset.numpy()[es[0].numpy()]],
                    "target_nodeid": f.index.to_numpy()[subset.numpy()[es[1].numpy()]],
                    "importance": edge_mask,
                }
            ).to_csv(directory / f"edge_mask_seed{seed}.csv", index=False)
            record = {
                **case,
                "explanation_seed": seed,
                "gnn_predicted_class": predicted,
                "actual_label": int(f.loc[node, "isFraud"]),
                "gnn_score": float(full_scores[position]),
                "gnn_threshold": threshold,
                "xgb_score": float(pred.loc[node, "xgboost_score"]),
                "subgraph_nodes": len(subset),
                "subgraph_message_edges": es.shape[1],
                "pairs": len(ordered),
                "positive_mask_pairs": sum(v > 0 for v in paired.values()),
                "removed_pairs": k,
                "target_confidence_drop": top_drop,
                "random_mean_drop": float(np.mean(random)),
                "excess_drop": top_drop - float(np.mean(random)) if k else None,
                "logit_subgraph_error": abs(sublogit - full),
                "seconds": time.perf_counter() - start,
            }
            audit.append(record)
            print(record, flush=True)
        # Actual local CPU inference timings, not reviewer investigation times.
        for _ in range(5):
            model(xs, es)
        for _ in range(30):
            start = time.perf_counter()
            with torch.no_grad():
                model(xs, es)
            latencies.append(
                {
                    "nodeid": node,
                    "seconds": time.perf_counter() - start,
                    "nodes": len(subset),
                    "edges": es.shape[1],
                }
            )
    stability = []
    for case in cases:
        for a, b in itertools.combinations(range(3), 2):
            aa, bb = sets[case["nodeid"], a], sets[case["nodeid"], b]
            stability.append(
                {
                    "nodeid": case["nodeid"],
                    "seed_a": a,
                    "seed_b": b,
                    "top20_pair_jaccard": len(aa & bb) / len(aa | bb) if aa | bb else None,
                }
            )
    pd.DataFrame(audit).to_csv(out / "masking.csv", index=False)
    pd.DataFrame(stability).to_csv(out / "stability.csv", index=False)
    pd.DataFrame(latencies).to_csv(out / "local_inference_timings.csv", index=False)
    write(
        out / "receipt.json",
        {
            "gnn_checkpoint_sha": sha(gdir / "model.pt"),
            "xgb_checkpoint_sha": sha(tdir / "xgboost.ubj"),
            "explanation_target": "raw logit centered at locked validation threshold; model prediction not ground truth",
            "limitations": [
                "Exploratory boundary cases, not representative population",
                "Masking is not causal",
                "Zero-mask pair ties are broken by node ID; report active-mask count alongside stability",
                "Subgraph timing excludes loading/network/browser/human review",
            ],
            "files": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
        },
    )


if __name__ == "__main__":
    run()
