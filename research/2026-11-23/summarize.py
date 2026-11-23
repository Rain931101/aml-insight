"""Verify all sealed outputs, then produce descriptive tables and standalone figures."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import HERE, INPUT, PLAN, REPORT, ROOT, RUNS, load, sha, verify_protocol, write


def verify(path):
    for name, h in load(path / "result_hashes.json").items():
        if sha(path / name) != h:
            raise ValueError(f"Changed result {path}/{name}")
    if not (path / "COMPLETE.json").exists():
        raise ValueError("Incomplete run")


def rows(path, kind, variant, seed):
    verify(path)
    metrics = load(path / "metrics.json")
    p = pd.read_csv(path / "predictions.csv")
    output = []
    for model, m in metrics.items():
        if len(p) != m["n"] or int(p.label.sum()) != m["positives"]:
            raise ValueError("Metrics/predictions mismatch")
        record = {
            "dataset": kind,
            "variant": variant,
            "seed": seed,
            "model": model,
            **{k: m[k] for k in ["pr_auc", "roc_auc", "f1", "threshold", "n", "positives"]},
        }
        for k in [100, 500]:
            record[f"precision_{k}"] = m["top_k"][str(k)]["precision"]
            record[f"recall_{k}"] = m["top_k"][str(k)]["recall"]
        for name, fraction in [("1pct", 0.01), ("5pct", 0.05), ("10pct", 0.1)]:
            k = max(1, int(np.ceil(len(p) * fraction)))
            r = m["top_k"][str(k)]
            record[f"precision_{name}"] = r["precision"]
            record[f"recall_{name}"] = r["recall"]
        record["source"] = str(path.relative_to(ROOT))
        output.append(record)
    return output


def run():
    verify_protocol()
    records = []
    sources = []
    for model, variants in [("gnn", PLAN["variants"]), ("tabular", PLAN["xgb_variants"])]:
        for variant in variants:
            for seed in PLAN["seeds"]:
                path = RUNS / f"{model}-{variant}-seed{seed}"
                records += rows(path, "AMLSim", variant, seed)
                sources.append(
                    {"path": str(path.relative_to(ROOT)), "metrics_sha": sha(path / "metrics.json")}
                )
    for seed in PLAN["seeds"]:
        path = INPUT / "elliptic/runs" / f"seed{seed}"
        records += rows(path, "Elliptic", "temporal", seed)
        sources.append(
            {"path": str(path.relative_to(ROOT)), "metrics_sha": sha(path / "metrics.json")}
        )
    df = pd.DataFrame(records)
    if len(df) != 85:
        raise ValueError(f"Expected 85 model records, got {len(df)}")
    metrics = [
        "pr_auc",
        "roc_auc",
        "f1",
        "precision_100",
        "recall_500",
        "recall_1pct",
        "recall_5pct",
        "recall_10pct",
    ]
    summary = df.groupby(["dataset", "variant", "model"])[metrics].agg(
        ["mean", "std", "min", "max", "count"]
    )
    df.to_csv(REPORT / "all_metrics.csv", index=False)
    summary.to_csv(REPORT / "summary.csv")
    differences = []
    a = df[(df.dataset == "AMLSim") & (df.model == "graphsage")]
    base = a[a.variant == "full"].set_index("seed")
    for variant in PLAN["variants"]:
        v = a[a.variant == variant].set_index("seed")
        for seed in PLAN["seeds"]:
            differences.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "ap_difference_vs_full": v.loc[seed, "pr_auc"] - base.loc[seed, "pr_auc"],
                    "r500_difference_vs_full": v.loc[seed, "recall_500"]
                    - base.loc[seed, "recall_500"],
                }
            )
    pd.DataFrame(differences).to_csv(REPORT / "paired_graph_differences.csv", index=False)
    write(REPORT / "sources.json", sources)
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    order = PLAN["variants"]
    group = a.groupby("variant").pr_auc
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        np.arange(len(order)),
        group.mean().reindex(order),
        yerr=group.std().reindex(order),
        color="#7764cb",
        capsize=4,
    )
    ax.set(
        xticks=np.arange(len(order)),
        xticklabels=order,
        ylabel="Test average precision",
        ylim=(0, 1),
        title="AMLSim S2 follow-up: matched 200 epochs / 3 candidates / 5 seeds",
    )
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(figures / "graph_ablation.png", dpi=180)
    plt.close(fig)
    b = df[df.dataset == "Elliptic"]
    group = b.groupby("model").pr_auc
    names = ["logistic", "random_forest", "xgboost", "graphsage"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        names,
        group.mean().reindex(names),
        yerr=group.std().reindex(names),
        color=["#9098a8", "#ba9157", "#39a58f", "#7764cb"],
        capsize=4,
    )
    ax.set(
        ylabel="Known-label test average precision",
        ylim=(0, 1),
        title="Elliptic: time 1–29 / 30–34 / 35–49 (fixed split)",
    )
    fig.tight_layout()
    fig.savefig(figures / "external_comparison.png", dpi=180)
    plt.close(fig)
    sub = pd.read_csv(REPORT / "small_amount_subgroups.csv")
    sub = sub[sub.setting == "S2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, g in enumerate(["has_small", "no_small"]):
        grouped = sub[sub.group == g].groupby("model").recall
        names = ["rule_based", "xgboost", "graphsage"]
        ax.bar(
            np.arange(3) + (j - 0.5) * 0.35,
            grouped.mean().reindex(names),
            width=0.35,
            yerr=grouped.std().reindex(names),
            label=g,
            capsize=3,
            color=["#a9a5c5", "#7160bc"][j],
        )
    ax.set(
        xticks=np.arange(3),
        xticklabels=names,
        ylabel="Recall at locked validation cutoff",
        ylim=(0, 1),
        title="Previously sealed 2026-11-16 S2: stratified by small-amount exposure",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "hard_subgroups.png", dpi=180)
    plt.close(fig)
    expected = [
        "shap_audit/receipt.json",
        "gnn_explanations/receipt.json",
        "duplicate_sensitivity/receipt.json",
    ]
    if not all((REPORT / f).exists() for f in expected):
        raise ValueError("Explanatory/sensitivity studies incomplete")
    for directory in ["shap_audit", "gnn_explanations"]:
        for name, h in load(REPORT / directory / "receipt.json")["files"].items():
            if sha(REPORT / directory / name) != h:
                raise ValueError(f"Changed explanation {name}")
    write(
        REPORT / "study_complete.json",
        {
            "amlsim_runs": 55,
            "elliptic_runs": 5,
            "model_records": 85,
            "human_validation": "omitted at user request; no empirical human claims",
            "sources": sources,
            "protocol_lock_sha": sha(HERE / "protocol_lock.json"),
            "limitations": [
                "AMLSim follow-up uses previously seen test",
                "Elliptic task differs; no bank transfer claim",
                "Five seeds are not independent datasets",
                "Masking is not causal",
            ],
        },
    )
    print(summary[["pr_auc", "recall_500"]].to_string())


if __name__ == "__main__":
    run()
