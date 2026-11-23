"""Scientific figures from completed explanation diagnostics, no model execution."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core import REPORT


def run():
    root = REPORT / "figures"
    root.mkdir(exist_ok=True)
    mask = pd.read_csv(REPORT / "gnn_explanations/masking.csv")
    stability = pd.read_csv(REPORT / "gnn_explanations/stability.csv")
    group = mask.groupby("nodeid").excess_drop.agg(["mean", "std"])
    ids = group.index.to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    a.bar(
        np.arange(len(ids)),
        group["mean"],
        yerr=group["std"],
        capsize=3,
        color=["#be705c" if v < 0 else "#7764cb" for v in group["mean"]],
    )
    a.axhline(0, color="#555555", lw=0.8)
    a.set(
        xticks=np.arange(len(ids)),
        xticklabels=ids,
        ylabel="Confidence drop minus random control",
        title="GNNExplainer: 3 mask seeds per case",
    )
    a.tick_params(axis="x", labelrotation=45)
    for i, node in enumerate(ids):
        vals = stability[stability.nodeid == node].top20_pair_jaccard.to_numpy()
        b.scatter(np.repeat(i, len(vals)), vals, color="#39a58f", s=24)
    b.set(
        xticks=np.arange(len(ids)),
        xticklabels=ids,
        ylim=(0, 1),
        ylabel="Top-20% pair Jaccard",
        title="Pairwise stability across explanation seeds",
    )
    b.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(root / "gnn_explanation_audit.png", dpi=180)
    plt.close(fig)
    f = pd.read_csv(REPORT / "shap_audit/directional_summary.csv")
    f = f[f.setting == "S2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, color in [(0, "#7764cb"), (1, "#39a58f")]:
        x = f[f.predicted_class == c]
        ax.plot(
            x.requested_k,
            x.mean_target_drop,
            "o-",
            label=f"Predicted {c}: SHAP support",
            color=color,
        )
        ax.plot(
            x.requested_k,
            x.mean_random_drop,
            "o--",
            label=f"Predicted {c}: random control",
            color=color,
            alpha=0.6,
        )
    ax.set(
        xlabel="Requested features removed (positive support only)",
        ylabel="Original-class confidence drop",
        title="S2: direction-aware SHAP masking; 200 sampled test accounts",
        xticks=[1, 3, 5],
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "shap_directional_audit.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
