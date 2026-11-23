"""Frozen-score subgroup and review-budget analysis; never tunes on test."""

import numpy as np
import pandas as pd
from core import REPORT, ROOT


def run():
    tx = pd.read_csv(ROOT / "data/raw/20K_fanin200cycle200/transactions.csv")
    small = tx[tx.value < 100]
    flagged = set(small.sourceNodeId) | set(small.targetNodeId)
    rows = []
    budgets = []
    for setting in ["S1", "S2"]:
        for seed in range(5):
            p = pd.read_csv(
                ROOT
                / f"experiments/member-b-amlsim-e1-20261116/E1-{setting}-seed{seed}/predictions.csv"
            )
            p["small_amount_incident"] = p.nodeid.isin(flagged)
            for group, take in [
                ("has_small", p.small_amount_incident),
                ("no_small", ~p.small_amount_incident),
            ]:
                f = p[take]
                positive = int(f.isFraud.sum())
                for m in ["xgboost", "graphsage", "rule_based"]:
                    tp = int(((f.isFraud == 1) & (f[f"{m}_pred"] == 1)).sum())
                    fp = int(((f.isFraud == 0) & (f[f"{m}_pred"] == 1)).sum())
                    rows.append(
                        {
                            "setting": setting,
                            "seed": seed,
                            "model": m,
                            "group": group,
                            "n": len(f),
                            "positives": positive,
                            "true_positives": tp,
                            "false_positives": fp,
                            "recall": tp / positive if positive else None,
                            "false_positive_rate": fp / int((f.isFraud == 0).sum())
                            if (f.isFraud == 0).any()
                            else None,
                        }
                    )
            for m in ["xgboost", "graphsage", "rule_based"]:
                order = p.sort_values([f"{m}_score", "nodeid"], ascending=[False, True])
                for fraction in [0.01, 0.05, 0.10, 0.125]:
                    k = int(np.ceil(len(p) * fraction))
                    part = order.head(k)
                    hits = int(part.isFraud.sum())
                    budgets.append(
                        {
                            "setting": setting,
                            "seed": seed,
                            "model": m,
                            "budget_fraction": fraction,
                            "reviewed": k,
                            "hits": hits,
                            "false_alerts": k - hits,
                            "precision": hits / k,
                            "recall": hits / int(p.isFraud.sum()),
                            "reviews_per_positive": k / hits if hits else None,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(REPORT / "small_amount_subgroups.csv", index=False)
    df.groupby(["setting", "model", "group"])[["recall", "false_positive_rate", "positives"]].agg(
        ["mean", "std", "min", "max"]
    ).to_csv(REPORT / "subgroup_summary.csv")
    pd.DataFrame(budgets).to_csv(REPORT / "review_budgets.csv", index=False)
    print("Existing-score subgroups and budgets saved; no new test selection")


if __name__ == "__main__":
    run()
