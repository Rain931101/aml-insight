"""Post-hoc calibration diagnostics of existing scores, no calibration fitting or reselection."""

import numpy as np
import pandas as pd
from core import INPUT, REPORT, RUNS
from sklearn.metrics import brier_score_loss


def run():
    out = REPORT / "calibration_diagnostics"
    out.mkdir(exist_ok=False)
    summaries = []
    bins = []
    jobs = [
        ("AMLSim", RUNS / f"{kind}-full-seed{seed}", seed)
        for seed in range(5)
        for kind in ["gnn", "tabular"]
    ]
    jobs += [("Elliptic", INPUT / "elliptic/runs" / f"seed{seed}", seed) for seed in range(5)]
    for dataset, path, seed in jobs:
        p = pd.read_csv(path / "predictions.csv")
        y = p.label.to_numpy()
        for column in p.columns:
            if not column.endswith("_score") or column.startswith("rule_"):
                continue
            s = p[column].to_numpy()
            index = np.minimum((s * 10).astype(int), 9)
            ece = 0.0
            for b in range(10):
                take = index == b
                n = int(take.sum())
                if n:
                    mean = float(s[take].mean())
                    rate = float(y[take].mean())
                    ece += n / len(y) * abs(mean - rate)
                else:
                    mean = rate = None
                bins.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": column[:-6],
                        "bin": b,
                        "n": n,
                        "mean_score": mean,
                        "observed_positive_rate": rate,
                    }
                )
            summaries.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": column[:-6],
                    "brier_score": float(brier_score_loss(y, s)),
                    "ece_10_equal_width_bins": ece,
                    "prevalence_baseline_brier": float(np.mean((y - y.mean()) ** 2)),
                }
            )
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "bins.csv", index=False)
    (out / "README.md").write_text(
        "Post-hoc diagnostic only. Ten fixed equal-width bins; sparse bins may be unstable. No recalibration fitting, no threshold retuning. Class-weighted sigmoid scores are not presumed calibrated probabilities.\n"
    )


if __name__ == "__main__":
    run()
