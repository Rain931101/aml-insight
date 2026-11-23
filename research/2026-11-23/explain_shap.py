"""Direction-aware follow-up on locked XGB models, no training or test reselection."""

import itertools

import numpy as np
import pandas as pd
import shap
from core import REPORT, ROOT, load, sha, write
from scipy.stats import spearmanr
from xgboost import XGBClassifier

from aml_insight.evaluation.artifacts import verify


def run():
    out = REPORT / "shap_audit"
    out.mkdir(exist_ok=False)
    all_mask = []
    stability = []
    receipts = []
    for setting in ["S1", "S2"]:
        root = ROOT / f"experiments/member-b-amlsim-e1-20261116/E1-{setting}-seed0"
        verify(root, check_source=False)
        verify(root, check_source=False, manifest="result_integrity.json")
        verify(root / "shap", check_source=False)
        cols = load(root / "columns.json")
        train = pd.read_parquet(root / "train.parquet")[cols]
        features = pd.read_csv(root / "shap/sample_features.csv").set_index("nodeid")[cols]
        test = pd.read_parquet(root / "test.parquet")
        labels = test.isFraud.reindex(features.index)
        model = XGBClassifier()
        model.load_model(root / "xgboost.ubj")
        p = model.predict_proba(features)[:, 1]
        threshold = load(root / "metrics.json")["models"]["xgboost"]["threshold"]
        predicted = (p >= threshold).astype(int)
        direction = 2 * predicted - 1
        values = (
            pd.read_csv(root / "shap/values.csv")
            .set_index("nodeid")
            .reindex(features.index)[cols]
            .to_numpy()
        )
        median = train.median().to_numpy()
        original = np.where(predicted == 1, p, 1 - p)
        for k in [1, 3, 5]:
            rows = []
            for i, node in enumerate(features.index):
                support = values[i] * direction[i]
                order = np.argsort(-support, kind="stable")
                chosen = order[support[order] > 0][:k]
                if not len(chosen):
                    continue
                masked = features.iloc[[i]].copy()
                masked.iloc[0, chosen] = median[chosen]
                pp = model.predict_proba(masked)[:, 1][0]
                confidence = pp if predicted[i] else 1 - pp
                random = []
                for seed in range(10):
                    rng = np.random.default_rng(seed * 100000 + int(node))
                    selected = rng.choice(len(cols), len(chosen), replace=False)
                    m = features.iloc[[i]].copy()
                    m.iloc[0, selected] = median[selected]
                    q = model.predict_proba(m)[:, 1][0]
                    random.append(original[i] - (q if predicted[i] else 1 - q))
                row = {
                    "setting": setting,
                    "nodeid": int(node),
                    "label": int(labels.loc[node]),
                    "predicted_class": int(predicted[i]),
                    "requested_k": k,
                    "actual_k": len(chosen),
                    "target_confidence_drop": original[i] - confidence,
                    "random_mean_drop": float(np.mean(random)),
                    "random_sd": float(np.std(random, ddof=1)),
                    "paired_excess_drop": original[i] - confidence - float(np.mean(random)),
                }
                rows.append(row)
            all_mask.extend(rows)
        variants = {0: values}
        for seed in [1, 2]:
            bg = train.sample(n=min(100, len(train)), random_state=seed)
            exp = shap.TreeExplainer(
                model, data=bg, model_output="probability", feature_perturbation="interventional"
            )(features)
            error = float(np.max(abs(exp.base_values + exp.values.sum(1) - p)))
            if error > 1e-4:
                raise ValueError("SHAP additivity failed")
            variants[seed] = exp.values
            pd.DataFrame(exp.values, index=features.index, columns=cols).to_csv(
                out / f"{setting}-background{seed}.csv", index_label="nodeid"
            )
        for a, b in itertools.combinations(variants, 2):
            for i, node in enumerate(features.index):
                va, vb = abs(variants[a][i]), abs(variants[b][i])
                k = min(5, len(cols))
                aa = set(np.argsort(-va, kind="stable")[:k])
                bb = set(np.argsort(-vb, kind="stable")[:k])
                rho = float(spearmanr(va, vb).statistic)
                stability.append(
                    {
                        "setting": setting,
                        "nodeid": int(node),
                        "background_a": a,
                        "background_b": b,
                        "top5_jaccard": len(aa & bb) / len(aa | bb),
                        "abs_rank_spearman": rho if np.isfinite(rho) else None,
                    }
                )
        receipts.append(
            {
                "setting": setting,
                "model_sha": sha(root / "xgboost.ubj"),
                "original_shap_receipt_sha": sha(root / "shap/integrity.json"),
                "n": len(features),
                "threshold": threshold,
            }
        )
    mask = pd.DataFrame(all_mask)
    mask.to_csv(out / "directional_masking.csv", index=False)
    mask.groupby(["setting", "predicted_class", "requested_k"]).agg(
        n=("nodeid", "size"),
        mean_target_drop=("target_confidence_drop", "mean"),
        mean_random_drop=("random_mean_drop", "mean"),
        mean_excess=("paired_excess_drop", "mean"),
        median_excess=("paired_excess_drop", "median"),
    ).to_csv(out / "directional_summary.csv")
    stab = pd.DataFrame(stability)
    stab.to_csv(out / "background_stability.csv", index=False)
    stab.groupby("setting")[["top5_jaccard", "abs_rank_spearman"]].agg(
        ["mean", "median", "min"]
    ).to_csv(out / "stability_summary.csv")
    write(
        out / "receipt.json",
        {
            "models": receipts,
            "limitation": "Post-hoc exploration on seen test; model-confidence masking, not causal or human validation",
            "files": {f.name: sha(f) for f in out.iterdir() if f.is_file()},
        },
    )
    print("SHAP direction and background audit saved", flush=True)


if __name__ == "__main__":
    run()
