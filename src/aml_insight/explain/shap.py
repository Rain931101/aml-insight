"""Training-background TreeSHAP and fixed-reference feature masking checks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aml_insight.evaluation.artifacts import read_json, seal, verify, write_json


def faithfulness(predict, features, train, values, ks=(1, 3, 5), seed=0):
    values = np.asarray(values)
    if values.shape != features.shape or list(features) != list(train):
        raise ValueError("SHAP values and training reference must align to feature matrix")
    baseline = train.median().to_numpy()
    original = np.asarray(predict(features))
    order = np.argsort(-np.abs(values), axis=1, kind="stable")
    rng = np.random.default_rng(seed)
    random_orders = np.array([rng.permutation(features.shape[1]) for _ in range(len(features))])
    results = []
    for k in sorted({min(int(k), features.shape[1]) for k in ks}):
        if k < 1:
            raise ValueError("Masking K must be positive")
        entry = {"k": k}
        for name, ranking in [("top", order), ("random", random_orders)]:
            masked = features.to_numpy(dtype=float).copy()
            rows = np.arange(len(features))[:, None]
            selected = ranking[:, :k]
            masked[rows, selected] = baseline[selected]
            score = np.asarray(
                predict(pd.DataFrame(masked, index=features.index, columns=features.columns))
            )
            drops = original - score
            prefix = "" if name == "top" else "random_"
            entry[prefix + "mean_score_drop"] = float(drops.mean())
            entry[prefix + "per_account_score_drop"] = drops.tolist()
        results.append(entry)
    return {
        "reference": "training_median",
        "seed": seed,
        "top_k": results,
        "original_scores": original.tolist(),
        "limitation": "Masking is not causal; correlated and out-of-distribution features "
        "may change scores. Drops can be negative for protective features.",
    }


def explain_run(run, sample_size=200, background_size=100):
    import shap
    from xgboost import XGBClassifier

    run = Path(run).resolve()
    verify(run)
    verify(run, check_source=False, manifest="result_integrity.json")
    if not (run / "COMPLETE.json").exists():
        raise ValueError("Finalize the run before explaining its fixed test predictions")
    if sample_size < 1 or background_size < 1:
        raise ValueError("Positive sample/background sizes required")
    cfg = read_json(run / "config.json")
    columns = read_json(run / "columns.json")
    seed = cfg["seed"]
    train = pd.read_parquet(run / "train.parquet")[columns]
    test = pd.read_parquet(run / "test.parquet")[columns]
    x = test.sample(n=min(sample_size, len(test)), random_state=seed)
    background = train.sample(n=min(background_size, len(train)), random_state=seed)
    model = XGBClassifier()
    model.load_model(run / "xgboost.ubj")
    out = run / "shap"
    out.mkdir(exist_ok=False)
    explainer = shap.TreeExplainer(
        model, data=background, model_output="probability", feature_perturbation="interventional"
    )
    explanation = explainer(x)
    values = np.asarray(explanation.values)
    scores = model.predict_proba(x)[:, 1]
    error = float(np.max(np.abs(explanation.base_values + values.sum(axis=1) - scores)))
    if error > 1e-4:
        raise ValueError(f"SHAP probability additivity failed: {error}")
    pd.DataFrame(values, index=x.index, columns=columns).to_csv(
        out / "values.csv", index_label="nodeid"
    )
    importance = pd.DataFrame({"feature": columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    importance.sort_values("mean_abs_shap", ascending=False).to_csv(
        out / "importance.csv", index=False
    )
    shap.summary_plot(values, x, show=False)
    plt.tight_layout()
    plt.savefig(out / "summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    shap.plots.waterfall(explanation[0], show=False)
    plt.savefig(out / "individual.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    # Raw inputs, base values, and scores allow exact reconstruction of each local explanation.
    x.to_csv(out / "sample_features.csv", index_label="nodeid")
    pd.DataFrame(
        {"nodeid": x.index, "base_value": explanation.base_values, "score": scores}
    ).to_csv(out / "local_predictions.csv", index=False)
    write_json(
        out / "explanation.json",
        {
            "sample_nodeids": x.index.tolist(),
            "background_nodeids": background.index.tolist(),
            "seed": seed,
            "output_space": "probability",
            "additivity_max_abs_error": error,
            "individual_nodeid": int(x.index[0]),
        },
    )
    result = faithfulness(lambda f: model.predict_proba(f)[:, 1], x, train, values, seed=seed)
    result["sample_nodeids"] = x.index.tolist()
    write_json(out / "faithfulness.json", result)
    seal(out, [p.name for p in out.iterdir() if p.is_file()])
    return out
