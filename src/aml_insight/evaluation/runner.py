"""Validation-only selection followed by an exclusive final test evaluation."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

from aml_insight.data.split import SEEDS, stratified_split
from aml_insight.evaluation.artifacts import (
    provenance,
    read_json,
    seal,
    sha256,
    verify,
    write_json,
)
from aml_insight.evaluation.inputs import align_predictions, load_features
from aml_insight.evaluation.metrics import evaluate, select_threshold
from aml_insight.models.rule_based import RuleBased
from aml_insight.models.xgb import fit_xgb


def resolve(base, path):
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("seed") not in SEEDS or cfg.get("setting") not in ("S1", "S2"):
        raise ValueError("Expected seed 0..4 and setting S1/S2")
    if cfg.get("experiment") not in ("E1", "E2", "E3", "E4", "E5"):
        raise ValueError("Experiment must be E1/E2/E3/E4/E5")
    if not cfg.get("run_id") or Path(cfg["run_id"]).name != cfg["run_id"]:
        raise ValueError("run_id must be a single directory name")
    return cfg, path.parent


def _time_validation(cfg):
    t = cfg.get("time", {})
    if not all(
        k in t
        for k in (
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "feature_observation_attestation",
        )
    ):
        raise ValueError("E3 requires time boundaries and upstream observation attestation")
    if not (t["train_start"] <= t["train_end"] < t["test_start"] <= t["test_end"]):
        raise ValueError("time windows must be ordered and non-overlapping")
    if not t["feature_observation_attestation"] or not cfg.get("time_test_features"):
        raise ValueError("E3 requires time_test_features and attestation")


def prepare_manifest(config_path, output):
    cfg, base = load_config(config_path)
    df, _ = load_features(
        resolve(base, cfg["features"]),
        cfg["feature_spec"],
        cfg["setting"],
        cfg.get("drop_groups", []),
    )
    split = stratified_split(df.index.to_numpy(), df.isFraud.to_numpy(), cfg["seed"])
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, split)
    return output


def tune(config_path):
    cfg, base = load_config(config_path)
    if cfg["experiment"] == "E3":
        _time_validation(cfg)
    if cfg["experiment"] == "E2" and not cfg.get("drop_groups"):
        raise ValueError("E2 requires explicit drop_groups")
    path = resolve(base, cfg["features"])
    df, columns = load_features(
        path, cfg["feature_spec"], cfg["setting"], cfg.get("drop_groups", [])
    )
    splits = stratified_split(df.index.to_numpy(), df.isFraud.to_numpy(), cfg["seed"])
    if cfg.get("split_manifest") and read_json(resolve(base, cfg["split_manifest"])) != splits:
        raise ValueError("Provided split manifest does not match account labels/seed")
    future = df
    if cfg["experiment"] == "E3":
        future, future_cols = load_features(
            resolve(base, cfg["time_test_features"]),
            cfg["feature_spec"],
            cfg["setting"],
            cfg.get("drop_groups", []),
        )
        if future_cols != columns or not future.index.equals(df.index):
            raise ValueError("time feature tables must have identical IDs and selected columns")
        if not np.array_equal(future.isFraud, df.isFraud):
            raise ValueError("Static labels must match across time windows")
    rules = cfg.get("rules", {}).get(cfg["setting"], [])
    use_rule = cfg.get("include_rules", True)
    if use_rule and (not rules or any(r["feature"] not in columns for r in rules)):
        raise ValueError("Provide nonempty rules using selected features; adjust for ablation")
    run = resolve(base, cfg.get("output_root", "../experiments")) / cfg["run_id"]
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", cfg)
    write_json(run / "manifest.json", splits)
    prov = provenance(base)
    prov["feature_file"] = {"path": str(path), "sha256": sha256(path)}
    if cfg["experiment"] == "E3":
        time_path = resolve(base, cfg["time_test_features"])
        prov["time_feature_file"] = {"path": str(time_path), "sha256": sha256(time_path)}
    write_json(run / "provenance.json", prov)
    write_json(run / "columns.json", columns)
    for part, ids in splits.items():
        source = future if part == "test" else df
        source.loc[ids, ["isFraud", *columns]].to_parquet(run / f"{part}.parquet")
    train = df.loc[splits["train"]]
    val = df.loc[splits["validation"]]
    if use_rule:
        rule = RuleBased(rules, cfg.get("rule_quantile", 0.95)).fit(val[columns])
        rule_scores = rule.predict_score(val[columns])
        rule_threshold = select_threshold(val.isFraud, rule_scores)
    grid = list(
        ParameterGrid(
            cfg.get(
                "xgb_grid",
                {
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 250],
                },
            )
        )
    )
    if not grid:
        raise ValueError("Empty XGBoost candidate grid")
    records, best_model, best_ap = [], None, -1.0
    for params in grid:
        model = fit_xgb(train[columns], train.isFraud, params, cfg["seed"])
        scores = model.predict_proba(val[columns])[:, 1]
        ap = float(average_precision_score(val.isFraud, scores))
        records.append({"params": params, "validation_pr_auc": ap})
        if ap > best_ap:  # First candidate wins exact ties, deterministic ParameterGrid order.
            best_model, best_ap, best_params, best_scores = model, ap, params, scores
    best_model.save_model(run / "xgboost.ubj")
    if use_rule:
        with (run / "rule.pkl").open("wb") as stream:
            pickle.dump(rule, stream)
    selection = {
        "criterion": "validation_average_precision; first candidate wins ties",
        "candidates": records,
        "best_params": best_params,
        "thresholds": {"xgboost": select_threshold(val.isFraud, best_scores)},
        "validation": {
            "xgboost": evaluate(
                val.isFraud, best_scores, select_threshold(val.isFraud, best_scores), val.index
            )
        },
        "pending": ["GraphSAGE predictions from member C"],
    }
    if use_rule:
        selection["thresholds"]["rule_based"] = rule_threshold
        selection["rule_thresholds"] = rule.thresholds
        selection["validation"]["rule_based"] = evaluate(
            val.isFraud, rule_scores, rule_threshold, val.index
        )
    external = cfg.get("graphsage")
    if external:
        meta_path = resolve(base, external["metadata"])
        meta = read_json(meta_path)
        if meta.get("selected_columns") != columns:
            raise ValueError("GraphSAGE selected columns must match the exact feature variant")
        if (
            meta.get("manifest_sha256") != sha256(run / "manifest.json")
            or meta.get("seed") != cfg["seed"]
            or meta.get("setting") != cfg["setting"]
            or meta.get("feature_file_sha256") != sha256(path)
            or not meta.get("training_config")
            or not meta.get("feature_version")
        ):
            raise ValueError("GraphSAGE provenance must match manifest, features, seed and setting")
        if (
            cfg["experiment"] == "E3"
            and meta.get("time_feature_file_sha256") != prov["time_feature_file"]["sha256"]
        ):
            raise ValueError("GraphSAGE time feature provenance mismatch")
        vs = align_predictions(resolve(base, external["validation"]), splits["validation"])
        ts = align_predictions(resolve(base, external["test"]), splits["test"])
        selection["thresholds"]["graphsage"] = select_threshold(val.isFraud, vs)
        selection["validation"]["graphsage"] = evaluate(
            val.isFraud, vs, selection["thresholds"]["graphsage"], val.index
        )
        pd.DataFrame({"nodeid": splits["test"], "score": ts}).to_csv(
            run / "graphsage_test.csv", index=False
        )
        pd.DataFrame({"nodeid": splits["validation"], "score": vs}).to_csv(
            run / "graphsage_validation.csv", index=False
        )
        write_json(run / "graphsage_metadata.json", meta)
        selection["pending"] = []
    metadata = cfg.get("analysis_metadata")
    if metadata:
        from aml_insight.evaluation.inputs import read_table, validate_ids

        md = read_table(resolve(base, metadata))
        validate_ids(md.nodeid)
        if set(md.nodeid) != set(df.index) or not md.has_small_amount.isin([0, 1]).all():
            raise ValueError("Analysis metadata requires all IDs and binary has_small_amount")
        md[["nodeid", "has_small_amount"]].to_csv(run / "analysis_metadata.csv", index=False)
    write_json(run / "selection.json", selection)
    seal(run, [p.name for p in run.iterdir() if p.is_file()])
    return run


def finalize(run):
    from xgboost import XGBClassifier

    from aml_insight.evaluation.report import plot_pr, write_error_analysis

    run = Path(run).resolve()
    if (run / "FINAL_EVALUATION_STARTED.json").exists():
        raise FileExistsError("Final evaluation already started; no automatic retries")
    verify(run)
    selection = read_json(run / "selection.json")
    columns = read_json(run / "columns.json")
    test = pd.read_parquet(run / "test.parquet")
    model = XGBClassifier()
    model.load_model(run / "xgboost.ubj")
    # Only load trusted local artifacts generated by this tool; never downloaded pickle files.
    scores = {"xgboost": model.predict_proba(test[columns])[:, 1]}
    if (run / "rule.pkl").exists():
        with (run / "rule.pkl").open("rb") as stream:
            rule = pickle.load(stream)
        scores["rule_based"] = rule.predict_score(test[columns])
    if (run / "graphsage_test.csv").exists():
        scores["graphsage"] = align_predictions(run / "graphsage_test.csv", test.index.tolist())
    # Exclusive creation guards concurrent invocations. Persist before reading test metrics.
    with (run / "FINAL_EVALUATION_STARTED.json").open("x") as stream:
        stream.write('{"state":"test evaluation started; retries prohibited"}\n')
    results = {
        name: evaluate(test.isFraud, s, selection["thresholds"][name], test.index)
        for name, s in scores.items()
    }
    cfg = read_json(run / "config.json")
    write_json(
        run / "metrics.json",
        {
            "experiment": cfg["experiment"],
            "seed": cfg["seed"],
            "setting": cfg["setting"],
            "purpose": cfg.get("purpose", "research"),
            "models": results,
            "pending": selection["pending"],
        },
    )
    predictions = pd.DataFrame({"nodeid": test.index, "isFraud": test.isFraud.to_numpy()})
    for name, s in scores.items():
        predictions[f"{name}_score"] = s
        predictions[f"{name}_pred"] = (s >= selection["thresholds"][name]).astype(int)
    predictions.to_csv(run / "predictions.csv", index=False)
    plot_pr(test.isFraud, scores, run / "pr_curve.png")
    write_error_analysis(run, predictions)
    write_json(run / "COMPLETE.json", {"state": "complete"})
    seal(
        run,
        [
            "metrics.json",
            "predictions.csv",
            "pr_curve.png",
            "common_false_negatives.csv",
            "error_analysis.json",
            "error_analysis.md",
            "FINAL_EVALUATION_STARTED.json",
            "COMPLETE.json",
        ],
        "result_integrity.json",
    )
    return results
