import json

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def prepared(tmp_path):
    rng = np.random.default_rng(4)
    n = 500
    frame = pd.DataFrame(
        {
            "nodeid": np.arange(n),
            "isFraud": [0] * 450 + [1] * 50,
            "count": rng.normal(size=n),
            "amount": rng.uniform(size=n),
        }
    )
    frame.to_csv(tmp_path / "features.csv", index=False)
    cfg = {
        "features": "features.csv",
        "setting": "S2",
        "seed": 0,
        "experiment": "E1",
        "feature_spec": {"groups": {"behavior": ["count"], "amount": ["amount"]}},
        "rules": {
            "S1": [{"feature": "amount", "direction": "low"}],
            "S2": [{"feature": "count", "direction": "high"}],
        },
        "xgb_grid": {"n_estimators": [3, 5], "max_depth": [2]},
        "output_root": "experiments",
        "run_id": "unit",
        "purpose": "test",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_tune_finalize_and_lock(prepared):
    from aml_insight.evaluation.runner import finalize, tune

    run = tune(prepared)
    assert not (run / "metrics.json").exists()
    lock = json.loads((run / "selection.json").read_text())
    assert len(lock["candidates"]) == 2
    assert "test" not in lock["candidates"][0]
    finalize(run)
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics["models"]) == {"rule_based", "xgboost"}
    assert metrics["models"]["xgboost"]["n"] == 100
    assert (run / "pr_curve.png").exists()
    assert (run / "predictions.csv").exists()
    with pytest.raises(FileExistsError):
        finalize(run)
    with pytest.raises(FileExistsError):
        tune(prepared)


def test_integrity_and_external_alignment(prepared, tmp_path):
    from aml_insight.evaluation.inputs import align_predictions
    from aml_insight.evaluation.runner import finalize, tune

    p = tmp_path / "p.csv"
    pd.DataFrame({"nodeid": [3, 1], "score": [0.3, 0.1]}).to_csv(p, index=False)
    assert align_predictions(p, [1, 3]).tolist() == [0.1, 0.3]
    with pytest.raises(ValueError, match="exactly"):
        align_predictions(p, [1, 2])
    run = tune(prepared)
    f = run / "selection.json"
    f.write_text(f.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        finalize(run)


def test_time_window_requires_future_features(prepared):
    from aml_insight.evaluation.runner import tune

    cfg = yaml.safe_load(prepared.read_text())
    cfg["experiment"] = "E3"
    prepared.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="time"):
        tune(prepared)


def test_final_results_tampering_is_rejected(prepared, tmp_path):
    from aml_insight.evaluation.report import aggregate
    from aml_insight.evaluation.runner import finalize, tune

    run = tune(prepared)
    finalize(run)
    metrics = run / "metrics.json"
    metrics.write_text(metrics.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        aggregate(run.parent, tmp_path / "report", include_test=True)


def test_graphsage_rejects_nonablated_feature_contract(prepared):
    from aml_insight.evaluation.artifacts import sha256, write_json
    from aml_insight.evaluation.runner import prepare_manifest, tune

    cfg = yaml.safe_load(prepared.read_text())
    manifest = prepared.parent / "split.json"
    prepare_manifest(prepared, manifest)
    meta = prepared.parent / "c.json"
    write_json(
        meta,
        {
            "manifest_sha256": sha256(manifest),
            "seed": 0,
            "setting": "S2",
            "feature_file_sha256": sha256(prepared.parent / "features.csv"),
            "training_config": {"epochs": 1},
            "feature_version": "v1",
            "selected_columns": ["count", "amount"],
        },
    )
    cfg["graphsage"] = {"metadata": "c.json", "validation": "v.csv", "test": "t.csv"}
    prepared.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="selected"):
        tune(prepared)


def test_e3_uses_future_features_only_for_test(prepared):
    from aml_insight.evaluation.runner import finalize, tune

    cfg = yaml.safe_load(prepared.read_text())
    future = pd.read_csv(prepared.parent / "features.csv")
    future["count"] += 1000
    future.to_csv(prepared.parent / "future.csv", index=False)
    cfg.update(
        experiment="E3",
        time_test_features="future.csv",
        time={
            "train_start": 1,
            "train_end": 89,
            "test_start": 90,
            "test_end": 149,
            "feature_observation_attestation": "unit test only",
        },
    )
    prepared.write_text(yaml.safe_dump(cfg))
    run = tune(prepared)
    assert pd.read_parquet(run / "train.parquet")["count"].max() < 10
    assert pd.read_parquet(run / "validation.parquet")["count"].max() < 10
    assert pd.read_parquet(run / "test.parquet")["count"].min() > 900
    finalize(run)
    assert json.loads((run / "metrics.json").read_text())["experiment"] == "E3"


def test_external_graphsage_happy_path(prepared):
    from aml_insight.evaluation.artifacts import read_json, sha256, write_json
    from aml_insight.evaluation.runner import finalize, prepare_manifest, tune

    cfg = yaml.safe_load(prepared.read_text())
    manifest = prepared.parent / "split.json"
    prepare_manifest(prepared, manifest)
    splits = read_json(manifest)
    for part in ["validation", "test"]:
        pd.DataFrame(
            {"nodeid": splits[part], "score": np.linspace(0.1, 0.9, len(splits[part]))}
        ).to_csv(prepared.parent / f"{part}.csv", index=False)
    write_json(
        prepared.parent / "c.json",
        {
            "manifest_sha256": sha256(manifest),
            "feature_file_sha256": sha256(prepared.parent / "features.csv"),
            "seed": 0,
            "setting": "S2",
            "selected_columns": ["count"],
            "feature_version": "fixture-v1",
            "training_config": {"epochs": 3, "train_labels_only": True},
        },
    )
    cfg["graphsage"] = {"metadata": "c.json", "validation": "validation.csv", "test": "test.csv"}
    cfg["split_manifest"] = "split.json"
    prepared.write_text(yaml.safe_dump(cfg))
    run = tune(prepared)
    results = finalize(run)
    assert set(results) == {"xgboost", "rule_based", "graphsage"}
    assert not read_json(run / "metrics.json")["pending"]
    assert read_json(run / "error_analysis.json")["comparison_complete"]


def test_test_features_cannot_change_selected_models(prepared):
    from aml_insight.evaluation.artifacts import read_json, sha256
    from aml_insight.evaluation.runner import tune

    first = tune(prepared)
    ids = read_json(first / "manifest.json")["test"]
    frame = pd.read_csv(prepared.parent / "features.csv")
    frame.loc[frame.nodeid.isin(ids), ["count", "amount"]] = 999999.0
    frame.to_csv(prepared.parent / "changed.csv", index=False)
    cfg = yaml.safe_load(prepared.read_text())
    cfg.update(features="changed.csv", run_id="test-perturbed")
    prepared.write_text(yaml.safe_dump(cfg))
    second = tune(prepared)
    assert read_json(first / "selection.json") == read_json(second / "selection.json")
    assert sha256(first / "xgboost.ubj") == sha256(second / "xgboost.ubj")


def test_ablation_can_disable_rules(prepared):
    from aml_insight.evaluation.runner import finalize, tune

    cfg = yaml.safe_load(prepared.read_text())
    cfg.update(experiment="E2", setting="S1", drop_groups=["behavior"], include_rules=False)
    prepared.write_text(yaml.safe_dump(cfg))
    run = tune(prepared)
    result = finalize(run)
    assert set(result) == {"xgboost"}
    assert json.loads((run / "columns.json").read_text()) == ["amount"]


def test_e4_config_is_supported(prepared):
    from aml_insight.evaluation.runner import load_config

    cfg = yaml.safe_load(prepared.read_text())
    cfg["experiment"] = "E4"
    prepared.write_text(yaml.safe_dump(cfg))
    loaded, _ = load_config(prepared)
    assert loaded["experiment"] == "E4"
    cfg["experiment"] = "E6"
    prepared.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Experiment must"):
        load_config(prepared)
