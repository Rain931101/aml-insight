import json

import numpy as np
import pandas as pd
import pytest


def test_aggregate_reports_pending(tmp_path):
    from aml_insight.evaluation.artifacts import seal, write_json
    from aml_insight.evaluation.report import aggregate

    for seed, ap in [(0, 0.4), (1, 0.6)]:
        run = tmp_path / f"r{seed}"
        run.mkdir()
        write_json(
            run / "config.json",
            {
                "seed": seed,
                "setting": "S2",
                "experiment": "E5",
                "purpose": "research",
                "feature_spec": {"groups": {}},
            },
        )
        write_json(
            run / "metrics.json",
            {
                "models": {"xgboost": {"pr_auc": ap, "f1": ap, "roc_auc": ap, "top_k": {}}},
                "seed": seed,
                "setting": "S2",
                "experiment": "E5",
                "purpose": "research",
            },
        )
        write_json(run / "COMPLETE.json", {"state": "complete"})
        seal(run, ["config.json"])
        seal(run, ["metrics.json", "COMPLETE.json"], "result_integrity.json")
    out = tmp_path / "report"
    table = aggregate(tmp_path, out)
    assert table.iloc[0]["pr_auc_mean"] == 0.5
    assert table.iloc[0]["pr_auc_std"] == pytest.approx(np.sqrt(0.02))
    assert table.iloc[0]["pr_auc_best"] == 0.6
    assert not bool(table.iloc[0]["five_seeds_complete"])
    text = (out / "results.md").read_text()
    assert "pending" in text and "r0/metrics.json" in text


def test_faithfulness_uses_training_reference():
    from aml_insight.explain.shap import faithfulness

    x = pd.DataFrame({"a": [4.0, 8.0], "b": [1.0, 2.0]})
    train = pd.DataFrame({"a": [0.0, 2.0], "b": [0.0, 0.0]})

    def predict(frame):
        return frame.a.to_numpy() / 10

    result = faithfulness(predict, x, train, np.array([[5.0, 1.0], [5.0, 1.0]]), [1], 0)
    assert result["top_k"][0]["mean_score_drop"] == pytest.approx(0.5)
    assert result["reference"] == "training_median"


def test_explain_saved_model(tmp_path):
    import yaml

    from aml_insight.evaluation.runner import finalize, tune
    from aml_insight.explain.shap import explain_run

    n = 500
    rng = np.random.default_rng(2)
    pd.DataFrame(
        {
            "nodeid": range(n),
            "isFraud": [0] * 450 + [1] * 50,
            "count": rng.normal(size=n),
            "amount": rng.random(n),
        }
    ).to_csv(tmp_path / "features.csv", index=False)
    cfg = {
        "features": "features.csv",
        "setting": "S2",
        "seed": 0,
        "experiment": "E1",
        "feature_spec": {"groups": {"behavior": ["count"], "amount": ["amount"]}},
        "rules": {"S2": [{"feature": "count", "direction": "high"}]},
        "xgb_grid": {"n_estimators": [3], "max_depth": [2]},
        "run_id": "explain",
        "output_root": "runs",
        "purpose": "test",
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = tune(path)
    finalize(run)
    explain_run(run, sample_size=8, background_size=20)
    assert (run / "shap/summary.png").exists()
    assert (run / "shap/individual.png").exists()
    m = json.loads((run / "shap/explanation.json").read_text())
    assert len(m["sample_nodeids"]) == 8
    assert set(m["background_nodeids"]).isdisjoint(m["sample_nodeids"])
    assert m["additivity_max_abs_error"] < 1e-4
    assert (run / "shap/faithfulness.json").exists()


def test_positive_count_is_semantic_across_seeds():
    from aml_insight.evaluation.metrics import evaluate
    from aml_insight.evaluation.report import _flatten

    a = _flatten(evaluate([1, 0, 0], [0.9, 0.2, 0.1], 0.5, [1, 2, 3]))
    b = _flatten(evaluate([1, 1, 0], [0.9, 0.8, 0.1], 0.5, [1, 2, 3]))
    assert a["recall_at_positive_count"] == b["recall_at_positive_count"] == 1


def test_graphsage_group_depends_on_c_training_config(tmp_path):
    from aml_insight.evaluation.artifacts import write_json
    from aml_insight.evaluation.report import _group_key

    cfg = {"experiment": "E1", "setting": "S2"}
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_json(
        a / "graphsage_metadata.json", {"training_config": {"epochs": 10}, "feature_version": "v1"}
    )
    write_json(
        b / "graphsage_metadata.json", {"training_config": {"epochs": 20}, "feature_version": "v1"}
    )
    assert _group_key(cfg, a, "graphsage") != _group_key(cfg, b, "graphsage")
    assert _group_key(cfg, a, "xgboost") == _group_key(cfg, b, "xgboost")
