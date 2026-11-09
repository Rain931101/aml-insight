import numpy as np
import pandas as pd
import pytest


def test_feature_contract(tmp_path):
    from aml_insight.evaluation.inputs import load_features

    p = tmp_path / "a.csv"
    pd.DataFrame(
        {"nodeid": [1, 2], "isFraud": [0, 1], "count": [4, 9], "amount": [10.0, 20.0]}
    ).to_csv(p, index=False)
    spec = {"groups": {"behavior": ["count"], "amount": ["amount"]}}
    frame, columns = load_features(p, spec, "S2")
    assert columns == ["count"] and list(frame.index) == [1, 2]
    with pytest.raises(ValueError, match="classified"):
        load_features(p, {"groups": {"behavior": ["count"]}}, "S1")
    pd.DataFrame({"nodeid": [1, 2], "isFraud": [0, 1], "fraudStep": [-1, 1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="forbidden"):
        load_features(p, {"groups": {"behavior": ["fraudStep"]}}, "S1")


def test_rules_fit_validation_quantile():
    from aml_insight.models.rule_based import RuleBased

    model = RuleBased([{"feature": "count", "direction": "high"}])
    val = pd.DataFrame({"count": np.arange(101)})
    model.fit(val)
    assert model.thresholds["count"] == 95
    assert model.predict_score(pd.DataFrame({"count": [0, 95, 100]})).tolist() == [0, 1, 1]
    low = RuleBased([{"feature": "amount", "direction": "low"}]).fit(
        pd.DataFrame({"amount": np.arange(101)})
    )
    assert low.thresholds["amount"] == 5
    assert low.predict_score(pd.DataFrame({"amount": [0, 5, 100]})).tolist() == [1, 1, 0]


def test_xgb_training_weight():
    from aml_insight.models.xgb import fit_xgb

    x = pd.DataFrame({"count": np.arange(20)})
    model = fit_xgb(x, [0] * 16 + [1] * 4, {"n_estimators": 3, "max_depth": 2}, 0)
    assert model.get_params()["scale_pos_weight"] == 4
    assert np.isfinite(model.predict_proba(x)).all()
    with pytest.raises(ValueError):
        fit_xgb(x, [0] * 20, {}, 0)
