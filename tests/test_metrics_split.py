import numpy as np
import pytest


def test_hand_calculated_metrics():
    from aml_insight.evaluation.metrics import evaluate

    m = evaluate([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 0.75, [4, 3, 2, 1], ks=[2, 100])
    assert m["pr_auc"] == pytest.approx(5 / 6)
    assert m["roc_auc"] == 0.75
    assert m["f1"] == 0.5
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["top_k"]["2"]["precision"] == 0.5
    assert m["top_k"]["2"]["recall"] == 0.5
    assert m["top_k"]["100"]["effective_k"] == 4


def test_ties_and_single_class():
    from aml_insight.evaluation.metrics import evaluate

    m = evaluate([1, 0], [0.5, 0.5], 0.5, [20, 10], ks=[1])
    assert m["top_k"]["1"]["precision"] == 0
    m = evaluate([0, 0], [0.2, 0.4], 0.5, [1, 2])
    assert m["pr_auc"] is None and m["roc_auc"] is None
    assert m["top_k"]["100"]["recall"] is None
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.1, np.nan], 0.5, [1, 2])


def test_threshold_validation_only():
    from aml_insight.evaluation.metrics import select_threshold

    assert select_threshold([0, 1, 1], [0.1, 0.7, 0.8]) == 0.7


@pytest.mark.parametrize("seed", range(5))
def test_stratified_split(seed):
    from aml_insight.data.split import stratified_split

    ids = np.arange(2000)
    y = np.array([0] * 1800 + [1] * 200)
    s = stratified_split(ids, y, seed)
    assert [len(s[k]) for k in ("train", "validation", "test")] == [1200, 400, 400]
    assert set(s["train"]).isdisjoint(s["test"])
    assert set(s["train"]).isdisjoint(s["validation"])
    assert set(s["validation"]).isdisjoint(s["test"])
    assert {i for part in s.values() for i in part} == set(ids)
    assert all(abs(y[v].mean() - 0.1) < 0.005 for v in s.values())
    assert s == stratified_split(ids, y, seed)


def test_split_rejects_small_or_duplicate_input():
    from aml_insight.data.split import stratified_split

    with pytest.raises(ValueError):
        stratified_split([1, 1, 2, 3], [0, 0, 1, 1], 0)
    with pytest.raises(ValueError):
        stratified_split(range(10), [0] * 9 + [1], 0)
