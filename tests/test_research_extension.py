"""Regression checks for the new study's scientific boundaries, not model scores."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "research/2026-11-23"
sys.path.insert(0, str(HERE))
from core import INPUT, load, scale, verify_protocol


def test_train_only_scaling_and_late_distribution():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]])
    b = np.array([[9999.0, 9999.0], [3.0, 4.0], [2.0, 3.0]])
    _x, late, mean, std = scale(a, b, [0, 1], signed_log=False)
    np.testing.assert_array_equal(mean, [2, 3])
    np.testing.assert_array_equal(std, [1, 1])
    np.testing.assert_array_equal(late[0], [9997, 9996])


def test_protocol_and_message_degree_controls():
    verify_protocol()
    if not (INPUT / "full.npy").exists():
        pytest.skip("Build study inputs first")
    base = np.load(INPUT / "full.npy")
    assert not (base[0] == base[1]).any()
    for seed in range(5):
        rewired = np.load(INPUT / f"rewire-{seed}.npy")
        np.testing.assert_array_equal(
            np.bincount(base[1], minlength=20000), np.bincount(rewired[1], minlength=20000)
        )
        assert base.shape == rewired.shape
    assert np.load(INPUT / "no_edges.npy").shape == (2, 0)


def test_external_time_and_unknown_boundaries():
    import pandas as pd

    root = INPUT / "elliptic"
    if not root.exists():
        pytest.skip("External dataset unavailable")
    frame = pd.read_parquet(root / "features.parquet")
    splits = load(root / "split.json")
    windows = {"train": (1, 29), "validation": (30, 34), "test": (35, 49)}
    seen = set()
    for name, (lo, hi) in windows.items():
        rows = splits[name]
        assert not seen.intersection(rows)
        seen.update(rows)
        assert frame.iloc[rows].time.between(lo, hi).all()
        assert frame.iloc[rows].label.isin([0, 1]).all()
        edge = np.load(root / f"{name}.npy")
        assert (frame.time.to_numpy()[edge] <= hi).all()
    assert not seen.intersection(np.flatnonzero(frame.label.to_numpy() == -1))


def test_result_lock_does_not_recompute_or_overwrite(tmp_path):
    from core import finish, sha

    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.8, 0.3, 0.6])
    ids = np.array([1, 2, 3, 4])
    finish(tmp_path, ids, y, {"model": s}, {"model": 0.5}, {"purpose": "test"})
    before = sha(tmp_path / "metrics.json")
    with pytest.raises(FileExistsError):
        finish(tmp_path, ids, y, {"model": 1 - s}, {"model": 0.5}, {"purpose": "test"})
    assert sha(tmp_path / "metrics.json") == before


def test_explainer_preserves_eval_mode_and_target_logit():
    import torch
    from explain_gnn import CenteredModel
    from torch_geometric.explain import Explainer, GNNExplainer

    from aml_insight.models.graphsage import GraphSAGE

    torch.manual_seed(4)
    m = GraphSAGE(2, 4, 0.5)
    m.eval()
    wrapper = CenteredModel(m, 0.7)
    wrapper.eval()
    x = torch.randn(4, 2)
    e = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    with torch.no_grad():
        before = m(x, e).clone()
    ex = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=2),
        explanation_type="model",
        edge_mask_type="object",
        model_config={"mode": "binary_classification", "task_level": "node", "return_type": "raw"},
    )
    ex(x, e, index=0)
    assert not wrapper.training and not m.training
    with torch.no_grad():
        torch.testing.assert_close(m(x, e), before, rtol=0, atol=0)
