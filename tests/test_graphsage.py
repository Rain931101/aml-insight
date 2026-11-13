"""成員 C：GraphSAGE 模組的單元測試（只用小型合成資料，不需 AMLSim）。"""

import numpy as np
import pandas as pd
import pytest
import torch

from aml_insight.models.graphsage import (
    build_edge_index,
    predict_proba,
    preprocess,
    train_and_select,
)


def test_preprocess_uses_train_statistics_only():
    features = pd.DataFrame({"a": [0.0, 1.0, 3.0, 1000.0]}, index=[10, 11, 12, 13])
    out = preprocess(features, train_ids=[10, 11, 12])
    # 改變非 train 帳戶的值，不應影響 train 帳戶標準化後的結果
    changed = features.copy()
    changed.loc[13, "a"] = -5.0
    out2 = preprocess(changed, train_ids=[10, 11, 12])
    np.testing.assert_allclose(out[:3], out2[:3])
    np.testing.assert_allclose(out[:3].mean(), 0.0, atol=1e-6)


def test_preprocess_constant_column_and_missing_ids():
    features = pd.DataFrame({"a": [2.0, 2.0, 2.0]}, index=[1, 2, 3])
    assert np.isfinite(preprocess(features, [1, 2])).all()
    with pytest.raises(ValueError):
        preprocess(features, [1, 99])


def test_build_edge_index_maps_ids_to_positions():
    edges = pd.DataFrame({"sourceNodeId": [100, 101], "targetNodeId": [101, 102]})
    ei = build_edge_index(edges, [100, 101, 102], direction="in")
    assert ei.tolist() == [[0, 1], [1, 2]]
    both = build_edge_index(edges, [100, 101, 102], direction="both")
    assert both.shape[1] == 4 and {(1, 0), (2, 1)} <= set(map(tuple, both.t().tolist()))
    with pytest.raises(ValueError):
        build_edge_index(edges, [100, 101], direction="in")
    with pytest.raises(ValueError):
        build_edge_index(edges, [100, 101, 102], direction="out")


def test_train_and_select_learns_neighbor_signal():
    # 兩群帳戶：同群內互連；標籤只能從「鄰居是誰」判斷，自身特徵是雜訊
    rng = np.random.default_rng(0)
    n = 120
    y = torch.tensor([1] * 30 + [0] * 90)
    src, dst = [], []
    for group in (range(30), range(30, n)):
        g = list(group)
        for i in g:
            for j in rng.choice(g, 4, replace=False):
                if i != j:
                    src.append(i)
                    dst.append(int(j))
    edge_index = build_edge_index(
        pd.DataFrame({"sourceNodeId": src, "targetNodeId": dst}), list(range(n)), "both"
    )
    marker = np.zeros((n, 1), dtype=np.float32)
    marker[rng.choice(30, 15, replace=False)] = 1.0  # 一半詐欺帳戶帶有可傳播的訊號
    x = torch.tensor(np.hstack([marker, rng.normal(size=(n, 3)).astype(np.float32)]))
    perm = rng.permutation(n)
    train, val = torch.tensor(perm[:80]), torch.tensor(perm[80:])
    best, records = train_and_select(
        x, edge_index, y, train, val, grid=[{"hidden": 8, "lr": 0.05}], epochs=60, seed=0
    )
    assert len(records) == 1 and 1 <= best["best_epoch"] <= 60
    assert best["validation_ap"] > 0.5  # 明顯高於隨機（正樣本比例 0.25）
    scores = predict_proba(best["model"], x, edge_index)
    assert scores.shape == (n,) and ((scores >= 0) & (scores <= 1)).all()


def test_train_requires_both_classes():
    x = torch.zeros((4, 2))
    ei = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        train_and_select(x, ei, torch.zeros(4, dtype=torch.long), torch.tensor([0, 1]),
                         torch.tensor([2, 3]), grid=[{"hidden": 4, "lr": 0.01}], epochs=1,
                         seed=0)
