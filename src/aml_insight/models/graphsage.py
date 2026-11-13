"""GraphSAGE 帳戶分類（成員 C）。

設計重點（對應 docs/experiment_protocol.md）：
- transductive：全圖的邊與特徵都可見，但 loss 只用 train 帳戶的標籤。
- 特徵前處理（signed log1p + 標準化）的平均值與標準差只用 train 帳戶計算。
- 超參數與 epoch 只依 validation AP 選擇；test 分數只輸出，不在這裡計算任何指標。
- 類別不平衡：BCE 的 pos_weight = train 負樣本數 / train 正樣本數。
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected


def preprocess(features: pd.DataFrame, train_ids) -> np.ndarray:
    """signed log1p 壓縮長尾數值，再用 train 帳戶的平均與標準差做標準化。"""
    x = features.to_numpy(dtype=np.float64)
    x = np.sign(x) * np.log1p(np.abs(x))
    rows = features.index.get_indexer(train_ids)
    if (rows < 0).any():
        raise ValueError("train IDs missing from feature table")
    mean = x[rows].mean(axis=0)
    std = x[rows].std(axis=0)
    std[std == 0] = 1.0
    return ((x - mean) / std).astype(np.float32)


def build_edge_index(edges: pd.DataFrame, node_ids, direction: str = "both") -> torch.Tensor:
    """把 (sourceNodeId, targetNodeId) 轉成以列位置編號的 edge_index。

    direction="both"：同時加上反向邊，讓帳戶能同時參考轉入與轉出對象。
    direction="in"：只保留原方向（SAGEConv 只會從轉入方聚合訊息）。
    """
    if direction not in {"both", "in"}:
        raise ValueError("direction must be 'both' or 'in'")
    position = pd.Series(np.arange(len(node_ids)), index=pd.Index(node_ids))
    src = position.reindex(edges["sourceNodeId"]).to_numpy()
    dst = position.reindex(edges["targetNodeId"]).to_numpy()
    if np.isnan(src).any() or np.isnan(dst).any():
        raise ValueError("edge endpoint missing from feature table")
    edge_index = torch.tensor(np.vstack([src, dst]).astype(np.int64))
    if direction == "both":
        edge_index = to_undirected(edge_index, num_nodes=len(node_ids))
    return edge_index


class GraphSAGE(torch.nn.Module):
    """2 層 GraphSAGE（mean 聚合），輸出每個帳戶一個 logit。"""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, 1, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)


def _fit_one(x, edge_index, y, train_idx, val_idx, *, hidden, lr, weight_decay, dropout,
             epochs, seed):
    torch.manual_seed(seed)
    model = GraphSAGE(x.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_train = y[train_idx].float()
    pos = float(y_train.sum())
    if pos == 0 or pos == len(y_train):
        raise ValueError("train split needs both classes")
    pos_weight = torch.tensor((len(y_train) - pos) / pos)
    y_val = y[val_idx].numpy()

    best = {"val_ap": -1.0, "epoch": 0, "state": None}
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], y_train, pos_weight=pos_weight
        )
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(x, edge_index)[val_idx]).numpy()
        val_ap = float(average_precision_score(y_val, val_scores))
        curve.append({"epoch": epoch, "train_loss": float(loss.detach()), "val_ap": val_ap})
        if val_ap > best["val_ap"]:  # 同分保留較早的 epoch
            best = {"val_ap": val_ap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return model, best["val_ap"], best["epoch"], curve


def train_and_select(x, edge_index, y, train_idx, val_idx, *, grid: list[dict], epochs: int,
                     seed: int, weight_decay: float = 5e-4, dropout: float = 0.5):
    """逐一訓練候選超參數，只用 validation AP 挑出最佳者並回傳該模型。"""
    if not grid:
        raise ValueError("empty hyper-parameter grid")
    records, best = [], None
    for params in grid:
        model, val_ap, epoch, curve = _fit_one(
            x, edge_index, y, train_idx, val_idx, hidden=params["hidden"], lr=params["lr"],
            weight_decay=weight_decay, dropout=dropout, epochs=epochs, seed=seed,
        )
        records.append({"params": params, "best_epoch": epoch, "validation_ap": val_ap,
                        "curve": curve})
        if best is None or val_ap > best["validation_ap"]:  # 同分保留 grid 中較前者
            best = {"model": model, "params": params, "best_epoch": epoch, "validation_ap": val_ap}
    return best, records


@torch.no_grad()
def predict_proba(model, x, edge_index) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(x, edge_index)).numpy()
