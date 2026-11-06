# %% [markdown]
# # 成員 C – Week 1：GraphSAGE 入門（Cora 資料集）
#
# 目的：在套用到 AMLSim 之前，先用一個小而標準的資料集熟悉 PyG 的三個核心概念：
#   - `x`          每個節點的特徵表
#   - `edge_index` 誰連到誰
#   - `train_mask` 哪些節點的答案可以給模型學
#
# 並做一個「迷你版 RQ1」：只看節點自己（MLP）vs. 也看鄰居（GraphSAGE），結果差多少？
#
# 執行方式（擇一）：
#   uv run python notebooks/C_week1_graphsage_cora.py
#   或在 VS Code / Jupyter 以 "# %%" 分格逐段執行
#
# Cora：2,708 篇論文（節點）、論文引用關係（邊）、每篇論文屬於 7 個領域之一（標籤）。
# 對照本專題：論文 = 帳戶、引用 = 轉帳、領域 = 是否為詐欺帳戶。

# %%
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "external" / "pyg"
OUT = PROJECT_ROOT / "reports" / "week1" / "C_cora_results.json"
DEVICE = torch.device("cpu")  # 資料很小，CPU 就夠快

# %% [markdown]
# ## 1. 載入資料，看看 PyG 的 Data 物件長什麼樣子

# %%
dataset = Planetoid(root=str(DATA_DIR), name="Cora")
data = dataset[0].to(DEVICE)

print("=== Data 物件 ===")
print(data)
print(f"節點數 num_nodes          : {data.num_nodes}")
print(f"邊數   num_edges          : {data.num_edges}（無向圖，每條邊存兩次：A→B 與 B→A）")
print(f"節點特徵 x 的形狀         : {tuple(data.x.shape)}  → 每篇論文 {data.num_features} 維特徵")
print(f"edge_index 的形狀         : {tuple(data.edge_index.shape)}  → 第 0 列是起點、第 1 列是終點")
print(f"類別數                    : {dataset.num_classes}")
print(f"train / val / test 節點數 : {int(data.train_mask.sum())} / "
      f"{int(data.val_mask.sum())} / {int(data.test_mask.sum())}")

# %% [markdown]
# ## 2. 看一個節點的鄰居
# edge_index 就是一份「起點 → 終點」清單。找某個節點的鄰居 = 找所有起點是它的邊。

# %%
node = 0
neighbors = data.edge_index[1, data.edge_index[0] == node].tolist()
print(f"\n節點 {node} 的標籤：{int(data.y[node])}")
print(f"節點 {node} 的鄰居：{neighbors}")
print(f"鄰居的標籤：{data.y[neighbors].tolist()}")
print("→ 觀察：鄰居的標籤常常和自己一樣。這就是「看鄰居」可能有幫助的原因。")

# %% [markdown]
# ## 3. 兩個模型
# - MLP：只看節點自己的特徵（對應本專題的 XGBoost 思路：表格資料，不看關係）
# - GraphSAGE：每一層都把「自己」和「鄰居特徵的平均」合起來（aggr="mean"）
# 兩者層數、隱藏維度相同，唯一差別是有沒有用到 edge_index。


# %%
class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None):  # edge_index 收下但不用
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GraphSAGE(torch.nn.Module):
    """2 層 GraphSAGE：每個節點最多參考到 2 層之外（鄰居的鄰居）。"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, out_dim, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))   # 第 1 層：吸收直接鄰居
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)        # 第 2 層：再吸收鄰居的鄰居


# %% [markdown]
# ## 4. 訓練與評估
# - 全圖都送進模型（transductive），但 loss 只用 train_mask 的節點計算
# - 用 validation 準確率挑最好的那一輪（early stopping 的概念），最後才看 test
# - 不在 test 上挑模型，這是本專題的硬規則


# %%
def accuracy(logits, y, mask) -> float:
    return float((logits[mask].argmax(dim=1) == y[mask]).float().mean())


def train_one(model_cls, seed: int, epochs: int = 200, hidden: int = 64) -> dict:
    torch.manual_seed(seed)
    model = model_cls(dataset.num_features, hidden, dataset.num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_test, best_epoch = 0.0, 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        val_acc = accuracy(out, data.y, data.val_mask)
        if val_acc > best_val:  # 只依 validation 決定最佳輪次
            best_val, best_epoch = val_acc, epoch
            best_test = accuracy(out, data.y, data.test_mask)
    return {"seed": seed, "best_epoch": best_epoch, "val_acc": best_val, "test_acc": best_test}


# %%
if __name__ == "__main__" or True:
    seeds = [0, 1, 2, 3, 4]  # 和正式實驗相同的 seed 清單：不只報最好的一次
    results = {}
    for name, cls in [("MLP（只看自己）", MLP), ("GraphSAGE（也看鄰居）", GraphSAGE)]:
        runs = [train_one(cls, s) for s in seeds]
        accs = torch.tensor([r["test_acc"] for r in runs])
        results[name] = {
            "runs": runs,
            "test_acc_mean": round(float(accs.mean()), 4),
            "test_acc_std": round(float(accs.std()), 4),
        }
        print(f"\n{name}")
        for r in runs:
            print(f"  seed {r['seed']}: 第 {r['best_epoch']:3d} 輪最佳, "
                  f"val {r['val_acc']:.3f}, test {r['test_acc']:.3f}")
        print(f"  → test 準確率 {accs.mean():.3f} ± {accs.std():.3f}（5 個 seed）")

    print("\n模型結構：")
    print(GraphSAGE(dataset.num_features, 64, dataset.num_classes))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT.relative_to(PROJECT_ROOT)}")
