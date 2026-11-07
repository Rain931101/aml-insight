# %% [markdown]
# # 成員 C – Week 1：畫出一個詐欺帳戶周圍的金流網絡
#
# 目的：親眼看看「詐欺帳戶的鄰居長什麼樣子」，為之後的 GraphSAGE 建立直覺。
#   - 以一個詐欺帳戶為中心，取出 1~2 跳（鄰居、鄰居的鄰居）的子圖
#   - 紅色 = 詐欺帳戶（isFraud=1）、灰色 = 正常帳戶、星形 = 中心帳戶
#   - 箭頭 = 轉帳方向（誰 → 誰），線越粗 = 這兩個帳戶之間的交易筆數越多
#
# 執行方式：
#   uv run python notebooks/C_week1_fraud_subgraph.py              # 自動挑一個帳戶
#   uv run python notebooks/C_week1_fraud_subgraph.py --node 123   # 指定帳戶
#
# 注意：這只是視覺化，不是模型結果；isFraud 在這裡只用來「上色」，不當特徵。

# %%
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # 不開視窗，直接存檔
import matplotlib.pyplot as plt
import networkx as nx

from aml_insight.data.loader import PROJECT_ROOT, load_dataset

DATASET = "20K_fanin200cycle200"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_JSON = PROJECT_ROOT / "reports" / "week1" / "C_fraud_subgraph.json"
MAX_NODES = 80  # 子圖超過這個點數就退回只畫 1 跳，避免整張圖糊成一團


# %% [markdown]
# ## 1. 把交易表變成有向圖
# 同一對帳戶可能轉很多次，這裡先合併成一條邊，記錄筆數與總金額。


# %%
def build_graph(nodes, tx) -> nx.DiGraph:
    pairs = (
        tx.groupby(["sourceNodeId", "targetNodeId"])
        .agg(n_tx=("value", "size"), total=("value", "sum"))
        .reset_index()
    )
    g = nx.DiGraph()
    g.add_nodes_from(
        (int(r.nodeid), {"isFraud": int(r.isFraud)}) for r in nodes.itertuples(index=False)
    )
    g.add_edges_from(
        (int(r.sourceNodeId), int(r.targetNodeId), {"n_tx": int(r.n_tx), "total": float(r.total)})
        for r in pairs.itertuples(index=False)
    )
    return g


# %% [markdown]
# ## 2. 取出中心帳戶的 k 跳子圖
# ego_graph(..., undirected=True)：不管轉入或轉出，有交易就算鄰居。


# %%
def ego(g: nx.DiGraph, center: int, radius: int) -> nx.DiGraph:
    return nx.ego_graph(g, center, radius=radius, undirected=True)


def pick_center(g: nx.DiGraph) -> int:
    """依 nodeid 由小到大，挑第一個「2 跳子圖不超過 MAX_NODES、且至少 5 個點」的詐欺帳戶。"""
    for n in sorted(n for n, d in g.nodes(data=True) if d["isFraud"] == 1):
        size = len(ego(g, n, 2))
        if 5 <= size <= MAX_NODES:
            return n
    raise RuntimeError("找不到大小合適的詐欺帳戶，請用 --node 指定")


# %% [markdown]
# ## 3. 畫圖並存檔


# %%
def draw(sub: nx.DiGraph, center: int, radius: int, path) -> None:
    pos = nx.spring_layout(sub, seed=0, k=0.6)
    colors = ["#d62728" if sub.nodes[n]["isFraud"] else "#bbbbbb" for n in sub]
    widths = [0.5 + min(d["n_tx"], 10) * 0.4 for _, _, d in sub.edges(data=True)]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(sub, pos, ax=ax, width=widths, alpha=0.5,
                           arrows=True, arrowsize=10, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=220,
                           edgecolors="black", linewidths=0.4)
    nx.draw_networkx_nodes(sub, pos, nodelist=[center], ax=ax, node_color="#d62728",
                           node_shape="*", node_size=900, edgecolors="black")
    nx.draw_networkx_labels(sub, pos, ax=ax, font_size=6)

    n_fraud = sum(sub.nodes[n]["isFraud"] for n in sub)
    ax.set_title(f"{DATASET} | center account {center} | {radius}-hop subgraph\n"
                 f"{sub.number_of_nodes()} accounts ({n_fraud} fraud = red), "
                 f"{sub.number_of_edges()} directed edges (width = #transactions)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# %%
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, default=None, help="中心帳戶 nodeid（需為詐欺帳戶）")
    args = parser.parse_args()

    ds = load_dataset(DATASET)
    g = build_graph(ds.nodes, ds.transactions)
    print(f"整張圖：{g.number_of_nodes():,} 個帳戶、{g.number_of_edges():,} 條（合併後）有向邊")

    center = args.node if args.node is not None else pick_center(g)
    if g.nodes[center]["isFraud"] != 1:
        print(f"[提醒] 帳戶 {center} 不是詐欺帳戶，照樣畫圖，但請留意。")

    radius = 2
    sub = ego(g, center, radius)
    if len(sub) > MAX_NODES:
        radius = 1
        sub = ego(g, center, radius)
        print(f"[提醒] 2 跳子圖太大，改畫 1 跳（{len(sub)} 個點）")

    # ---- 數字摘要：中心帳戶的直接鄰居有多少是詐欺帳戶？ ----
    nbrs = set(g.predecessors(center)) | set(g.successors(center))
    nbr_fraud = sum(g.nodes[n]["isFraud"] for n in nbrs)
    base_rate = sum(d["isFraud"] for _, d in g.nodes(data=True)) / g.number_of_nodes()
    summary = {
        "dataset": DATASET,
        "center": center,
        "radius": radius,
        "subgraph_nodes": sub.number_of_nodes(),
        "subgraph_edges": sub.number_of_edges(),
        "subgraph_fraud_nodes": sum(sub.nodes[n]["isFraud"] for n in sub),
        "center_in_degree": g.in_degree(center),
        "center_out_degree": g.out_degree(center),
        "direct_neighbors": len(nbrs),
        "direct_neighbors_fraud": nbr_fraud,
        "global_fraud_rate": round(base_rate, 4),
    }
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    if nbrs:
        print(f"→ 直接鄰居中詐欺比例 {nbr_fraud / len(nbrs):.1%}，"
              f"全體帳戶詐欺比例 {base_rate:.1%}（這只是一個帳戶的觀察，不能推論整體）")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"C_week1_subgraph_{center}.png"
    draw(sub, center, radius, fig_path)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {fig_path.relative_to(PROJECT_ROOT)}")
    print(f"[saved] {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
