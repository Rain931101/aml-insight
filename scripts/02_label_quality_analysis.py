"""Week 1 – 步驟 6、7：欄位、標籤、資料品質、類別比例與洩漏分析。

回答的問題：
  Q1 標籤在哪個層級？（帳戶 / 交易 / 群組）
  Q2 正負樣本比例？與 metadata 是否一致？
  Q3 哪些欄位會洩漏標籤？（fraudStep、init_balance …）
  Q4 能否從帳戶標籤推導出「交易層級」標籤？推導的可信度如何？
  Q5 詐欺帳戶在圖上的群組結構（對應 200 個 pattern？）
  Q6 時間分布是否允許做時間切分（E3）？

用法：
    uv run python scripts/02_label_quality_analysis.py
輸出：
    reports/week1/02_label_quality.md
    reports/week1/02_summary.json
    reports/figures/week1_*.png
"""

from __future__ import annotations

import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_insight.data.loader import DATASETS, PROJECT_ROOT, load_dataset

REPORT_DIR = PROJECT_ROOT / "reports" / "week1"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def single_feature_scores(y: pd.Series, x: pd.Series) -> dict:
    """單一欄位對標籤的區辨力（診斷用，非模型結果）。

    AUC 取 max(auc, 1-auc)，代表「不論方向」的可分性。
    單一欄位 AUC 接近 1.0 = 強烈洩漏嫌疑；接近 0.5 = 無資訊。
    """
    auc = roc_auc_score(y, x)
    return {
        "roc_auc_any_direction": round(max(auc, 1 - auc), 4),
        "average_precision": round(average_precision_score(y, x if auc >= 0.5 else -x), 4),
        "positive_rate": round(float(y.mean()), 4),
    }


def node_activity(nodes: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """全期間帳戶活動統計（僅供 EDA；正式特徵需依切分時間點重算以避免洩漏）。"""
    out_ = tx.groupby("sourceNodeId").agg(out_cnt=("value", "size"), out_amt=("value", "sum"),
                                         out_deg=("targetNodeId", "nunique"))
    in_ = tx.groupby("targetNodeId").agg(in_cnt=("value", "size"), in_amt=("value", "sum"),
                                        in_deg=("sourceNodeId", "nunique"))
    df = nodes.set_index("nodeid").join(out_).join(in_).fillna(0)
    df["net_flow"] = df["in_amt"] - df["out_amt"]
    return df.reset_index()


def analyze(name: str) -> tuple[str, dict]:
    ds = load_dataset(name)
    nodes, tx = ds.nodes, ds.transactions
    buf = io.StringIO()
    w = lambda s="": buf.write(s + "\n")
    summ: dict = {"dataset": name}

    w(f"## {name}\n")

    # ---------- Q1/Q2 類別比例 ----------
    n_pos = int(nodes["isFraud"].sum())
    summ["n_nodes"], summ["n_fraud_nodes"] = len(nodes), n_pos
    summ["fraud_node_ratio"] = round(n_pos / len(nodes), 4)
    w("### Q1/Q2 標籤層級與類別比例\n")
    w("- `nodes.csv` 有 `isFraud` → **帳戶（節點）層級標籤**。")
    w("- `transactions.csv` **沒有任何標籤欄位**；也沒有 alert / pattern ID 欄位。")
    w(f"- 詐欺帳戶：{n_pos:,} / {len(nodes):,}（{n_pos / len(nodes):.2%}），"
      f"正負比約 1 : {(len(nodes) - n_pos) / n_pos:.1f}")
    w(f"- metadata 宣稱：`{ds.metadata.splitlines()[0]}` → "
      f"{'✅ 一致' if f'({n_pos} fraud' in ds.metadata else '⚠️ 與實際計數不一致，以 nodes.csv 為準'}\n")

    # ---------- Q3 洩漏檢查 ----------
    w("### Q3 洩漏檢查\n")
    ct = pd.crosstab(nodes["isFraud"], nodes["fraudStep"] == -1,
                     rownames=["isFraud"], colnames=["fraudStep == -1"])
    w("`fraudStep` 與標籤的交叉表：\n")
    w(ct.to_markdown())
    fs = nodes.loc[nodes["isFraud"] == 1, "fraudStep"]
    w(f"\n- 詐欺帳戶的 fraudStep 範圍：{fs.min()} ~ {fs.max()}"
      f"；詐欺帳戶中 fraudStep == -1 的比例 {(fs == -1).mean():.1%}")
    neg_all_minus1 = bool((nodes.loc[nodes.isFraud == 0, "fraudStep"] == -1).all())
    w(f"- 非詐欺帳戶 fraudStep 全為 -1：{'是' if neg_all_minus1 else '否'}"
      " → `fraudStep != -1` 時**必為詐欺**（單向洩漏，precision 100%）。")
    w("- **結論：`fraudStep` 不是完整標籤代理，但屬於標籤衍生欄位，必須排除於特徵之外**；"
      "可作為 EDA 中「詐欺行為發生時點」的參考。")
    summ["fraudStep_one_way_leak"] = neg_all_minus1
    summ["fraud_with_fraudStep_minus1_ratio"] = round(float((fs == -1).mean()), 4)
    fs = fs[fs != -1]  # 以下時間分析只用有效的 fraudStep

    act = node_activity(nodes, tx)
    diag = {}
    for col in ["init_balance", "in_cnt", "out_cnt", "in_amt", "out_amt", "in_deg", "out_deg",
                "net_flow"]:
        diag[col] = single_feature_scores(act["isFraud"], act[col])
    diag_df = pd.DataFrame(diag).T.sort_values("roc_auc_any_direction", ascending=False)
    w("\n單一欄位區辨力（全期間統計，**僅為診斷，不是模型結果**）：\n")
    w(diag_df.to_markdown())
    w("\n> 判讀：單一欄位 AUC ≥ 0.95 視為「可能是模擬器產生的捷徑訊號」，"
      "需在報告中討論；`init_balance` 為帳戶屬性，若區辨力異常高需評估是否保留。")
    summ["single_feature_auc"] = diag_df["roc_auc_any_direction"].to_dict()

    by_label = act.groupby("isFraud")[["init_balance", "in_cnt", "out_cnt", "in_amt",
                                       "out_amt"]].median()
    w("\n依標籤分組中位數：\n")
    w(by_label.round(2).to_markdown())

    # ---------- Q4 交易層級推導標籤 ----------
    w("\n### Q4 能否得到交易層級標籤？\n")
    lab = nodes.set_index("nodeid")
    tx = tx.assign(
        src_fraud=tx["sourceNodeId"].map(lab["isFraud"]),
        dst_fraud=tx["targetNodeId"].map(lab["isFraud"]),
        src_fstep=tx["sourceNodeId"].map(lab["fraudStep"]),
        dst_fstep=tx["targetNodeId"].map(lab["fraudStep"]),
    )
    both = (tx.src_fraud == 1) & (tx.dst_fraud == 1)
    either = (tx.src_fraud == 1) | (tx.dst_fraud == 1)
    w(f"- 兩端皆詐欺帳戶的交易：{both.sum():,}（{both.mean():.2%}）")
    w(f"- 至少一端為詐欺帳戶的交易：{either.sum():,}（{either.mean():.2%}）")
    w("- 兩端皆詐欺帳戶的交易中，發生時間與 fraudStep 的關係：")
    bt = tx[both]
    rel = pd.Series({
        "time == 兩端 fraudStep 較大者": (bt.time == np.maximum(bt.src_fstep, bt.dst_fstep)).mean(),
        "time >= 兩端 fraudStep 較大者": (bt.time >= np.maximum(bt.src_fstep, bt.dst_fstep)).mean(),
        "time <  兩端 fraudStep 較小者": (bt.time < np.minimum(bt.src_fstep, bt.dst_fstep)).mean(),
    })
    w(rel.map(lambda v: f"{v:.2%}").to_frame("比例").to_markdown())
    w("\n- 金額比較（兩端皆詐欺 vs 其他）：")
    w(tx.assign(group=np.where(both, "both_fraud", "other"))
      .groupby("group")["value"].describe().round(2).to_markdown())
    summ["tx_both_fraud"] = int(both.sum())
    summ["tx_either_fraud"] = int(either.sum())
    summ["tx_both_fraud_time_ge_fraudstep"] = round(float(rel.iloc[1]), 4)
    w("\n- **判讀：交易層級沒有真實標籤。**「兩端皆詐欺」只能當作弱標籤（weak label），"
      "因為詐欺帳戶之間也存在一般金額的交易（見 Q7）。")

    # ---------- Q7 金額捷徑（模擬器產物） ----------
    w("\n### Q7 金額捷徑檢查（模擬器產物）\n")
    other_min = tx.loc[~both, "value"].min()
    small = tx[tx.value < other_min]
    small_acc = set(small.sourceNodeId) | set(small.targetNodeId)
    flag = nodes.nodeid.isin(small_acc)
    y = nodes.isFraud == 1
    prec = (flag & y).sum() / max(flag.sum(), 1)
    rec = (flag & y).sum() / y.sum()
    w(f"- 非「兩端皆詐欺」交易的最小金額：{other_min:.2f}")
    w(f"- 金額 < {other_min:.2f} 的交易：{len(small):,} 筆，"
      f"其中兩端皆詐欺的比例 {(small.index.isin(tx.index[both])).mean():.1%}")
    w(f"- 規則「帳戶有任一筆金額 < {other_min:.0f} 的交易」→ 標記 {flag.sum():,} 帳戶，"
      f"precision {prec:.1%}、recall {rec:.1%}")
    no_small = nodes[y & ~flag]
    w(f"- 沒有小額交易的詐欺帳戶：{len(no_small):,}，其中 fraudStep == -1 的比例 "
      f"{(no_small.fraudStep == -1).mean():.1%}")
    w("- **判讀：AMLSim（此版本）的洗錢模式交易金額落在一般交易分布之外，"
      "形成可被單一門檻捕捉的捷徑。**模型效能必須同時在「含金額捷徑」與"
      "「控制金額捷徑」兩種設定下報告，否則 RQ1 的比較會被此人工訊號主導。")
    summ["amount_shortcut_threshold"] = round(float(other_min), 2)
    summ["amount_shortcut_rule_precision"] = round(float(prec), 4)
    summ["amount_shortcut_rule_recall"] = round(float(rec), 4)
    summ["fraud_without_small_tx"] = len(no_small)

    # ---------- Q5 詐欺子圖結構 ----------
    w("\n### Q5 詐欺帳戶的群組結構\n")
    g = nx.DiGraph()
    g.add_nodes_from(nodes.loc[nodes.isFraud == 1, "nodeid"])
    g.add_edges_from(bt[["sourceNodeId", "targetNodeId"]].itertuples(index=False))
    comps = [len(c) for c in nx.weakly_connected_components(g)]
    n_cycles_comp = sum(
        1 for c in nx.weakly_connected_components(g)
        if len(c) > 1 and not nx.is_directed_acyclic_graph(g.subgraph(c))
    )
    sizes = pd.Series(comps)
    w(f"- 只用「兩端皆詐欺」的交易建圖：{len(comps)} 個弱連通元件"
      f"（大小中位數 {sizes.median():.0f}，最大 {sizes.max()}，孤立詐欺帳戶 {(sizes == 1).sum()}）")
    w(f"- 含有向環的元件：{n_cycles_comp}")
    w(f"- metadata 宣稱 pattern 數：`{ds.metadata.splitlines()[-1]}`")
    w("- 意義：沒有 pattern/alert ID 欄位，只能用連通元件**近似**群組；"
      "群組層級標籤不可直接取得。")
    summ["fraud_wcc_count"] = len(comps)
    summ["fraud_wcc_with_cycle"] = n_cycles_comp

    # ---------- Q6 時間分布 ----------
    w("\n### Q6 時間分布（E3 時間切分可行性）\n")
    t_max = tx.time.max()
    cuts = {"前 60%": int(t_max * 0.6), "前 80%": int(t_max * 0.8)}
    for k, c in cuts.items():
        w(f"- {k}（time ≤ {c}）：交易 {(tx.time <= c).mean():.1%}；"
          f"fraudStep ≤ {c} 的詐欺帳戶 {(fs <= c).mean():.1%}")
    w("- 每個 step 的交易量：平均 "
      f"{tx.groupby('time').size().mean():,.0f}，標準差 {tx.groupby('time').size().std():,.0f}")
    summ["max_step"] = int(t_max)

    # ---------- 圖表 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tx.groupby("time").size().plot(ax=axes[0], title="Transactions per step")
    tx[both].groupby("time").size().reindex(range(1, t_max + 1), fill_value=0).plot(
        ax=axes[1], color="crimson", title="Fraud-to-fraud tx per step")
    fs.plot.hist(bins=30, ax=axes[2], color="crimson", title="fraudStep of fraud accounts")
    for ax in axes:
        ax.set_xlabel("time step")
    fig.suptitle(name)
    fig.tight_layout()
    fig_path = FIG_DIR / f"week1_{name}_time.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["init_balance", "in_cnt", "out_cnt"]):
        for lbl, color in [(0, "steelblue"), (1, "crimson")]:
            act.loc[act.isFraud == lbl, col].plot.hist(
                bins=40, alpha=0.6, density=True, ax=ax, color=color, label=f"isFraud={lbl}")
        ax.set_title(col)
        ax.legend()
    fig.suptitle(f"{name} – node feature distribution by label (full period, EDA only)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"week1_{name}_node_dist.png", dpi=120)
    plt.close(fig)
    w(f"\n圖：`reports/figures/week1_{name}_time.png`、`week1_{name}_node_dist.png`\n")

    return buf.getvalue(), summ


def cross_dataset_check() -> str:
    """三個資料集是否共用同一批正常帳戶？（決定能否當獨立資料集做跨資料比較）"""
    dsets = {n: load_dataset(n) for n in DATASETS}
    base = dsets[DATASETS[0]].nodes
    lines = ["## 跨資料集關係\n"]
    for n in DATASETS[1:]:
        other = dsets[n].nodes
        same_bal = (base.init_balance.values == other.init_balance.values).mean()
        lines.append(f"- `{DATASETS[0]}` vs `{n}`：init_balance 逐列相同比例 {same_bal:.1%}")
    ids_c = set(dsets["20K_cycle200"].nodes.query("isFraud==1").nodeid)
    ids_f = set(dsets["20K_fanin200"].nodes.query("isFraud==1").nodeid)
    ids_cf = set(dsets["20K_fanin200cycle200"].nodes.query("isFraud==1").nodeid)
    lines.append(f"- 詐欺帳戶 ID 交集：cycle∩fanin={len(ids_c & ids_f)}，"
                 f"cycle∩combined={len(ids_c & ids_cf)}，fanin∩combined={len(ids_f & ids_cf)}")
    lines.append("- 判讀：三者由同一模擬設定（同一 seed）產生，正常帳戶高度重疊，"
                 "**不是獨立樣本**；跨資料集比較只能作為「不同洗錢模式」的補充分析。\n")
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    parts = ["# Week 1 – 標籤、資料品質與洩漏分析\n",
             "自動產生：`scripts/02_label_quality_analysis.py`。"
             "所有數字皆由實際資料計算；單一欄位 AUC 為診斷用途，不代表模型效能。\n"]
    summaries = []
    for name in DATASETS:
        text, summ = analyze(name)
        parts.append(text)
        summaries.append(summ)
    parts.append(cross_dataset_check())
    report = "\n".join(parts)
    (REPORT_DIR / "02_label_quality.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "02_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
