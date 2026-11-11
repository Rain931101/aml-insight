"""Generate six data-derived Week 1 EDA plots and their written interpretations."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aml_insight.data.clean import clean_dataset
from aml_insight.data.loader import PROJECT_ROOT, load_dataset
from aml_insight.features.account import build_account_features


def main() -> None:
    ds = load_dataset("20K_fanin200cycle200")
    nodes, tx, quality = clean_dataset(ds)
    out = PROJECT_ROOT / "reports" / "week1"
    figs = PROJECT_ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.2})
    activity = build_account_features(nodes, tx, include_labels=True)
    tx_per_step = tx.groupby("time").size().reindex(range(tx.time.min(), tx.time.max() + 1), fill_value=0)
    labels = nodes.set_index("nodeid").isFraud
    both_fraud = tx.sourceNodeId.map(labels).eq(1) & tx.targetNodeId.map(labels).eq(1)
    amount_threshold = float(tx.loc[~both_fraud, "value"].min())

    plots: list[tuple[str, str]] = []
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(tx.value, bins=60, color="#4c78a8"); ax.set(title="Transaction amount distribution", xlabel="Amount", ylabel="Transactions")
    path = figs / "A_eda_amount_distribution.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); plots.append((path.name, "金額呈長尾分布；小額交易區域另需與帳戶標籤關係對照，不能單憑此分布推論犯罪。"))

    fig, ax = plt.subplots(figsize=(8, 4)); tx_per_step.plot(ax=ax, color="#f58518"); ax.set(title="Transactions per simulation step", xlabel="Step", ylabel="Transactions")
    path = figs / "A_eda_transactions_per_step.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); plots.append((path.name, "各 step 的交易量用於檢查時間覆蓋與時間窗；step 是模擬步數，不代表實際日曆時間。"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    upper = max(20, int(max(activity.in_degree.quantile(.99), activity.out_degree.quantile(.99))))
    for label, color in [(0, "#4c78a8"), (1, "#e45756")]:
        subset = activity.loc[activity.isFraud == label]
        axes[0].hist(subset.in_degree, bins=np.arange(0, upper + 2), density=True, alpha=.55, color=color, label=f"isFraud={label}")
        axes[1].hist(subset.out_degree, bins=np.arange(0, upper + 2), density=True, alpha=.55, color=color, label=f"isFraud={label}")
    axes[0].set(title="In-degree by label", xlabel="Distinct senders", ylabel="Density"); axes[1].set(title="Out-degree by label", xlabel="Distinct recipients", ylabel="Density")
    for ax in axes: ax.legend()
    path = figs / "A_eda_degree_by_label.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); plots.append((path.name, "依標籤分組比較入度與出度；這些是全期間 EDA 統計，正式預測須依各資料切分重算。"))

    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(np.log1p(activity.in_tx_count), bins=45, alpha=.6, label="incoming"); ax.hist(np.log1p(activity.out_tx_count), bins=45, alpha=.6, label="outgoing"); ax.set(title="Account event counts (log1p)", xlabel="log(1 + transaction count)", ylabel="Accounts"); ax.legend()
    path = figs / "A_eda_event_counts.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); plots.append((path.name, "帳戶事件量偏斜明顯，因此報圖採 log1p；帳戶事件數和不同鄰居數是不同概念。"))

    fig, ax = plt.subplots(figsize=(7, 4)); grouped = tx.assign(pair_label=np.where(both_fraud, "both endpoints fraud", "other")); grouped.boxplot(column="value", by="pair_label", ax=ax, showfliers=False); ax.set(title="Amounts by endpoint-label group", xlabel="Endpoint group", ylabel="Amount"); fig.suptitle("")
    path = figs / "A_eda_amount_by_endpoint_labels.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); plots.append((path.name, "依兩端帳戶標籤檢視交易金額，揭示合成資料可能存在的金額捷徑；這不是交易真實標籤。"))

    fs = nodes.loc[nodes.isFraud.eq(1) & nodes.fraudStep.ge(0), "fraudStep"]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(fs, bins=30, color="#e45756"); ax.set(title="fraudStep among labeled fraud accounts", xlabel="fraudStep", ylabel="Accounts")
    path = figs / "A_eda_fraudstep.png"; fig.tight_layout(); fig.savefig(path); plt.close(fig); plots.append((path.name, "fraudStep 只供標籤品質與時間 EDA 使用；它會直接透露標籤，禁止作為模型輸入。"))

    summary = {"dataset": ds.name, "accounts": len(nodes), "fraud_accounts": int(nodes.isFraud.sum()),
               "fraud_rate": float(nodes.isFraud.mean()), "transactions": len(tx),
               "duplicate_rows": quality.exact_duplicate_rows, "self_loop_rows": quality.self_loop_rows,
               "isolated_accounts": quality.accounts_without_transactions,
               "small_amount_threshold": amount_threshold, "eda_plots": [n for n, _ in plots]}
    (out / "A_eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    small_accounts = set(tx.loc[tx.value < amount_threshold, "sourceNodeId"]) | set(tx.loc[tx.value < amount_threshold, "targetNodeId"])
    hard = nodes.loc[nodes.isFraud.eq(1) & ~nodes.nodeid.isin(small_accounts)]
    fraud = nodes.loc[nodes.isFraud.eq(1)]
    hard_report = ["# 詐欺帳戶難辨識子集分析", "",
                   f"以非兩端皆詐欺交易的最小金額 {amount_threshold:.2f} 作為小額捷徑界線。",
                   f"有小額交易的詐欺帳戶 {len(fraud) - len(hard):,}；無小額交易的詐欺帳戶 {len(hard):,} / {len(fraud):,}（{len(hard) / len(fraud):.1%}）。",
                   f"該子集 `fraudStep == -1`：{(hard.fraudStep == -1).mean():.1%}；全部詐欺帳戶 `fraudStep == -1`：{(fraud.fraudStep == -1).mean():.1%}。",
                   "", "此集合只表示沒有小額交易捷徑，不表示帳戶一定無法偵測。模型錯誤分析應分開呈現此子集 recall，並避免以此資料切出新的標籤或交易層級標籤。", ""]
    (PROJECT_ROOT / "reports" / "week2" / "hard_to_detect_accounts.md").write_text("\n".join(hard_report), encoding="utf-8")
    lines = ["# Member A — Week 1 EDA", "", "主要資料集 `20K_fanin200cycle200`。下方圖表與摘要由原始 CSV 即時計算；全期間圖只用於描述，不代表時間切分模型輸入。", "",
             f"- 帳戶 {len(nodes):,}；詐欺帳戶 {int(nodes.isFraud.sum()):,}（{nodes.isFraud.mean():.2%}）",
             f"- 交易 {len(tx):,}；完全重複列 {quality.exact_duplicate_rows:,}；self-loop {quality.self_loop_rows:,}；無交易帳戶 {quality.accounts_without_transactions}", ""]
    for i, (name, interpretation) in enumerate(plots, 1):
        lines += [f"## {i}. {name}", "", f"![{name}](../figures/{name})", "", interpretation, ""]
    (out / "A_week1_eda.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
