"""Read-only view of the frozen follow-up study; no inference or fitting."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPORT = Path(__file__).resolve().parents[1] / "reports/research_extension_2026-11-23"


def render(heading, chart):
    heading(
        "06 / RESEARCH VALIDATION",
        "把模型的優勢，放進更嚴格的對照。",
        "2026-11-23 探索性追蹤研究 · 五個 seeds · 展示已封存結果，不重新訓練或選擇測試門檻。",
    )
    st.info(
        "此工作區使用固定研究資料，獨立於側欄 AMLSim 資料選擇。AMLSim test 已在先前研究被檢視；本輪屬探索性分析，不能視為全新盲測。"
    )
    try:
        receipt = json.loads((REPORT / "web_receipt.json").read_text())
        for name, expected in receipt.items():
            if hashlib.sha256((REPORT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"研究證據雜湊不符：{name}")
        rows = pd.read_csv(REPORT / "all_metrics.csv")
        if rows.duplicated(["dataset", "variant", "seed", "model"]).any():
            raise ValueError("研究結果包含重複索引")
        if not rows.pr_auc.between(0, 1).all():
            raise ValueError("研究指標超出合法範圍")
    except (OSError, ValueError) as exc:
        st.error(f"研究證據無法讀取：{exc}")
        return
    a, b, c = st.columns(3)
    a.metric("模型結果", len(rows))
    b.metric("資料來源", rows.dataset.nunique())
    c.metric("隨機種子", rows.seed.nunique())
    section = st.radio(
        "研究問題", ["圖與時間", "外部資料", "解釋可靠性", "完整報告"], horizontal=True
    )
    if section in ["圖與時間", "外部資料"]:
        external = section == "外部資料"
        subset = rows[rows.dataset.eq("Elliptic" if external else "AMLSim")]
        if not external:
            models = st.multiselect(
                "比較模型", sorted(subset.model.unique()), default=["graphsage", "xgboost"]
            )
            subset = subset[subset.model.isin(models)]
        if subset.empty:
            st.info("請至少選擇一個模型。")
            return
        metric = st.selectbox("評估指標", ["pr_auc", "f1", "roc_auc", "recall_5pct"])
        summary = (
            subset.groupby(["variant", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
        )
        figure = go.Figure()
        for model, data in summary.groupby("model"):
            figure.add_trace(
                go.Bar(
                    name=model,
                    x=data.variant,
                    y=data["mean"],
                    error_y={"type": "data", "array": data["std"]},
                    customdata=data["count"],
                    hovertemplate="%{x}<br>%{y:.4f}<br>seeds=%{customdata}<extra>%{fullData.name}</extra>",
                )
            )
        figure.update_layout(barmode="group", yaxis_title=metric, yaxis_range=[0, 1])
        chart(figure)
        st.caption(
            "誤差線為五個 seed 的樣本標準差；不是信賴區間或顯著性檢定。AP 使用 average precision。"
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)
        if external:
            st.warning(
                "Elliptic 是 Bitcoin 交易節點分類，與 AMLSim 銀行帳戶任務不同。模型在外部資料重新訓練，不能稱為同一模型直接跨域泛化。"
            )
            st.markdown(
                "Random Forest 與 XGBoost 在這份外部比較優於 GraphSAGE。僅採用 93 個本地特徵、保留時間切分，未知標籤不當作正常樣本。此處只發布彙總資料。"
            )
            st.link_button(
                "官方資料與授權", "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
            )
        else:
            st.markdown(
                "**full**：完整圖；**no_edges**：移除鄰居訊息；**rewire**：保持度數重接邊；**drop50**：移除一半無向邊；**no_structure**：移除圖衍生特徵；**early_late**：早期訓練、晚期輸入。"
            )
            st.warning(
                "時間視窗沿用相同帳戶與靜態標籤，不等於預測未來新犯罪。此輪 GraphSAGE 為 200 epochs，不能直接併入既有 500 epochs E1 排名。"
            )
        st.download_button(
            "下載目前比較資料", subset.to_csv(index=False), "research_comparison.csv", "text/csv"
        )
    elif section == "解釋可靠性":
        st.subheader("解釋能重現，是否也足夠穩定？")
        st.markdown(
            "GNNExplainer 使用 8 個代表案例 × 3 個解釋 seed；移除前 20% 關係對照隨機移除。平均額外信心下降約 0.112，但重要邊集合的平均 Jaccard 僅約 0.258，顯示解釋仍不穩定。"
        )
        st.image(str(REPORT / "figures/gnn_explanation_audit.png"))
        st.image(str(REPORT / "figures/shap_directional_audit.png"))
        st.caption(
            "包含沒有改善的案例；圖中的解釋不是犯罪因果證據。SHAP 與 GNN 的擾動單位不同，不把忠實度數字直接排名。"
        )
    else:
        report = (REPORT / "research_report.md").read_text()
        st.markdown(report)
        st.download_button("下載研究報告", report, "research_report.md", "text/markdown")
        st.download_button(
            "下載全部彙總指標", rows.to_csv(index=False), "all_metrics.csv", "text/csv"
        )
