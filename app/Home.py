"""AML Insight research workbench. Read-only evidence, never evaluates a model again."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from research_data import (
    COLORS,
    MODELS,
    cohort,
    neighborhood,
    precision_recall,
    predictions,
    read_json,
    results,
    run_path,
    verify_bundle,
)

from aml_insight.data.loader import DATASETS, RAW_DIR, load_dataset

st.set_page_config(page_title="AML Insight | Research Lab", page_icon="◈", layout="wide")

THEMES = {
    "淺色": {
        "bg": "#f5f6f8",
        "surface": "#ffffff",
        "text": "#202334",
        "muted": "#626a7c",
        "border": "#dce0e8",
        "accent": "#6652cc",
    },
    "深色": {
        "bg": "#101116",
        "surface": "#191b24",
        "text": "#eeeff6",
        "muted": "#a6acc1",
        "border": "#303343",
        "accent": "#a99afa",
    },
}
with st.sidebar:
    st.markdown("### ◈ AML INSIGHT")
    st.caption("RESEARCH WORKBENCH / 研究工作台")
    pages = ["研究總覽", "資料鑑識", "模型實驗", "帳戶與網絡", "解釋證據", "研究驗證", "方法與來源"]
    initial = st.query_params.get("view", pages[0])
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = initial if initial in pages else pages[0]
    page = st.radio("工作區", pages, key="workspace")
    st.query_params["view"] = page
    st.divider()
    theme = st.selectbox("介面", list(THEMES))
    dataset = st.selectbox("資料集", DATASETS, index=2)
    st.caption("IBM AMLSim · 合成交易資料\n\n研究輔助，不作犯罪判定。")
T = THEMES[theme]
st.markdown(
    f"""<style>
:root {{color-scheme: {"dark" if theme == "深色" else "light"};}}
.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp label,
[data-testid="stMetricValue"] {{color:{T["text"]}!important;}}
[data-testid="stCaptionContainer"] p {{color:{T["muted"]}!important;}}
.stApp {{background:{T["bg"]};color:{T["text"]};}}
[data-testid="stSidebar"] {{background:{T["surface"]};border-right:1px solid {T["border"]};}}
[data-testid="stHeader"] {{background:transparent;}}
.block-container {{padding-top:2.7rem;max-width:1440px;padding-bottom:4rem;}}
h1 {{font-size:clamp(1.8rem,3vw,2.8rem)!important;letter-spacing:-.045em;font-weight:650!important;}}
h2 {{font-size:1.35rem!important;letter-spacing:-.025em;margin-top:1.2rem!important;}}
p,li {{line-height:1.7;}}
[data-testid="stMetric"] {{border-top:2px solid {T["border"]};padding:16px 0 12px;}}
[data-testid="stMetricValue"] {{font-variant-numeric:tabular-nums;letter-spacing:-.045em;}}
.eyebrow {{font:600 11px monospace;letter-spacing:.16em;color:{T["accent"]};margin-bottom:12px;}}
.deck {{max-width:850px;color:{T["muted"]};font-size:1rem;margin-bottom:28px;}}
.evidence {{border-left:3px solid {T["accent"]};padding:16px 20px;background:{T["surface"]};margin:16px 0 24px;}}
.evidence b {{color:{T["accent"]};}}
button:focus-visible,a:focus-visible,input:focus-visible {{outline:3px solid {T["accent"]}!important;outline-offset:3px;}}
button {{min-height:44px!important;}}
[data-testid="stRadio"] label {{min-height:44px;align-items:center;}}
@media(max-width:640px) {{.block-container {{padding:3rem 1rem 2rem;}} h1 {{font-size:1.8rem!important;}}}}
@media(prefers-reduced-motion:reduce) {{* {{animation:none!important;transition:none!important;scroll-behavior:auto!important;}}}}
</style>""",
    unsafe_allow_html=True,
)


def heading(kicker, title, description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="deck">{description}</div>', unsafe_allow_html=True)


def note(title, body):
    st.markdown(f'<div class="evidence"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)


def chart(fig, height=330):
    fig.update_layout(
        height=height,
        margin={"l": 20, "r": 20, "t": 35, "b": 30},
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["surface"],
        font={"color": T["text"], "family": "system-ui"},
        legend={"orientation": "h", "y": 1.15},
        hovermode="closest",
        colorway=list(COLORS.values()),
    )
    fig.update_xaxes(gridcolor=T["border"], zeroline=False)
    fig.update_yaxes(gridcolor=T["border"], zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


@st.cache_data(show_spinner="讀取原始交易與標籤…")
def data(name):
    ds = load_dataset(name)
    return ds.nodes, ds.transactions


@st.cache_data(show_spinner="核對研究證據完整性…")
def evidence():
    return verify_bundle(), results()


try:
    # Same fixed official sample source as the deployed main branch; optional for offline use.
    from aml_insight.data.fetch import ensure_samples

    if not (RAW_DIR / dataset / "transactions.csv").exists():
        with st.spinner("下載固定版本的官方合成資料…"):
            ensure_samples(datasets=(dataset,))
    nodes, tx = data(dataset)
    receipt, all_results = evidence()
    if dataset == receipt["dataset"]:
        import hashlib

        for filename, expected in receipt["raw_sha256"].items():
            if hashlib.sha256((RAW_DIR / dataset / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("原始資料與模型證據的版本不符，已停止展示以避免混用。")
except (OSError, ValueError, KeyError) as exc:
    st.error(f"無法載入或驗證資料：{exc}")
    st.info("請檢查網路或重建平台證據；不會以示範數字替代。")
    st.stop()

research_dataset = dataset == receipt["dataset"]
if page in ["模型實驗", "帳戶與網絡", "解釋證據"] and not research_dataset:
    heading(
        "DATA SCOPE",
        "這份資料尚無已封存的模型結果",
        "現有研究模型只對 fanin200cycle200 建立證據。切換至該資料集即可瀏覽；其他資料仍可用於資料鑑識。",
    )
    st.stop()

if page == "研究總覽":
    heading(
        "AML / GRAPH LEARNING / EXPLAINABILITY",
        "讓每一個風險判斷，都能回到證據。",
        "從合成交易、帳戶關係到模型誤判，逐層檢視圖學習是否帶來可重現的改善。",
    )
    a, b, c, d = st.columns(4)
    a.metric("研究帳戶", f"{len(nodes):,}")
    b.metric("交易紀錄", f"{len(tx):,}")
    c.metric("標籤正例率", f"{nodes.isFraud.mean():.2%}")
    d.metric("已封存實驗", str(len(receipt["sources"])), "S1 / S2 × 5 seeds", delta_color="off")
    note(
        "目前可以回答什麼？",
        "E1 / E5：比較三個模型的既有 test 結果與 seed 差異。"
        "模型證據僅來自 fanin200cycle200；所有數字皆可在「方法與來源」追溯。",
    )
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("模型在兩種設定下如何表現？")
        fig = go.Figure()
        for model, name in MODELS.items():
            group = all_results[all_results.model == model].groupby("setting").pr_auc
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=["S1", "S2"],
                    y=group.mean().reindex(["S1", "S2"]),
                    error_y={"type": "data", "array": group.std().reindex(["S1", "S2"])},
                    marker_color=COLORS[model],
                )
            )
        fig.update_layout(barmode="group", yaxis_title="Test average precision (AP)")
        chart(fig)
        st.caption("5 seeds 的平均 ± 樣本標準差；AP 非梯形積分。16 特徵版本，2026-11-16。")
    with right:
        st.subheader("研究證據地圖")
        st.markdown(
            "**01 · 模型比較 E1**  \n三模型、相同切分，10 組已封存。\n\n"
            "**02 · 消融 E2**  \n特徵與圖結構對照已納入研究驗證。\n\n"
            "**03 · 時間泛化 E3**  \n三模型早晚視窗結果已納入研究驗證。\n\n"
            "**04 · 解釋 E4**  \nSHAP、GNNExplainer 忠實度與穩定性均可查。\n\n"
            "**05 · 穩定性 E5**  \n重用 E1 的 5 seeds，避免重複 test。"
        )
    st.subheader("閱讀路徑")
    st.markdown(
        "先在 **資料鑑識** 檢查模擬器訊號，再到 **模型實驗** 比較 S1 / S2；"
        "最後用 **帳戶與網絡** 找出誤判，並在 **解釋證據** 檢視特徵貢獻。"
    )

elif page == "資料鑑識":
    heading(
        "01 / DATA FORENSICS",
        "先質疑資料，再相信模型。",
        "切換時間窗與交易金額，觀察資料分布與標籤關係。篩選只影響本頁探索，不會重算模型表現。",
    )
    window = st.slider(
        "交易時間窗（step，並非日曆日期）",
        int(tx.time.min()),
        int(tx.time.max()),
        (int(tx.time.min()), int(tx.time.max())),
    )
    view = cohort(nodes, tx, *window)
    a, b, c = st.columns(3)
    a.metric("視窗交易", f"{len(view):,}")
    b.metric("涉及帳戶", f"{len(set(view.sourceNodeId) | set(view.targetNodeId)):,}")
    c.metric("交易金額總和", f"{view.value.sum():,.2f}")
    l, r = st.columns(2)
    with l:
        st.subheader("交易活動節奏")
        series = view.groupby("time").size().reindex(range(window[0], window[1] + 1), fill_value=0)
        chart(
            go.Figure(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    fill="tozeroy",
                    line_color="#8477ef",
                )
            ).update_layout(xaxis_title="時間 step", yaxis_title="交易筆數")
        )
    with r:
        st.subheader("交易金額分布")
        chart(
            go.Figure(go.Histogram(x=view.value, nbinsx=45, marker_color="#2fa793")).update_layout(
                xaxis_title="模擬交易金額（原始單位）", yaxis_title="交易筆數"
            )
        )
    st.subheader("模擬器訊號檢查：小額交易是否暗示標籤？")
    cutoff = st.number_input(
        "探索用小額門檻（不作模型選參）", min_value=0.0, value=100.0, step=10.0
    )
    small = view[view.value < cutoff]
    both = int(((small.source_label == 1) & (small.target_label == 1)).sum())
    if len(small):
        note(
            f"{both:,} / {len(small):,} 筆小額交易的兩端皆有正例標籤",
            f"占比 {both / len(small):.2%}。這是既有帳戶標籤的關聯統計，不能視為交易犯罪標籤或因果證據。"
            "S2 移除金額與餘額特徵，正是為了檢查模型是否依賴這類模擬訊號。",
        )
    else:
        st.info("此條件沒有交易，請調整門檻或時間窗。")
    labels = view.groupby(["source_label", "target_label"]).size().reset_index(name="交易筆數")
    st.dataframe(
        labels.rename(columns={"source_label": "來源帳戶標籤", "target_label": "目標帳戶標籤"}),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("資料品質與原始交易"):
        st.write(
            {
                "帳戶 ID 唯一": bool(nodes.nodeid.is_unique),
                "帳戶標籤為二元": bool(nodes.isFraud.isin([0, 1]).all()),
                "交易端點皆存在": bool(
                    tx.sourceNodeId.isin(nodes.nodeid).all()
                    and tx.targetNodeId.isin(nodes.nodeid).all()
                ),
            }
        )
        st.dataframe(view.head(100), hide_index=True, use_container_width=True)
        st.caption("顯示前 100 筆；下方可匯出目前時間窗的全部交易。")
    st.download_button(
        "匯出時間窗交易 CSV", view.to_csv(index=False), "transactions.csv", "text/csv"
    )

elif page == "模型實驗":
    heading(
        "02 / MODEL BENCHMARK",
        "比較的不只是最高分。",
        "以相同帳戶切分比較 Rule-based、XGBoost 與 GraphSAGE；同時檢查不平衡資料、誤判與 seed 變異。",
    )
    note(
        "固定研究快照 · 2026-11-16",
        "S1 含金額；S2 排除金額與餘額。"
        "本頁重播既有 test 輸出，不重新評估，也不把瀏覽後的新設定視為獨立 test。",
    )
    setting = st.segmented_control("特徵設定", ["S1", "S2"], default="S2") or "S2"
    metric = st.selectbox("比較指標", ["pr_auc", "roc_auc", "f1", "precision_100", "recall_500"])
    selected = all_results[all_results.setting == setting]
    fig = go.Figure()
    for model, name in MODELS.items():
        row = selected[selected.model == model]
        fig.add_trace(
            go.Scatter(
                x=row.seed,
                y=row[metric],
                name=name,
                mode="lines+markers",
                line={"color": COLORS[model]},
                marker={"size": 9},
            )
        )
    fig.update_layout(xaxis={"title": "Seed", "dtick": 1}, yaxis_title=metric)
    chart(fig)
    table = selected.groupby("model")[metric].agg(["mean", "std", "min", "max", "count"])
    st.dataframe(table.rename(index=MODELS), use_container_width=True)
    st.caption("std 為 ddof=1 的樣本標準差；五次切分重用同一批帳戶，並非五份獨立外部資料。")
    seed = st.select_slider("檢視單次結果", options=list(range(5)), value=0)
    pred = predictions(setting, seed)
    l, r = st.columns(2)
    with l:
        st.subheader("Precision–Recall 曲線")
        fig = go.Figure()
        for model, name in MODELS.items():
            recall, precision = precision_recall(pred.isFraud, pred[f"{model}_score"])
            fig.add_trace(
                go.Scatter(x=recall, y=precision, name=name, mode="lines", line_color=COLORS[model])
            )
        fig.add_hline(y=pred.isFraud.mean(), line_dash="dot", line_color="#999999")
        chart(fig.update_layout(xaxis_title="Recall", yaxis_title="Precision"))
    with r:
        st.subheader("門檻下的錯誤成本")
        model = st.selectbox("混淆矩陣模型", list(MODELS), format_func=MODELS.get)
        metrics = read_json(run_path(setting, seed) / "metrics.json")["models"][model]
        chart(
            go.Figure(
                go.Heatmap(
                    z=metrics["confusion_matrix"],
                    x=["預測負例", "預測正例"],
                    y=["實際負例", "實際正例"],
                    text=metrics["confusion_matrix"],
                    texttemplate="%{text}",
                    colorscale="Purples",
                    showscale=False,
                )
            )
        )
        st.caption(f"門檻 {metrics['threshold']:.6f}，由 validation 選定；此處不可依 test 調整。")
    st.download_button("下載全部 seed 指標", all_results.to_csv(index=False), "model_metrics.csv")

elif page == "帳戶與網絡":
    heading(
        "03 / ACCOUNT INVESTIGATION",
        "從風險排序，走進交易關係。",
        "只展示指定 seed 的 held-out test 帳戶分數；未評分的鄰居不補造分數。網絡線條代表原始交易關係。",
    )
    a, b, c = st.columns(3)
    setting = a.selectbox("設定", ["S2", "S1"])
    seed = b.selectbox("Seed", list(range(5)))
    model = c.selectbox("排序模型", list(MODELS), format_func=MODELS.get)
    pred = predictions(setting, seed)
    status = st.radio(
        "篩選案例", ["全部", "假陽性 FP", "假陰性 FN", "三模型皆漏判"], horizontal=True
    )
    filtered = pred.copy()
    if status == "假陽性 FP":
        filtered = filtered[(filtered.isFraud == 0) & (filtered[f"{model}_pred"] == 1)]
    elif status == "假陰性 FN":
        filtered = filtered[(filtered.isFraud == 1) & (filtered[f"{model}_pred"] == 0)]
    elif status == "三模型皆漏判":
        filtered = filtered[
            (filtered.isFraud == 1) & (filtered[[f"{m}_pred" for m in MODELS]] == 0).all(axis=1)
        ]
    filtered = filtered.sort_values([f"{model}_score", "nodeid"], ascending=[False, True])
    st.caption(f"{len(filtered):,} 個符合條件的 test 帳戶；分數並非經校準的犯罪機率。")
    if filtered.empty:
        st.info("此設定沒有符合條件的案例，請切換篩選。")
        st.stop()
    node = st.selectbox("搜尋或選取帳戶 ID", filtered.nodeid.tolist())
    row = pred.set_index("nodeid").loc[node]
    cols = st.columns(3)
    for col, (m, name) in zip(cols, MODELS.items()):
        col.metric(
            name,
            f"{row[f'{m}_score']:.4f}",
            "門檻判定：" + ("正例" if row[f"{m}_pred"] else "負例"),
            delta_color="off",
        )
    st.caption(f"資料原有標籤：{int(row.isFraud)}。此標籤僅供事後驗證，不參與風險排序。")
    incident, edges = neighborhood(tx, node)
    l, r = st.columns([1.5, 1])
    with l:
        st.subheader("一階交易子圖")
        ids = sorted(set(edges.sourceNodeId) | set(edges.targetNodeId) | {node})
        neighbors = [i for i in ids if i != node]
        angles = np.linspace(0, 2 * np.pi, max(len(neighbors), 1), endpoint=False)
        positions = {i: (float(np.cos(a)), float(np.sin(a))) for i, a in zip(neighbors, angles)}
        positions[node] = (0.0, 0.0)
        fig = go.Figure()
        ex, ey = [], []
        for edge in edges.itertuples():
            x0, y0 = positions[edge.sourceNodeId]
            x1, y1 = positions[edge.targetNodeId]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line={"color": "#9297ac", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[positions[i][0] for i in ids],
                y=[positions[i][1] for i in ids],
                mode="markers",
                text=[f"帳戶 {i}" for i in ids],
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "size": [24 if i == node else 11 for i in ids],
                    "color": [T["accent"] if i == node else "#8097a7" for i in ids],
                },
                showlegend=False,
            )
        )
        chart(fig.update_layout(xaxis_visible=False, yaxis_visible=False), 390)
        st.caption(
            "紫色為查詢帳戶；灰色為鄰居（不代表正常）。最多 40 個有向帳戶對，"
            "依交易次數排序；圖中省略方向，以下表格保留來源與目標。這不是 GNNExplainer。"
        )
    with r:
        st.subheader("交易軌跡")
        st.metric("相關交易", f"{len(incident):,}")
        trend = incident.groupby("time").value.sum()
        chart(
            go.Figure(
                go.Scatter(x=trend.index, y=trend.values, mode="lines", line_color="#2fa793")
            ).update_layout(xaxis_title="時間 step", yaxis_title="進出交易總金額"),
            260,
        )
    st.dataframe(
        edges.rename(
            columns={
                "sourceNodeId": "來源",
                "targetNodeId": "目標",
                "count": "交易次數",
                "amount": "交易金額總和",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("風險清單與案例匯出"):
        st.dataframe(filtered.head(100), hide_index=True, use_container_width=True)
        st.download_button("下載篩選案例", filtered.to_csv(index=False), "account_cases.csv")

elif page == "解釋證據":
    heading(
        "04 / EXPLANATION AUDIT",
        "模型為什麼給出這個分數？",
        "以封存的 Tree SHAP 檢查 XGBoost 的全域與個案特徵貢獻；解釋是模型行為描述，並非因果證據。",
    )
    setting = st.selectbox("SHAP 設定", ["S2", "S1"])
    path = run_path(setting, 0) / "shap"
    meta = read_json(path / "explanation.json")
    st.caption(
        f"Seed 0 · {len(meta['sample_nodeids'])} 個 test 抽樣帳戶 · "
        f"{len(meta['background_nodeids'])} 個 train 背景帳戶 · probability 輸出空間"
    )
    importance = pd.read_csv(path / "importance.csv")
    values = pd.read_csv(path / "values.csv").set_index("nodeid")
    local = pd.read_csv(path / "local_predictions.csv").set_index("nodeid")
    l, r = st.columns(2)
    with l:
        st.subheader("全域特徵貢獻")
        mean_abs = values.abs().mean().sort_values().tail(12)
        chart(
            go.Figure(
                go.Bar(x=mean_abs.values, y=mean_abs.index, orientation="h", marker_color="#8477ef")
            ).update_layout(xaxis_title="平均 |SHAP|（200 個抽樣帳戶）"),
            430,
        )
    with r:
        st.subheader("個案解釋")
        node = st.selectbox("有 SHAP 證據的帳戶", values.index.tolist())
        contributions = values.loc[node].sort_values(key=abs).tail(10)
        chart(
            go.Figure(
                go.Bar(
                    x=contributions.values,
                    y=contributions.index,
                    orientation="h",
                    marker_color=["#c8715c" if v > 0 else "#2fa793" for v in contributions.values],
                )
            ).update_layout(xaxis_title="SHAP：正值推高分數 / 負值降低分數"),
            375,
        )
    base, score = local.loc[node, ["base_value", "score"]]
    st.code(
        f"背景基準 {base:.6f} + 全部特徵貢獻 {values.loc[node].sum():+.6f}"
        f" = {base + values.loc[node].sum():.6f}\n封存模型分數 = {score:.6f}"
    )
    st.caption(
        f"最大加總誤差：{meta['additivity_max_abs_error']:.2e}；上圖只列十項，加總檢查用全部特徵。"
    )
    faith = read_json(path / "faithfulness.json")
    st.subheader("忠實度檢驗：移除重要特徵後如何變化？")
    ftable = pd.DataFrame(
        [{"移除特徵數": x["k"], "平均分數下降": x["mean_score_drop"]} for x in faith["top_k"]]
    )
    st.dataframe(ftable, hide_index=True, use_container_width=True)
    st.info(
        "以 train 中位數遮罩；平均下降若為負，表示遮罩後分數反而上升。"
        "不能據此宣稱解釋已通過忠實度驗證，仍需依預測類別、貢獻方向與隨機對照分析。"
    )
    note(
        "GraphSAGE + GNNExplainer · 研究驗證已接入",
        "目前的交易子圖只是關係展示，不是 GNN 解釋。此頁保留原 E1 SHAP；新 GNNExplainer 忠實度與穩定性請至「研究驗證」。",
    )
    st.download_button("下載 SHAP 特徵貢獻", values.to_csv(), f"shap_{setting}_seed0.csv")

elif page == "研究驗證":
    from research_validation import render

    render(heading, chart)

else:
    heading(
        "05 / METHODS & PROVENANCE",
        "讓結果可查核，也讓限制可見。",
        "每個模型數字對應一份封存結果，展示層不載入模型權重、不選參，也不重新使用 test 調整門檻。",
    )
    st.subheader("實驗約定")
    st.markdown(
        "1. 以帳戶為分類單位，標籤為 `isFraud`；排除 `fraudStep`。\n"
        "2. 分層隨機切分 train / validation / test = 60 / 20 / 20，seeds 0–4。\n"
        "3. XGBoost 選參與分類門檻僅用 validation。Rule-based 分位數也來自 validation，"
        "因此 validation 分數可能偏樂觀。\n"
        "4. GraphSAGE 為 transductive：可見未標記節點與圖關係，loss 只使用 train 標籤；"
        "不等於未見帳戶的 inductive 泛化。\n"
        "5. test 結果一次封存；本平台只讀取已有結果。AP 為 average precision，"
        "五 seeds 報告平均與樣本標準差。"
    )
    note(
        "版本邊界",
        "模型證據為 2026-11-16 的 16 特徵版本，早於 min_amount 加入。"
        "資料鑑識使用原始交易，不能據此宣稱模型已使用新特徵。後續修改需獨立版本與揭露既有 test 已被檢視。",
    )
    st.subheader("資料來源與完整性")
    raw_match = {}
    import hashlib

    for name, expected in receipt["raw_sha256"].items():
        path = RAW_DIR / receipt["dataset"] / name
        raw_match[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() == expected
            if path.exists()
            else "尚未下載"
        )
    st.write({"模型證據封裝": "SHA-256 檢查通過", "模型對應原始資料": raw_match})
    st.dataframe(pd.DataFrame(receipt["sources"]), hide_index=True, use_container_width=True)
    st.download_button(
        "下載來源與檔案雜湊",
        __import__("json").dumps(receipt, indent=2),
        "research_receipt.json",
        "application/json",
    )
    st.link_button(
        "檢視研究方法文件",
        "https://github.com/Rain931101/aml-insight/blob/main/docs/experiment_protocol.md",
    )
    st.subheader("尚未能回答的問題")
    st.markdown(
        "- 合成資料上的表現不能外推至真實銀行。\n"
        "- 帳戶標籤並不提供逐筆交易犯罪標籤。\n"
        "- 圖結構消融與特徵消融已列於研究驗證；其差異不能直接視為因果貢獻。\n"
        "- 時間視窗、GNNExplainer、外部資料與校準診斷已完成探索性分析；尚無真人使用驗證與真實銀行驗證。\n"
        "- 五個 seed 的差異不是統計顯著性檢定；此處不宣稱顯著改善。"
    )

st.divider()
st.caption(
    "AML INSIGHT / Research prototype · IBM AMLSim synthetic data · 資料標籤 ≠ 模型預測 ≠ 犯罪證據"
)
