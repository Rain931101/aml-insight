"""AML Insight 平台首頁（Week 1 骨架）。

執行：
    uv run streamlit run app/Home.py

目前只顯示「資料概況」，所有數字都即時由 data/raw/ 的原始資料計算，
不寫死任何數字。模型分數、解釋等頁面會在之後的週次加入 app/pages/。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from aml_insight.data.loader import DATASETS, RAW_DIR, load_dataset

st.set_page_config(page_title="AML Insight", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner="讀取資料中…")
def overview(name: str) -> dict:
    ds = load_dataset(name)
    nodes, tx = ds.nodes, ds.transactions
    per_step = tx.groupby("time").size().rename("交易筆數")
    return {
        "n_accounts": len(nodes),
        "n_fraud": int(nodes["isFraud"].sum()),
        "n_tx": len(tx),
        "t_min": int(tx["time"].min()),
        "t_max": int(tx["time"].max()),
        "per_step": per_step,
        "amount": tx["value"].describe(),
    }


st.title("🔎 AML Insight — 智慧金流風險分析平台")
st.caption(
    "研究用途。資料為 IBM AMLSim 合成資料；模型分數只代表模型在特定資料與設定下的預測，"
    "**不是犯罪證據**。"
)

available = [d for d in DATASETS if (RAW_DIR / d / "nodes.csv").exists()]
if not available:
    st.error("找不到資料。請先執行：`bash scripts/00_fetch_amlsim_samples.sh`")
    st.stop()

default = available.index("20K_fanin200cycle200") if "20K_fanin200cycle200" in available else 0
name = st.sidebar.selectbox("資料集", available, index=default)
info = overview(name)

st.subheader(f"資料概況：`{name}`")
c1, c2, c3, c4 = st.columns(4)
c1.metric("帳戶數", f"{info['n_accounts']:,}")
c2.metric("交易筆數", f"{info['n_tx']:,}")
c3.metric("詐欺帳戶數", f"{info['n_fraud']:,}")
c4.metric("詐欺帳戶比例", f"{info['n_fraud'] / info['n_accounts']:.2%}")

st.markdown(f"**時間範圍**：step {info['t_min']} ～ {info['t_max']}")
st.line_chart(info["per_step"], x_label="時間 step", y_label="交易筆數")

with st.expander("交易金額統計"):
    st.dataframe(pd.DataFrame(info["amount"]).T.round(2))

st.info(
    "標籤只存在於帳戶層級（isFraud）；`fraudStep` 為洩漏欄位，不作為模型特徵。"
    "詳見 docs/data_dictionary.md。"
)
