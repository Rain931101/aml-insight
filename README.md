# AML Insight — 智慧金流風險分析平台

**結合圖神經網路與可解釋人工智慧之異常金流偵測與潛在洗錢網絡分析系統**
*An Explainable Graph Neural Network System for Suspicious Transaction Detection and Potential Money Laundering Network Analysis*

> ⚠️ 研究用途。資料為 IBM AMLSim 合成資料；模型分數只代表模型在特定資料與設定下的預測，**不是犯罪證據**，也不代表在真實銀行環境中的效能。

## 研究問題

- **RQ1** 圖結構資訊是否能改善異常金流偵測？（Rule-based vs. XGBoost vs. GraphSAGE）
- **RQ2** SHAP 與 GNNExplainer 能否提供忠實且穩定的判斷依據？
- **RQ3** 模型在時間切分、不同隨機種子、圖結構消融下是否穩定？

以上皆為待驗證問題，結果請見 `reports/`（隨進度更新）。

## 目前進度

- [x] Week 1：環境、資料取得、資料檢查、標籤與洩漏分析、資料字典、建模決策初稿
- [ ] Week 2：特徵工程、Rule-based、XGBoost、評估流程
- [ ] Week 3：圖建構、GraphSAGE
- [ ] Week 4–5：模型比較、SHAP、GNNExplainer、消融
- [ ] Week 6–8：Streamlit 平台、穩定性、報告與 Demo

## Week 1 主要資料發現

詳見 [`docs/data_dictionary.md`](docs/data_dictionary.md) 與 [`reports/week1/02_label_quality.md`](reports/week1/02_label_quality.md)。

1. 標籤只在**帳戶層級**（`isFraud`）；交易沒有標籤 → 主任務為帳戶節點分類。
2. `fraudStep` 為單向洩漏欄位，已排除。
3. 金額 < 100 的交易 100% 發生在詐欺帳戶之間（模擬器產物），實驗需同時報告「含金額」與「控制金額」兩種設定。

## 快速開始

需求：macOS / Linux、[uv](https://docs.astral.sh/uv/)、git。XGBoost 在 macOS 需要 OpenMP（`brew install libomp`）。

```bash
git clone https://github.com/<your-account>/aml-insight.git
cd aml-insight
uv sync                                             # 建立 .venv 並安裝所有套件
bash scripts/00_fetch_amlsim_samples.sh             # 下載 AMLSim 並解壓 20K 樣本到 data/raw/
uv run python scripts/01_inspect_files.py           # 檔案/欄位/品質檢查
uv run python scripts/02_label_quality_analysis.py  # 標籤/比例/洩漏分析
uv run pytest                                       # 基本測試
```

> 註：torch 限制為 `<2.12`，因為 2.12 以後不再提供 macOS 13 的 wheel。

## Member A：資料與圖建置

取得 AMLSim 樣本後，以下流程會輸出帳戶特徵、S1/S2 Parquet、NetworkX/PyG 圖資料、EDA 圖與資料檢查報告：

```bash
uv run python scripts/03_build_features_graph.py
uv run python scripts/04_eda.py
uv run python scripts/05_leakage_check.py
uv run python scripts/06_build_e2_variants.py
uv run python scripts/07_create_member_a_notebooks.py
uv run python scripts/08_embed_member_a_notebook_outputs.py
```

PyG 檔案會輸出到 `data/processed/graph_s1.pt`、`data/processed/graph_s2.pt`（`x`, `edge_index`, `edge_attr`, `y`, `node_id`）；`graph.pt` 是 S1 相容名稱。原始與衍生資料仍由 `.gitignore` 排除；執行腳本即可重建。Notebook 驗算與圖表在 `notebooks/A_week1_verify.ipynb`、`notebooks/A_week1_eda.ipynb`。

同一流程另產生 E3 的 `time_train_features.parquet` / `time_test_features.parquet`（別名 `account_features_steps1_89.parquet` / `account_features_steps90_149.parquet`）、`feature_groups.json` 與含資料/code 版本、視窗和交易/邊數的 `time_windows.json`。每個時間窗的帳戶與圖結構特徵只由該窗交易重建。

## 成員 C：GraphSAGE

在 A 的 `03_build_features_graph.py` 之後執行，訓練 S1/S2 × seeds 0–4（CPU 約 30 分鐘）：

```bash
uv run python scripts/09_train_graphsage.py
uv run python scripts/09_train_graphsage.py --setting S2 --seed 0   # 只跑一組
```

- 特徵、帳戶切分與欄位順序直接沿用 B 的 `load_features` / `stratified_split`，確保三個模型的比較條件一致。
- 2 層 GraphSAGE（mean 聚合），轉帳邊加上反向邊；不使用 `edge_attr`，所以 S2 不會帶入金額。
- 特徵先做 signed log1p，再用 **train 帳戶**的平均與標準差標準化；loss 只用 train 標籤。
- 超參數與 epoch 只依 validation AP 選擇。test 帳戶只輸出分數，test 指標由 B 的 `finalize` 計算一次。
- 輸出在 `data/processed/C/E1-<S>-seed<k>/`（`validation.csv`、`test.csv`、`metadata.json`），可直接填入 B config 的 `graphsage:` 區塊。

## 專案結構

```
aml-insight/
├── app/                    # Streamlit 平台（Home.py + pages/）
├── configs/                # 實驗設定檔（YAML）
├── data/
│   ├── raw/                # 原始資料（不進 git，由腳本重建）
│   ├── interim/            # 清理後中間資料
│   ├── processed/          # 特徵、圖資料
│   └── external/           # AMLSim 原始 repo（不進 git）
├── docs/                   # 資料字典、建模決策、週計畫、會議紀錄
├── experiments/            # 每個實驗的 config、seed、指標輸出
├── models/                 # 訓練好的模型（不進 git）
├── notebooks/              # EDA 與教學 notebook
├── reports/                # 自動產生的報告與圖表
├── scripts/                # 可執行流程腳本（00_、01_ … 依序）
├── src/aml_insight/
│   ├── data/               # 讀取、驗證、切分
│   ├── features/           # 特徵工程
│   ├── graph/              # 圖建構（NetworkX / PyG Data）
│   ├── models/             # rule-based、XGBoost、GraphSAGE
│   ├── explain/            # SHAP、GNNExplainer
│   ├── evaluation/         # 指標、實驗流程
│   └── viz/                # 圖表與網絡視覺化
└── tests/
```

## 資料來源

IBM AMLSim：<https://github.com/IBM/AMLSim>（官方 repo 內 `sample/*.tgz` 20K 樣本）。使用的 commit 記錄於 `data/raw/AMLSIM_COMMIT.txt`。

## 研究倫理

- 異常 ≠ 犯罪；系統定位為研究與人工檢視輔助工具，不做帳戶凍結或犯罪判定。
- 所有圖表與數字皆由程式實際執行產生，不使用虛構數據。
- 不使用標籤衍生欄位或測試期間的未來資訊建立特徵。

## 授權

研究與教學用途。AMLSim 資料依其原始授權（Apache-2.0）。


## 成員 B：評估與實驗工具

已新增指標、分層切分、S1/S2 規則、XGBoost validation-only 調參、一次性 test 評估、SHAP／忠實度、特徵消融與五 seeds 設定、來源可追溯的結果彙整。

```bash
uv sync --locked
bash scripts/check_b.sh -q
bash scripts/run_b.sh --help
```

操作與 A／C 接入格式：[docs/member_b_handoff.md](docs/member_b_handoff.md)。
研究方法：[docs/experiment_protocol.md](docs/experiment_protocol.md)。
設定範例：[configs/member_b.example.yaml](configs/member_b.example.yaml)。

目前尚無 A 正式特徵、C GraphSAGE 或時間特徵，因此實作驗證使用測試資料；不代表正式 E1–E5 已完成。E2 包含特徵與圖結構消融；目前提供特徵群組移除，圖結構變體與 GraphSAGE 結果待 A／C。E4 為可解釋 AI 案例分析：B 提供 XGBoost + SHAP，與 C 的 GraphSAGE + GNNExplainer 代表案例對照仍 pending。
