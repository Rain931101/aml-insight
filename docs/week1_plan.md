# Week 1 工作計畫與驗收標準

> 已完成的部分（由專案初始化產生）：資料夾架構、`uv` 環境、資料下載腳本、檔案檢查與標籤分析程式、資料字典初稿、建模決策初稿。
> 第一週的重點是：**每位成員親自重跑、讀懂、驗證**這些結果，並補上尚未完成的部分。

## 0. 全員共同（第 1–2 天）

| 任務 | 驗收標準 |
|---|---|
| 安裝環境（見 README「快速開始」） | `uv run python -c "import torch_geometric, shap, streamlit"` 成功 |
| 重跑 `00`、`01`、`02` 三支腳本 | 自己產生的 `reports/week1/*.md` 數字與 repo 版本一致 |
| 讀 `docs/data_dictionary.md`、`docs/modeling_decision.md` | 三人開會確認或修改決策，會議紀錄寫入 `docs/meetings/week1.md` |
| 閱讀：GraphSAGE 論文（Hamilton et al., 2017）摘要與第 3 節、AMLSim README | 每人寫 5 行重點於會議紀錄 |

## 成員 A：Data Engineering & Graph Construction

| 任務 | 驗收標準 |
|---|---|
| 手動驗證報告中 3 個關鍵數字（詐欺比例、金額 < 100 規則、fraudStep 交叉表），用 notebook 自己算一次 | `notebooks/A_week1_verify.ipynb`，數字與報告一致 |
| EDA notebook：金額分布、每 step 交易量、度數分布（依標籤分組） | `notebooks/A_week1_eda.ipynb`，至少 6 張圖，每張附 1–2 句解讀 |
| 決定 self-loop、重複交易、無交易帳戶的處理方式 | 寫入 `docs/data_dictionary.md` 第 4 節，含理由 |
| （選用，time-box 6 小時）以 Docker 嘗試執行新版 AMLSim | 成功：產出資料 + 步驟紀錄；失敗：記錄卡住的原因，停止 |

## 成員 B：Machine Learning & Experiment Evaluation

| 任務 | 驗收標準 |
|---|---|
| 撰寫 `src/aml_insight/evaluation/metrics.py`：PR-AUC、Precision@K、Recall@K、F1、Confusion Matrix | `tests/test_metrics.py` 以手算小例子驗證，`uv run pytest` 通過 |
| 撰寫切分函式 `src/aml_insight/data/split.py`：分層隨機 60/20/20、固定 seed | 測試：三份不重疊、正樣本比例差 < 0.5 個百分點 |
| 草擬 E1–E5 實驗協定（RQ、假設、自變數、應變數、設定、指標、限制） | `docs/experiment_protocol.md`，三人審過 |

## 成員 C：GNN & System Integration

| 任務 | 驗收標準 |
|---|---|
| 跑 PyG 官方 GraphSAGE 範例（Cora 或任一小資料）理解 `Data`、`edge_index`、`train_mask` | `notebooks/C_week1_pyg_intro.ipynb` 可執行 |
| 用 NetworkX 從主要資料集取出一個詐欺帳戶的 1-hop / 2-hop 子圖並畫出 | notebook 中有子圖圖片，節點依標籤上色、邊有方向 |
| 建立 Streamlit 骨架 `app/Home.py`：讀取資料並顯示帳戶數、交易數、詐欺比例 | `uv run streamlit run app/Home.py` 可開啟，數字來自實際資料 |
| 建立 GitHub repo、分支保護、Issue 標籤、Project 看板 | 見下方「GitHub 管理方式」全部完成 |

## GitHub 管理方式

- Repo：`aml-insight`（public），`main` 受保護，只能透過 PR 合併，至少 1 人 review。
- 分支命名：`feat/<成員>-<主題>`、`fix/...`、`exp/<實驗代號>`（例：`exp/e1-baseline`）。
- Commit 訊息：`<type>: <說明>`，type ∈ feat / fix / data / exp / docs / refactor / test。
- Issue 標籤：`P0`、`P1`、`P2`、`data`、`model`、`xai`、`app`、`experiment`、`bug`。每週開一個 Milestone（Week 1 … Week 8）。
- Projects 看板：Backlog → This Week → In Progress → Review → Done。
- **不要 commit**：`data/raw`、`data/external`、模型權重、`.env`（已在 `.gitignore`）。資料靠 `scripts/00_fetch_amlsim_samples.sh` 重建。
- 每個實驗結果（`experiments/<id>/`）要包含：config、seed、git commit hash、指標 JSON。

## 第一週交付物檢查表

- [ ] 三人皆可重跑資料檢查腳本
- [ ] 資料字典定稿
- [ ] 建模決策定稿（主任務、特徵排除、S1/S2、切分）
- [ ] 評估指標模組 + 測試
- [ ] EDA notebook
- [ ] Streamlit 骨架
- [ ] GitHub repo 與看板
- [ ] Week 1 會議紀錄

## 第二週預告（先不要做）

建立第一版特徵 → Rule-based baseline（含金額門檻規則與不含金額的規則）→ XGBoost（S1 / S2）→ 第一版評估流程。
