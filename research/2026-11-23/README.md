# AML Insight 研究補強執行包

這個版本用來補資料可信度、控制實驗、時間視窗與解釋檢查。先讀 `protocol.md`、
`protocol_lock.json`、`scope_update.md`、`literature_review.md` 與 `external_data_card.md`。
**不是新演算法，也不保證競賽名次。網站沒有部署。**

## 結果與原始輸出

- 彙整與圖表：`reports/research_extension_2026-11-23/`。
- AMLSim新增模型／預測：`experiments/research-extension-20261123/`。
- 外部資料與逐筆輸出：`data/processed/research_extension_20261123/elliptic/`（ignored，本機用）。
- 外部原始ZIP／CSV：`data/external/elliptic/`（ignored，不重發布）。
- 原E1封存保持不變：`experiments/member-b-amlsim-e1-20261116/`。

## 在新的乾淨工作副本重現

不得刪除或覆寫已封存結果來重跑。請另建空白工作副本，確認原始資料與輸入雜湊。

```bash
uv sync --locked
uv run python -c 'from aml_insight.data.fetch import ensure_samples; ensure_samples()'
uv run python scripts/03_build_features_graph.py
uv run python research/2026-11-23/audit.py
uv run python research/2026-11-23/prepare.py
uv run python research/2026-11-23/batch.py
```

研究主環境採fe14d70基線的特徵定義（16特徵，S2選10）；更換source或特徵版本要另立研究版本。
外部資料使用Kaggle官方下載（不是不明鏡像），將官方三個CSV置於data/external/elliptic/，
保留原始雜湊與官方metadata後：

```bash
uv run python research/2026-11-23/prepare_external.py
uv run python research/2026-11-23/batch_external.py
uv run python research/2026-11-23/explain_shap.py
uv run python research/2026-11-23/explain_gnn.py
uv run python research/2026-11-23/subgroups.py
uv run python research/2026-11-23/duplicate_sensitivity.py
uv run python research/2026-11-23/calibration_diagnostics.py
uv run python research/2026-11-23/summarize.py
```

SHAP舊模型檢查依賴舊E1封存模型與200個解釋樣本；這些不是Elliptic資料。
每個run資料夾排他建立，test評估前另有排他鎖；失敗不得靜默重試。batch遇到現有run不重跑，
summarize仍要求COMPLETE與雜湊全部通過，不能把跳過當成功。

## 執行失敗紀錄

- 首次批次在進入任何訓練前遇到import錯誤，沒有產生run/test鎖；保留
  `preflight_import_failure/`，修正導入後人工重新啟動。
- 首次GNN解釋因包裝模型回復training模式，導致dropout與兩跳子圖一致性失敗；
  失效輸出保留在 `gnn_explanations_failed_dropout_mode/`，不納入報告。
  修正wrapper.eval並加入回歸測試後，重新產生獨立解釋目錄，沒有重訓或重評模型test。
- run_amlsim兩個來源雜湊差異僅為兩條純定義import的排序；原檔與AST一致性證據
  在 `source_archive/`。核心訓練器與GraphSAGE程式雜湊一致。

## 交付GitHub時

交付自寫程式、協定、文件、測試與精簡彙總。原始資料、外部逐筆衍生資料、
模型權重、二進位大型輸入與本機環境不直接上傳；依來源授權與小組流程另行保存。
本次沒有推送、開PR或部署。人工作用驗證依使用者要求先略過，模板不代表結果。
