# 成員 B 操作與接入說明

## 安裝與檢查

在專案根目錄：

```bash
uv sync --locked
bash scripts/check_b.sh -q
bash scripts/run_b.sh --help
```

套件統一由 `pyproject.toml` + `uv.lock` 管理，以 `uv sync --locked` 安裝、`uv run` 執行；不另維護 B 版本清單。Python 需 3.11–3.12。
macOS 先執行 `brew install libomp`；啟動腳本不再設定 PYTHONPATH 或 OpenMP runtime 繞路。
SHAP／Matplotlib 的第三方 PendingDeprecationWarning 不影響本次驗證，沒有隱藏警告。

## A 需要提供什麼

1. `data/processed/account_features.parquet`（或 CSV）：唯一整數 nodeid、二元 isFraud、有限數值特徵，每列一帳戶。
2. 更新 `configs/member_b.example.yaml` 的 feature_spec.groups，所有特徵恰好分類一次。amount 必須存在（可空），包含起始餘額與所有金額衍生特徵。behavior／structure／temporal 名稱可調整。
3. 調整 S1／S2 規則到真實存在的特徵。範例欄位只是接入示例，**不是已完成的 A 特徵工程**。
4. 選用 `account_analysis_metadata.csv`：恰好全體帳戶 `nodeid,has_small_amount`，後者 0/1，只供子群分析。
5. E3 需相同帳戶與靜態標籤的早／晚窗口特徵表、界線及無未來交易的版本聲明。早視窗供 train/validation，晚視窗僅供 test 帳戶。

禁止特徵：nodeid、tx_id、isFraud、fraudStep 及任何標籤衍生資訊。程式能檢查明確欄位與對齊，不能從任意改名欄位證明無語意洩漏。

## 一次實驗

先複製設定並改用真實特徵，再執行：

```bash
bash scripts/run_b.sh split configs/member_b.example.yaml experiments/manifests/seed0.json
bash scripts/run_b.sh tune configs/member_b.example.yaml
# 檢視 experiments/member-b-E1-S1-seed0/selection.json 後才執行最終 test：
bash scripts/run_b.sh finalize experiments/member-b-E1-S1-seed0
bash scripts/run_b.sh explain experiments/member-b-E1-S1-seed0 --samples 200 --background 100
bash scripts/run_b.sh aggregate experiments reports/member_b/results
```

`tune` 不產生 test metrics。`finalize` 每 run 只允許一次，禁止覆寫。
調參前會建立輸出目錄；失敗的目錄會保留診斷，不自動覆蓋；修正後用新的 run_id。
source 或模型／split／config 快照改動會阻止 finalize，需重新 tune。
分析舊結果不要求當前 source 等同舊 source，但會驗證當時保存的 artifacts 與 final 結果指紋。
不要載入來路不明的 rule.pkl；它是此工具在本機 tune 階段產生的可信 pickle。

## C 的接入

先使用上面的 split 指令輸出 manifest。C 必須使用此 manifest 的 ID、相同特徵版本與 seed，validation/test 各交付 CSV 或 Parquet，且只含 `nodeid,score`，score 為 [0,1] 有限值。
每份帳戶集合需精確相符；多列、漏列、重複與跨 split 分數會拒絕。

C 另提供 JSON：

```json
{
  "seed": 0,
  "setting": "S1",
  "manifest_sha256": "SHA256 of exported manifest.json bytes",
  "feature_file_sha256": "SHA256 of A feature table bytes",
  "feature_version": "A feature version or commit",
  "selected_columns": ["exact", "ordered", "feature", "list"],
  "training_config": {"architecture": "GraphSAGE", "epochs": 100, "train_labels_only": true}
}
```

`selected_columns` 必須與 B 依 groups 順序選出的欄位完全相同（含 E2 消融與 S2 移除 amount）。
`training_config` 記錄完整訓練與圖設定，seed 放頂層；不同 C 設定不可合併穩定性數字。
E3 另加 `time_feature_file_sha256`。檔案 SHA256 可用 `shasum -a 256 <file>` 取得。

將 graphsage.metadata／validation／test 及 split_manifest 路徑填入 config，再 tune。
B 只在 validation 決定 C 的分類閾值，不代替 C 訓練；圖的邊／特徵不得因 S2 偷帶金額資訊。
沒有 C 檔時兩模型仍可跑，但同一 run final 後不能追加 C 並重測；要比較三模型，請在 final 前準備好 C 的輸入。

## 批次設定、E2／E3／E5

```bash
bash scripts/run_b.sh suite configs/member_b.example.yaml configs/generated-member-b
```

只產生可檢閱的 YAML，不自動訓練或看 test。相對路徑會轉成絕對路徑。
輸出 E1 S1/S2 × 五 seeds、可行的 E2 非金額群組移除 × 五 seeds。
產生時間實驗時，base config 的 `features` 是全視窗基準，另提供 `time_train_features`（早視窗）及 `time_test_features`（晚視窗）。
產生器只把 E3 的 `features` 改為早視窗，E1 保留全視窗；缺少獨立早視窗路徑會報錯。不要以全視窗特徵訓練 E3。
目前產生的 E2 特徵消融只跑 XGBoost；圖結構消融（保持度數重接邊、移除邊）待 A 提供受控變體及 C 的 GraphSAGE 結果，尚未接入批次工具。S1 移除 amount 已等於 S2，避免重跑。E5 由 E1 五 seeds 彙整。
C 輸入／manifest 不會從 base 複製到所有 seeds，需每份設定接入正確變體。
每份設定依序 tune → 檢視 validation 記錄 → finalize；需要解釋的 run 再 explain。
若保留同一組多個候選研究 run，aggregate 會拒絕重複 seed，請指定只含選定 run 的資料夾。

## 每個 run 的交付

- config.json、provenance.json、columns.json、manifest.json：設定、版本、source／資料 hash、帳戶切分。
- train/validation/test.parquet：鎖定快照，E3 test 使用晚視窗。
- selection.json：全候選 validation AP、選定超參數、validation metrics、規則與分類閾值。
- xgboost.ubj、rule.pkl（若啟用）：選定模型；integrity.json：調參檔案指紋。
- metrics.json、predictions.csv、pr_curve.png：一次最終 test 評估。
- common_false_negatives.csv、error_analysis.json／.md：共同漏判與無小額交易 recall。
- result_integrity.json、FINAL_EVALUATION_STARTED.json、COMPLETE.json：測試鎖與結果完整性。
- shap/：summary、waterfall、重要度、個別 SHAP、背景／抽樣 ID、加總驗證與忠實度結果。

## W1–W8 驗收邊界

| 週 | B 已提供的實作 | 正式驗收尚需 |
|---|---|---|
| W1 | 手算指標測試、分層切分、E1–E5 協定 | A 正式特徵版本 |
| W2 | S1/S2 規則、XGBoost、統一 config runner、JSON／seed／provenance | 真實輸入後執行 |
| W3 | validation-only grid、逐候選記錄、test 排他鎖 | 正式 validation 選模與 final |
| W4 | 三模型接入、PR 圖、共同漏判／子群分析 | C 預測、A 分析 metadata |
| W5 | SHAP 全域／個別、消融 config、masking 忠實度 | A 群組定義、正式 run |
| W6 | 時間窗口驗證與早／晚特徵接入、E3 config | A 早／晚窗口特徵與聲明 |
| W7 | 五 seeds configs、mean／sample SD／best／有效數 | E1 五次完整真實執行 |
| W8 | 自動結果章節草稿及每個數字的來源連結 | 真實結果後撰寫支持的結論 |

完整方法與限制見 experiment_protocol.md。現階段不應把「工具已實作」寫成「八週正式研究已完成」。


## 指標與參數速查

| 實作 API／參數 | 用途與判讀 |
|---|---|
| sklearn.average_precision_score | AP 主指標；相對 prevalence 基線判讀，不能與梯形積分混稱 |
| sklearn.precision_recall_curve | 產生 PR 曲線及 validation 的 F1 閾值候選 |
| sklearn.roc_auc_score | 輔助排名指標；極不平衡時需同看 AP 與檢視預算 Top-K |
| sklearn.confusion_matrix | 固定 labels=[0,1]，避免單類資料讓矩陣形狀改變 |
| XGBoost max_depth | 樹越深越能捕捉複雜交互作用，也更易過擬合 |
| XGBoost learning_rate | 每棵樹的更新幅度；較小通常需更多 n_estimators |
| XGBoost n_estimators | 樹的數量；與 depth／learning_rate 一起只由 validation 比較 |
| XGBoost scale_pos_weight | train 負／正樣本比，不使用 validation 或 test 標籤計算 |
| SHAP TreeExplainer | 解釋指定模型與背景分布；SHAP 大小不是因果效果 |

例如正樣本僅 1% 時，全預測正常也有 99% accuracy，卻有 0 recall；因此不以 accuracy 決定研究結論。

## 已保存的流程驗證

`reports/member_b/smoke/` 在本機保存 S1／S2 × 五 seeds 的合成測試輸出與兩份 SHAP 圖（不隨 Git 上傳，可用下方指令重建），
全部標記 purpose=test，僅驗證程式可執行，**不得放入論文當成 AMLSim 實驗結果**。
預設 aggregate 會排除這些 run。可使用新目錄重現：

```bash
bash scripts/smoke_b.sh /tmp/aml-b-smoke-new
```

執行 aggregate 後才會產生 `reports/member_b/results/results.md`；全新 clone 不包含本機產出的報告。目前尚無正式結果，產生的章節應如實標示 pending。

## E4 可解釋 AI 案例分析

沿用已鎖定的模型、切分與 SHAP 輸出，挑選代表性 TP／FP／FN／TN 帳戶，保存選例理由及 nodeid，與 C 相同帳戶的 GraphSAGE + GNNExplainer 對照。不得因案例分析重新選參或重跑 test 評估。完整案例分析 pending C；現有 explain 可直接用於 E1 完成的 run，不必為 E4 重訓同一模型。
