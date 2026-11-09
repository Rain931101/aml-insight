# 成員 B 階段工作日誌

紀錄期間：2026-11-07 至 2026-11-09。依組長審查精簡；舊版完整日誌仍可從 Git 歷史查閱。

## 目前成果與範圍

只實作 B：指標、切分、Rule-based、XGBoost、實驗執行／彙整、SHAP／忠實度與接入文件。
沒有實作 A 特徵工程、C GraphSAGE／GNNExplainer，也沒有宣稱正式 E1–E5 結果完成。

- 指標：AP、ROC-AUC、F1、混淆矩陣、Precision／Recall@100、500、正樣本數；同分依 nodeid 決定。
- 切分：帳戶分層 60/20/20，seeds 0..4；互斥、覆蓋、類別比例檢查。
- 防洩漏：排除 ID／標籤欄位，所有特徵明確分類；S2 移除 amount 全群。
- 訓練：XGBoost class weight 只用 train；超參數與最終 F1 閾值只用 validation。
- Test：同 run 的排他評估鎖及來源／結果完整性檢查；不能防止刻意另開 run 反覆偷看 test。
- SHAP：train 背景、機率加總驗證、summary／waterfall／個別特徵貢獻；以 train 中位數遮罩及隨機對照分析忠實度，不作因果宣稱。
- 彙整：預設排除合成測試，保留來源，報平均／樣本 SD／最佳值；不足五 seeds 標 pending。

## 工作歷程

| 階段 | 工作與證據 |
|---|---|
| 11/07 | 盤點 ZIP 專案、確認只處理 B，建立操作契約與實驗協定；初次 27 passed、5 skipped |
| 11/07–11/09 | 完成十次合成 tune／finalize、兩設定 SHAP 與彙整；皆為測試證據 |
| 11/09 初次交付 | 將 B 檔案上傳 feat/member-b-evaluation，建立 PR #2，保留主分支及其他成員內容 |
| 11/09 階段複查 | 修正 E3 suite 把全視窗誤當早視窗的問題，新增三個回歸案例；30 passed、5 skipped、Ruff 通過 |
| 11/09 組長回饋 | 組長實跑 30 項及 Ruff 通過，補充 E2／E4 定義並要求統一 uv |
| 11/09 本次修訂 | 同步方法、文件、設定與報告狀態；移除另一份 lockfile／setup；新增 E4 回歸測試；31 passed、5 skipped（本機 OpenMP 限制見驗證紀錄） |

## 本次審查修訂

1. **E2**：特徵與圖結構消融。現有工具只做節點特徵群組移除；A 的受控重接／移除邊與 C 的 GraphSAGE 結果尚待接入。
2. **E4**：可解釋 AI 案例分析。B 的 XGBoost + SHAP 已有工具；與 C 的 GraphSAGE + GNNExplainer 相同帳戶對照仍 pending。可重用已完成 E1 run 的 explain，避免重測 test。
3. **設定與輸出**：load_config 接受 E4；suite pending 與 results.md 不再聲稱 E4 未定義。
4. **環境**：只用 pyproject.toml + uv.lock；uv sync --locked／uv run。不修改團隊 lockfile，刪除 requirements-b.lock、setup_b.sh 及啟動腳本的 PYTHONPATH／DYLD 設定。
5. **文件**：更新 protocol、README、handoff、verification，精簡本日誌。
6. **選配建議**：暫保留原約定的 validation 分位數規則，明載樂觀偏差；train 分位數方法可另行比較，不混入本次必要修訂。

## 驗證與環境限制

- `uv sync --locked` 成功；既有 pyproject.toml、uv.lock 未變更。
- uv 環境下 pytest：31 passed、5 skipped、3 個第三方 SHAP warnings；Ruff、bash 語法、CLI help 與 diff 空白檢查通過。
- 本機沒有 Homebrew／系統 libomp，移除腳本繞路後直接測試會因 XGBoost 載入失敗。README 要求 Mac 執行 brew install libomp。
- 本次 pytest 成功有使用**程序限定的本機 runtime 環境變數**，沒有改回腳本；不可宣稱已在標準 Homebrew 環境驗證。完整命令、失敗與成功條件見 verification。
- 先前與本次合成資料均不可作正式論文數字；本機生成結果不提交。

## 下一階段交接

| 成員 | 尚需輸入 |
|---|---|
| A | 正式帳戶特徵及分組、全／早／晚視窗、圖消融變體與版本／seed／控制條件、選用子群 metadata |
| C | 每個相同 seed／設定／split 的 GraphSAGE 預測與訓練 provenance、圖變體結果、相同代表帳戶的 GNNExplainer |
| B | 收到正式資料後依協定選模與一次 test、SHAP 案例、正式表圖及五 seeds 統計；不以 synthetic 填補 |

正式資料前可交付的是可重現流程與驗證，完整研究結論仍 pending。
PR：[Rain931101/aml-insight #2](https://github.com/Rain931101/aml-insight/pull/2)，合併由組長審查通過後處理。
操作：[member_b_handoff.md](../member_b_handoff.md)；方法：[experiment_protocol.md](../experiment_protocol.md)；驗證：[verification.md](../../reports/member_b/verification.md)。
