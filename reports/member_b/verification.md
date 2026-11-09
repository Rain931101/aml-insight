# 成員 B 交付驗證

更新日期：2026-11-09。範圍只含成員 B；沒有 A 特徵工程、C GraphSAGE／GNNExplainer 或正式研究結果。

## 最新修訂

- E2 定義包含節點特徵與圖結構消融。目前批次工具只產生特徵群組移除；保持度數重接邊、移除邊及 GraphSAGE 結果待 A／C 接入。
- E4 定義為可解釋 AI 案例分析。B 已提供 XGBoost + SHAP；相同代表帳戶的 GraphSAGE + GNNExplainer 對照 pending C。
- `load_config` 接受 E4，新增回歸測試並持續拒絕 E6。E4 案例分析可沿用 E1 鎖定 run 的 explain，不需重新評估 test。
- 安裝統一 `uv sync --locked`，啟動統一 `uv run`；移除 B 專用 lockfile／setup、PYTHONPATH 與 DYLD_LIBRARY_PATH 腳本繞路。
- E3 批次維持全／早／晚三份輸入，缺早視窗或與全視窗同路徑時拒絕。

## 2026-11-09 本機驗證

| 項目 | 結果 |
|---|---|
| `uv sync --locked` | 成功；使用既有 pyproject.toml + uv.lock，未修改其內容 |
| pytest（見下方環境限制） | **31 passed、5 skipped、3 warnings**；原有 30 項加 E4 設定回歸測試 |
| `uv run ruff check src tests scripts/smoke_b.py` | 通過 |
| `bash -n scripts/run_b.sh scripts/check_b.sh scripts/smoke_b.sh` | 通過 |
| `bash scripts/run_b.sh --help` | 通過；安裝後可找到 aml_insight，不需 PYTHONPATH |
| `git diff --check` | 通過 |
| uv run 合成 smoke（同下方程序限定 runtime） | S1／S2 × 五 seeds 共十次 tune／finalize、兩份 SHAP、seed 彙整及排除測試資料的正式空報告通過 |

五項 skipped：既有 test_loader.py 因缺原始 AMLSim 資料的 module marker 跳過。
三項 warnings：SHAP 使用 Matplotlib colormap 的 PendingDeprecationWarning，未隱藏。

### 本機 OpenMP 限制

這台 macOS 沒有 Homebrew／系統 libomp。按新腳本直接執行完整測試時，XGBoost 載入失敗：16 failed、15 passed、5 skipped。CLI help 不載入模型，不能代表 XGBoost 可執行。
README 標準前置步驟為 `brew install libomp`。本次沒有安裝 Homebrew 或改動系統路徑，也沒有把舊繞路加回提交腳本。
為核對程式修訂，**僅在以下本機驗證程序**暫用已安裝 sklearn wheel 的 runtime：

```bash
env DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/sklearn/.dylibs" \
  uv run python -m pytest -q
```

上述條件下為 31 passed；這不等於本機已完成 `brew install libomp` 後的新腳本全流程驗證。其他 Mac 請先安裝 README 指定的系統依賴。

## 重要行為涵蓋

手算 AP／F1／Top-K／矩陣、單類／同分／超大 K；五 seeds 的比例與互斥；洩漏欄位及特徵分類拒絕；validation 規則分位數、train-only class weight；test 變更不影響選模；同 run 拒絕再次 final；結果完整性檢查；C 的 ID／特徵／版本對齊；特徵消融停用不適用規則；E3 早／晚快照；SHAP probability 加總與 train 背景／中位數遮罩；平均／樣本 SD／最佳值與不足五 seeds 狀態；E4 設定接受與非法實驗拒絕。

## 正式結果邊界

合成 smoke 皆為 `purpose=test`，不能作 AMLSim 論文數字。正式彙整排除此類 run；尚無正式結果時保持 pending。
仍需 A 正式特徵、圖變體、時間窗口與 provenance，及 C 相同 seed／設定／split 的 GraphSAGE 預測與 GNNExplainer 案例。
RuleBased 本次保留原本 validation 分位數方法；同一 validation 用於規則分布及 F1 閾值，存在樂觀偏差，已在 protocol 說明。改用 train 分位數屬可選後續方法變更，尚未採用。
