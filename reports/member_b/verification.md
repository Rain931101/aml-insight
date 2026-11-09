# 成員 B 交付驗證

日期：2026-11-07。實作範圍為成員 B，未實作 A 特徵工程或 C GraphSAGE。

## 已執行

- `bash scripts/check_b.sh -q`：27 passed、5 skipped。
- 跳過原因：原始 AMLSim 資料尚未下載，既有 test_loader.py 的五項測試由原本 module marker 跳過。
- 三項警告：SHAP 的第三方 Matplotlib colormap PendingDeprecationWarning；未隱藏，未造成失敗。
- ruff 檢查 src、tests、scripts/smoke_b.py：通過。
- `uv pip check --python .venv/bin/python`：32 套件相依相容。
- bash 語法檢查 setup_b.sh、run_b.sh、check_b.sh、smoke_b.sh：通過。
- CLI help、suite 設定產生、正式空結果報告：通過。
- 合成 CLI smoke：S1/S2 × seeds 0..4，共十個 tune/finalize，兩份 SHAP／忠實度、seed 統計彙整通過。
- 圖片檢查：已查看 S2 seed0 PR curve 及 SHAP summary，圖面正常。
- 獨立程式檢查：外部預測特徵契約、結果完整性、C 訓練版本分組、正樣本數 K 的語意彙整已修正並加入回歸測試；複查未發現剩餘重大問題。

## 重要行為測試

手算 AP／F1／Top-K／混淆矩陣；單類／同分／超大 K；五 seeds 分層比例與互斥；
洩漏欄位／缺特徵分類拒絕；validation 規則分位數；train-only class weight；
修改 test 特徵不影響選模；同 run 拒絕再次 final；修改結果拒絕彙整；
C 的 ID 與特徵版本對齊、三模型接入；E2 停用移除特徵的規則；E3 早／晚特徵快照；
SHAP probability 加總驗證、train 背景與 train median masking；mean／sample SD／best 與不足五 seeds 狀態。

## 正式結果邊界

`smoke/` 內全為 purpose=test 的合成驗證，不是 AMLSim 正式結果。
`results/results.md` 如實標示 pending，未填入虛構研究數字。
需要 A 正式特徵、群組與時間視窗資料，以及 C 每個相同 seed／設定／split 的預測及 provenance。
E4 原圖未定義，保持未定義。
