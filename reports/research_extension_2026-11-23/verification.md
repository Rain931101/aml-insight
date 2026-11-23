# 驗證與完成邊界（2026-11-23）

## 實際執行

- 55個AMLSim runs全部完成：30個GraphSAGE圖／特徵／時間條件、25個XGB條件，另10份train-quantile規則結果。
- 5個Elliptic runs全部完成：GraphSAGE、XGB、RF、LR各五seed；總計85份model/run指標。
- 每run有TEST_STARTED、COMPLETE、選參、預測與結果SHA；summarize逐一核對通過。
- SHAP：原S1/S2各200帳戶、方向性遮罩、10個隨機對照、三個train背景抽樣。
- GNNExplainer：8個事先定義選法的XGB邊界案例×3種子，24份邊遮罩、隨機對照與穩定性。
- 真實兩跳子圖／全圖目標logit完全一致（本次8例最大誤差0）。
- 原封存資料與模型不修改；沒有重跑舊finalize。
- 全套pytest：**61 passed、5個既有第三方棄用警告**，完整輸出tests.log。
- Ruff檢查涵蓋app、src、tests、research與平台匯出腳本；git diff --check。
- 新增圖表已直接查看，未見文字裁切或數值軸錯誤。

## 安全／研究邊界

- 本輪AMLSim使用已看過test，明列exploratory follow-up，不作新的獨立最終測試。
- 原E1為500epochs；本輪GNN各條件3候選×200epochs，不混報。
- 外部Elliptic是另一個交易節點任務，使用官方本機資料，unknown不作負例。
- 外部train/val/test時間圖邊界與known-label split已測試；原始公開特徵前處理仍有不可驗證限制。
- 五seed不是五份獨立資料，沒有用重疊test做虛假的獨立樣本顯著性結論。
- 解釋檢查保留不穩定與負面案例；遮罩非因果。
- 未選擇更好test後再改grid、budget或閾值。
- 受試者、訪談與人工實測依使用者最新要求略過，沒有相關結果或背書。
- 文獻是有範圍的原始來源對照，兩篇較接近的2026工作全文未能讀取，不宣稱首創或窮盡文獻。
- 沒有GitHub推送、PR或網站部署；沒有公開原始金融研究資料、外部逐筆衍生資料與模型權重。

## 失敗與修正保留

訓練前import錯誤沒有產生run/test評估，修正後人工重啟，原log保留。
第一次GNN解釋由兩跳一致性檢查發現dropout模式回復錯誤，失效資料隔離且不納入；
修正後有專門回歸測試，沒有重訓或修改封存模型test指標。
run_amlsim的兩版來源差異僅兩條純定義import排序；除import外AST一致，原檔已保留。
核心訓練器和GraphSAGE模型程式來源雜湊一致。

## 最終獨立指標回放

85份新model/run的AP、F1與混淆矩陣均由獨立sklearn計算核對，AP誤差<1e-7。
這是已封存預測的算式核對，不是重新選模或新的科學test評估。
原10份E1及原SHAP的完整性封印仍全部通過。見metric_replay_verification.json。
