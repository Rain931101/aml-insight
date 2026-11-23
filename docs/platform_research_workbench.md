# 研究分析平台改版（2026-11-23）

## 本機預覽

```bash
uv sync --locked
uv run streamlit run app/Home.py --server.port 8511
```

入口仍為 `app/Home.py`，保留 Streamlit，不另建一套前端。線上部署使用
`app/requirements.txt`，只需要 Streamlit、pandas、NumPy、Plotly，沒有模型推論依賴。
原有線上站尚未更新；本次為本機改版，須透過小組 PR 審查後部署。

## 工作區

- 研究總覽：三模型 S1/S2 的 AP 與 seed 樣本標準差、研究進度與閱讀路徑。
- 資料鑑識：原始交易時間窗、金額分布、小額交易與帳戶標籤的關聯、資料品質與 CSV 匯出。
- 模型實驗：五 seeds 的 AP / ROC-AUC / F1 / P@100 / R@500，PR 曲線、固定 validation 門檻混淆矩陣。
- 帳戶與網絡：test 帳戶風險排序、FP/FN/三模型皆漏判、帳戶搜尋、一階交易子圖與交易軌跡。
- 解釋證據：seed 0 的 S1/S2、200 個 test 樣本 Tree SHAP、加總驗算與遮罩忠實度結果。
- 方法與來源：資料版本、來源檔、SHA-256、研究限制及來源 JSON 下載。

工作區寫入 URL `view`，可分享頁面；資料集、模型與帳戶篩選使用 Streamlit 控制項。
無可用結果時停止展示，不能將其他資料集套入 fanin200cycle200 的模型結果。

## 證據封裝

`app/assets/research/` 是從現有 2026-11-16 E1 封存實驗匯出的唯讀資料。
`python scripts/export_platform_evidence.py` 先核對原始 integrity / result_integrity / SHAP
封印再複製；不調參、不訓練、不執行 finalize。匯出 10 runs、80 個檔案約 4.1 MB。

證據包含 metrics、test predictions、manifest、欄位、選參與 provenance；SHAP 僅 seed 0。
receipt 保留原始 metrics 雜湊、相對來源與封裝檔案雜湊。平台讀取時再次驗證。
原始 CSV 雜湊不符時停止，防止模型證據與資料版本混用。

模型基線為 fe14d70、16 個特徵，早於 min_amount；不能稱為最新特徵模型。
GraphSAGE 分數是 C 已合併訓練流程的本機重現，不宣稱與 C 的既有分數逐位相同。
五 seeds 使用相同帳戶母體，並非五份獨立資料。圖中誤差棒不是信賴區間。

## 嚴格區分

1. 原有 `isFraud` 是模擬帳戶標籤，不是網站判定或交易標籤。
2. 排序使用既有模型分數，沒有線上推論或任意新資料上傳功能。
3. 子圖為一階交易關係，最多 40 個有向帳戶對；圖面省略方向，表格保留方向。
   它不是 GNNExplainer；鄰居不賦予虛構分數或正常標記。
4. 遮罩後的平均分數下降可能是負值，忠實度尚不能宣稱通過。
5. E2 圖消融、E3 三模型、完整 E4 對照仍待補。改版沒有把未完成研究包裝為完成。
6. test 已被檢視。後續實驗須揭露，不得重用同一 test 宣稱新的獨立最終驗證。

## 驗證

新增測試以 sklearn AP 獨立重算全部 30 個模型/seed 指標，驗證平台 PR 曲線的同分群組處理，
並透過 Streamlit AppTest 執行六工作區。另執行全套測試、Ruff 與桌面 / 390px 手機瀏覽器檢查。
資料只讀：不覆寫既有封存檔案，也不訓練模型。

### 本次執行結果

- 全套 pytest：56 passed，5 個既有第三方棄用警告。
- 平台測試：10 passed；含六頁載入、30 組 AP 核對、同分 PR 曲線、FN 篩選、資料集邊界與導航持續狀態。
- Ruff：app、src、tests、匯出腳本通過；git diff --check 通過。
- Chrome：桌面研究總覽、帳戶頁、模型頁，390 × 844 手機帳戶頁，深色對比修正後確認。
- 本機瀏覽器未記錄 error 層級訊息；已回復一般桌面 viewport，保留預覽分頁。
- 本機服務 http://127.0.0.1:8511；未 push、未部署、未更動正式線上平台。

## 2026-11-23 研究驗證整合

新增研究驗證工作區：85 筆封存指標、圖與時間控制、Elliptic 方法重現、GNNExplainer/SHAP 可靠性與完整報告下載。使用 web_receipt.json 驗證展示來源；與舊 E1 500 epochs 分版，沒有重新評估 test。正式網站尚未部署。
