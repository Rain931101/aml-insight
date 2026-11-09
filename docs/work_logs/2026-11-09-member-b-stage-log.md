# AML Insight 成員 B｜目前階段工作日誌與複查紀錄

**紀錄期間：2026-11-07 至 2026-11-09**  
**本次複查日期：2026-11-09（Asia/Taipei）**  
**工作範圍：機器學習與實驗評估，僅成員 B**  
**小組 repository：<https://github.com/Rain931101/aml-insight>**  
**交付 PR：<https://github.com/Rain931101/aml-insight/pull/2>**

本日誌依協作過程、實際程式、GitHub 狀態與重新執行的命令整理，不虛構每日工時、正式實驗數字或組員已交付的內容。W1–W8 是原分工表的驗收項目，不代表本次已完成八週正式研究。程式由 AI 協作實作與檢查，研究設計及論文結論仍需小組確認。

---

## 1. 階段結論：目前是否可以交付？

**可以作為「成員 B 的程式與流程階段成果」交給小組審查；尚不能作為「正式研究實驗與論文結果已完成」的結案成果。**

本次已重新驗證安裝、指標、切分、規則模型、XGBoost、選模隔離、測試鎖、外部模型接入、SHAP、忠實度、消融／時間設定與五 seeds 彙整。另發現並修正 E3 批次設定的特徵視窗問題。

| 面向 | 截至本次複查的狀態 | 判讀 |
|---|---|---|
| B 程式與測試 | 已完成實作與重新驗證 | 可供組員審查、接入正式資料 |
| 新 Git 副本的獨立環境 | 已建立新的 .venv，利用既有 uv 快取安裝成功 | 不是沿用原下載資料夾的 .venv，也不是全新網路下載驗證 |
| 全部 pytest | **30 passed、5 skipped、3 warnings** | 5 項跳過、3 項警告原因見第 8 節 |
| Ruff | 通過 | 覆蓋 src、tests、smoke_b.py |
| 命令列全流程 | S1/S2 × 5 seeds 共 10 個合成 run 完成 | 驗證程式可執行，不是 AMLSim 正式研究 |
| SHAP | 兩份 smoke run 完成，機率空間加總誤差通過 | 不等於全部正式 test 帳戶已解釋 |
| E3 批次設定 | 複查發現問題後修正，新增三項回歸案例 | E1 全視窗與 E3 早視窗現在分開指定 |
| GitHub 上傳 | 初版 28 個變更檔案已逐檔核對 | 本次修正與日誌在同一 PR 延續交付 |
| GitHub 整合 | 檢查時 PR open、未 merged、可自動合併但 blocked | 尚需具適當權限的組員處理審查／保護規則 |
| 正式 A 特徵與時間資料 | 未提供到本次檢查副本 | 正式 E1/E2/E3/E5 尚未執行 |
| C 的正式 GraphSAGE 分數 | 未提供 | 三模型正式比較尚未完成 |
| E4 | 原圖未定義 | 未自行發明規格或結果 |

本次檢查沒有進行 main 合併、部署、修改分支保護或替其他組員批准 PR。

## 2. 原始任務、範圍與設計決定

### 2.1 原始需求

使用者要求依圖片的「成員 B：機器學習與實驗評估」工作表，在現有 aml-insight-main 專案中完成對應工作；允許自行安裝需要的套件，遇到問題可提問。

工作表列出的 B 責任包括：

- 公平比較 Rule-based、XGBoost 與 C 提供的 GraphSAGE。
- 建立評估指標、分層切分、統一實驗執行器與可追溯的輸出。
- validation-only 調參與閾值決定，最終 test 僅評估一次。
- PR 圖、共同漏判分析、無小額交易正樣本 recall。
- SHAP 全域、個別解釋與移除重要特徵後的分數變化。
- 接入 A 的消融資料、時間特徵，以及五 seeds 的穩定性比較。
- 撰寫能追溯實驗檔案的結果章節。

### 2.2 已確認的責任界線

使用者明確確認「先只完成成員 B」。因此沒有代替 A 建立正式帳戶特徵、沒有代替 C 實作正式 AML GraphSAGE，也沒有以臨時自製資料填入研究結果。

沿用專案既有 Python 架構，不更換技術棧、不新增產品前端，也不將帳戶標籤錯當成交易標籤。

### 2.3 主要決定與理由

| 決定 | 原因 | 代價／限制 |
|---|---|---|
| 介面優先，讀 A 的特徵、接 C 的預測 | 避免越過組員分工與版本不一致 | 需要 A／C 到齊後才有正式比較 |
| S1 保留合法金額，S2 排除 amount 群組 | 既有 AMLSim 分析顯示金額捷徑 | 特徵是否正確分類仍需 A 負責 |
| 保留 train/validation/test 快照及 manifest | 讓三模型使用相同帳戶集合 | 每次資料與程式更動都應建立新實驗 |
| tune 與 finalize 分開 | 避免 test 反饋影響選模 | 正式 final 後不能同 run 加模型重測 |
| E5 引用 E1 的五個 seeds | 避免為穩定性重複評估相同 test | 需事先跑滿五個 seeds，不能只挑最好一次 |
| E2 的金額移除由 S1/S2 對照代表 | S1 去 amount 已等同 S2 | 不額外製造重複的測試結果 |
| E2 預設只跑 XGBoost | 消融可能移除規則所依賴的特徵 | 不宣稱已完成三模型消融 |
| SHAP 使用 train 背景及 train 中位數遮罩 | 避免 test 擬合參考值 | 遮罩非因果試驗、可能產生分布外樣本 |
| 原始下載資料夾保留，另建立 Git 副本 | 原始資料夾無 .git，直接整包覆蓋會丟掉組員新增內容 | 後續應以 aml-insight-git 為主要工作目錄 |

## 3. 已發生的工作時序

### 3.1 2026-11-07：專案盤點與範圍確認

1. 檢查 aml-insight-main 的 README、pyproject、loader、既有測試及 W1 分析文件。
2. 確認專案當時以 W1 資料讀取與品質分析為主，evaluation、models、explain 等目錄多為空模組。
3. 確認本機 data/raw 尚無原始資料，亦無 A 正式特徵、C 正式模型分數。
4. 讀取既有建模決策：主任務為帳戶節點分類；排除 fraudStep；需要 S1/S2 金額控制。
5. 使用者確認只做 B，隨後核准設計並允許開始實作。
6. 實作方案先寫成設計及計畫，分成指標／切分、輸入／模型、runner、解釋／報告、交付驗證五組。

### 3.2 2026-11-07：實作、除錯與第一次驗證

1. 建立 B 的 Python 3.11 虛擬環境，安裝評估與解釋所需套件。
2. 以手算範例檢查 AP、Top-K、F1、混淆矩陣；補上單類、同分及 K 大於樣本數的行為。
3. 實作五 seeds、帳戶排序、60/20/20 分層切分及比例偏差驗證。
4. 實作 A／C 的輸入契約；拒絕 ID 重複、集合不一致、未分類特徵及明確洩漏欄位。
5. 實作 validation 分位數規則、train-only class weight 與 XGBoost 候選記錄。
6. 實作 tune／finalize 分離、模型與資料快照、source／資料指紋及結果檔完整性。
7. 實作 SHAP、個別 waterfall、重要度與 training-median masking。
8. 實作消融／時間設定產生、五 seeds 平均／樣本 SD／最佳值、pending 報告。
9. 修正本機缺少 libomp 的執行問題：啟動腳本使用 sklearn wheel 內附 runtime，不修改系統安裝。
10. 獨立檢查發現四項問題並修正：C 特徵契約、final 輸出完整性、C 訓練版本分組，以及 K=正樣本數的語意彙整。
11. 第一次交付時的結果為 27 passed、5 skipped、Ruff 通過；並完成十次合成 CLI 驗證。

### 3.3 2026-11-09：上傳小組 GitHub

1. 找到與使用者圖片相符的 Rain931101/aml-insight，確認目前帳號可存取且具 push 權限。
2. 讀取最新 main，發現已包含成員 C 的入門 notebook、子圖及 Streamlit 首頁骨架。
3. 以最新 main 為基底建立獨立副本，只加入 B 的變更，保留 C 與其他既有檔案。
4. GitHub connector 的 Git tree 寫入回傳 403 Resource not accessible by integration；這是連線寫入能力不足，不是程式測試失敗。
5. 命令列推送亦因沒有可用的 HTTPS 登入憑證而未成功；沒有在命令列顯示或擷取 token。
6. 改用使用者已登入的 GitHub 網頁，建立 feat/member-b-evaluation 分支並依既有目錄上傳檔案。
7. 初版共 28 個變更檔案、10 個網頁提交；28 個 blob SHA 全部與本機一致，其他既有檔案內容未遺失或變更。
8. 建立 PR #2，目標 main；沒有自動合併。
9. 建立 aml-insight-git 作為有 Git 紀錄的後續工作副本，保留原 aml-insight-main。

### 3.4 2026-11-09：本次完整複查

1. 重新讀取 PR、檔案清單、review 記錄、main 分支與 CI 狀態。
2. 在新 Git 副本建立獨立 .venv。第一次線上安裝因執行環境 DNS 限制失敗，改用已下載的 uv 快取成功安裝；不是修改套件版本來迴避測試。
3. 基準版本重跑：27 passed、5 skipped、3 warnings，Ruff 通過。
4. 重新進行只讀程式審查，發現 E3 批次產生器對全／早視窗的路徑定義不足。
5. 先新增三項會失敗的回歸案例，確認原版本不能滿足正確的時間比較定義。
6. 修正 suite 及對應設定與文件，再執行全套測試：30 passed、5 skipped，Ruff 通過。
7. 複查該修正，確認 E1 保持全視窗，只有 E3 使用早視窗，缺失／誤用路徑會提早拒絕。
8. 在新副本重新執行十次合成 CLI smoke；SHAP、summary、排除 test run 的預設行為均成功。
9. 產生本日誌與本機驗證證據，延續同一 PR 交付。

## 4. GitHub 基準快照與整合狀態

以下為本次修正前讀取的 GitHub 快照，避免將「已上傳」「已核准」「已合併」混為一談。

| 項目 | 實際快照 |
|---|---|
| PR | #2，open |
| base branch | main |
| base SHA | f22ff043303669b666d43f9b44b4b0574345feee |
| 檢查開始時的 head SHA | cbdaa3b6466b418caeee55d6f2b68f9f65d25c11 |
| 初版 changed files | 28 |
| 初版 commits | 10 |
| 初版 additions / deletions | 2058 / 0 |
| merged | false |
| mergeable | true：程式差異可自動合併，沒有內容衝突 |
| mergeable_state | blocked：尚有 GitHub 合併政策條件未滿足 |
| main protected | true |
| PR reviews | 0 筆；不能稱為已獲組員批准 |
| check runs | 0；不能稱 GitHub CI 已通過 |
| combined commit status | pending、statuses 空陣列；不等於已有失敗測試 |
| repository rulesets 清單 | 空陣列；不能據此推論不存在舊式分支保護 |

隨後在 PR 網頁確認了具體阻擋條件：**至少需要 1 位具有 write 權限的審查者給予 approving review**。頁面明示「At least 1 approving review is required by reviewers with write access.」。因此應請另一位有寫入權限的組員審查並核准，再依小組流程合併。本次未關閉保護、未繞過必要審查。受保護規則阻擋與程式能否自動合併是兩件事。

本次 E3 修正後 head SHA 與檔案數會繼續變動，應以 PR 最新提交為準；本表刻意保留複查開始時的歷史狀態。

## 5. 成員 B 初版 28 個變更檔案清單

| 類別 | 檔案 | 責任與用途 |
|---|---|---|
| 專案 | .gitignore | 排除本機工具快取與可重建合成 smoke 產物 |
| 專案 | README.md | 新增 B 工具入口與正式資料缺口說明，保留原內容 |
| 環境 | requirements-b.lock | 32 個已測試套件的固定版本 |
| 設定 | configs/member_b.example.yaml | S1/S2、規則、grid、外部預測、時間設定範例 |
| 方法 | docs/experiment_protocol.md | E1–E5、RQ、指標、閾值、限制與可追溯方法 |
| 接入 | docs/member_b_handoff.md | A／C 檔案格式、安裝與命令、驗收邊界 |
| 驗證 | reports/member_b/verification.md | 初次交付的 27 項通過驗證紀錄，保留歷史 |
| 安裝 | scripts/setup_b.sh | 建立最小 B 環境，不強迫安裝 C 的 GNN 套件 |
| 驗證 | scripts/check_b.sh | pytest 與 Ruff 的一致入口 |
| 執行 | scripts/run_b.sh | PYTHONPATH、Matplotlib 快取及 Mac OpenMP 啟動環境 |
| smoke | scripts/smoke_b.py | 生成明確 purpose=test 的整合測試資料，跑十個 run |
| smoke | scripts/smoke_b.sh | smoke 的跨路徑啟動包裝 |
| 切分 | src/aml_insight/data/split.py | 固定帳戶 ID manifest、分層比例與互斥檢查 |
| CLI | src/aml_insight/evaluation/__main__.py | split／tune／finalize／explain／aggregate／suite |
| 證據 | src/aml_insight/evaluation/artifacts.py | JSON、SHA256、Git／環境版本及檔案封存檢查 |
| 契約 | src/aml_insight/evaluation/inputs.py | A 特徵驗證、C score 的精確 ID 對齊 |
| 指標 | src/aml_insight/evaluation/metrics.py | AP、ROC-AUC、F1、混淆矩陣、Top-K 與閾值 |
| 報告 | src/aml_insight/evaluation/report.py | PR 圖、錯誤分析、來源清單、分組 seed 統計 |
| 編排 | src/aml_insight/evaluation/runner.py | validation 選模、保存快照、一次 test 與模型接入 |
| 批次 | src/aml_insight/evaluation/suite.py | 產生 E1/E2/E3 × S1/S2 × 五 seeds 的設定 |
| SHAP | src/aml_insight/explain/shap.py | TreeSHAP、加總驗證、重要度、waterfall 與遮罩分析 |
| 規則 | src/aml_insight/models/rule_based.py | validation 尾端分位數規則與觸發比例分數 |
| XGB | src/aml_insight/models/xgb.py | train-only 不平衡權重、固定隨機種子、CPU 訓練 |
| 測試 | tests/test_metrics_split.py | 手算指標、極端輸入、五 seeds 切分 |
| 測試 | tests/test_inputs_models.py | 特徵分類、洩漏、規則分位數與模型權重 |
| 測試 | tests/test_runner.py | tune/final、test 隔離、完整性、C／E2／E3 接入 |
| 測試 | tests/test_report_explain.py | SHAP、忠實度、sample SD、positive-count K 與 C 分組 |
| 測試 | tests/test_suite.py | 批次設定與新增的全／早／晚視窗路徑回歸 |

本日誌是初版之後的新交付文件；不能因此把初版檔案數仍寫成最新 PR 檔案總數。

## 6. 方法與實作行為的逐項複查

### 6.1 指標

- AP 使用 Average Precision，不把梯形 PR 積分當成同一個數字。
- 混淆矩陣固定 labels=[0,1]，順序為 [[TN, FP], [FN, TP]]。
- Top-K 的 K 為 100、500、該 split 正樣本數；超出樣本數時明記 effective K。
- 排名同分採 nodeid 升序，避免檔案列序左右結果。
- 無正樣本時 AP／Recall 回傳 null 並留原因；單類 ROC-AUC 亦為 null。
- validation 最大 F1 決定閾值，同分採最高閾值，降低同分時的警報數。
- 跨 seeds 的「K=正樣本數」以共同語意欄位彙整，不因實際正樣本數不同而被拆成不完整的數字欄。
- 不以 accuracy 作主指標，避免不平衡資料下全部判正常也得到高準確率。

### 6.2 切分與防洩漏

- 依 nodeid 排序後切分，A 的資料列序改變不會改帳戶 membership。
- train/validation/test = 60/20/20，seeds=[0,1,2,3,4]。
- 每份需含兩類，正樣本比例與全體差距必須小於 0.005。
- split 互斥且覆蓋全體，與 C 共用相同 manifest。
- 明確拒絕 fraudStep、標籤、ID 作為模型特徵；但不能只靠聚合表檢出所有「改名後的標籤衍生特徵」。
- 上游補值、標準化或特徵計算是否使用未來資訊，仍需 A 的版本與方法證明。

### 6.3 規則、XGBoost 與公平比較

- Rule-based 各特徵門檻只由 validation 分布決定。
- high 採第 95 百分位；low 採第 5 百分位，等同反向特徵的第 95 百分位。
- 邊界含等號，離散 ties 可能使超過 5% 的帳戶觸發，不能把分位數誤當固定警報比例。
- 規則分數為觸發比例，並不是校準過的犯罪或詐欺機率。
- XGBoost scale_pos_weight 只用 train 的負／正樣本比。
- 每個超參數候選只記錄 validation AP；同分採確定的第一個候選。
- 已測試只擾動 test 特徵時，選出的模型內容、候選結果與閾值不變。
- S2 只依 A 明確指定的 amount 群組移除金額，圖邊或衍生欄位是否仍偷帶金額資訊需組員共同確認。

### 6.4 test 鎖與實驗完整性

- tune 不產生 test metrics。
- finalize 驗證 source／模型／快照／設定指紋，並以排他檔案建立同 run 的測試鎖。
- 同 run 再次 finalize 會拒絕；評估中途失敗也不自動重試。
- final 結果另有 result_integrity.json，修改結果檔後 aggregate 會拒絕。
- SHA 指紋是誤改與可追溯檢查，不是抵抗可同時重寫資料與指紋者的簽章或安全證明。
- 不同 run_id 仍可能指向同一 test；程式鎖不能代替研究者不反覆偷看 test 的規範。

### 6.5 C 的接入

- validation/test 分數各自只含 nodeid、score，且帳戶集合精確符合 manifest。
- 重複、漏列、多列、跨 split 或非有限分數會拒絕。
- metadata 比對 seed、S1/S2、manifest SHA、特徵檔 SHA、selected_columns 及特徵版本。
- E3 額外比對晚視窗特徵檔指紋。
- 五 seeds 彙整時，GraphSAGE 的 training_config 與 feature_version 亦影響分組，避免混合不同 C 模型。
- B 無法從分數檔單獨證明 C 沒有使用 test 標籤或違反 transductive 設定；仍需 C 的訓練紀錄。

### 6.6 SHAP 與忠實度

- TreeExplainer 採 interventional、probability 空間。
- 背景只取 train，解釋抽樣只取 test，兩者帳戶集合互斥。
- 保存 SHAP values、base value、預測分數與原始抽樣特徵，能重建個別解釋。
- 計算 base value + SHAP 加總與模型分數的誤差，門檻為 1e-4。
- 忠實度按每個帳戶絕對 SHAP 值取 top-1／3／5，以 train median 替代，維度不足時截斷。
- 同時保存固定 seed 的隨機特徵遮罩對照。
- 分數下降可以為負值；移除保護性特徵可能提高風險分數。相關特徵、分布外替代及非因果性要寫入報告。

### 6.7 統計與報告

- 分組必須保持資料 bytes、程式指紋、環境、特徵定義與選參策略一致。
- 報平均、樣本 SD（ddof=1）、最佳值、有效 run 數與 seeds。
- 只有一次 run 時 SD 為 null，不以 0 偽裝穩定。
- 同組重複 seed 會拒絕，不挑最好的一次冒充平均。
- 五 seeds 的 test 可能有帳戶重疊；這個 SD 不是獨立外部資料的信賴區間。
- 預設排除 purpose=test，正式無結果時報 pending；原始資料及大型合成產物未當作程式碼上傳。

## 7. 本次發現並修正的 E3 問題

### 7.1 問題與影響

原批次產生器複製同一份 base config 的 features 到 E1 與 E3。可是協定要求比較：

- E1：全視窗特徵，分層隨機 train/validation/test。
- E3：早視窗特徵供 train/validation，晚視窗特徵供 test。

如果 base features 指向早視窗，E1 就不再是全視窗基準；如果指向全視窗，E3 訓練就可能含晚期資訊。這不是單純文字問題，會讓比較定義與防未來資訊的主張不一致。

目前沒有用這個錯誤設定產生正式研究數字，所以沒有需要撤回的正式結果；但需在 A 的時間資料接入前修正。

### 7.2 修正後的契約

| 設定欄位 | 在 suite 的意義 | 用途 |
|---|---|---|
| features | 全時間視窗特徵 | E1 及其 E2 基準 |
| time_train_features | 獨立的早視窗特徵 | 只用來產生 E3 的 features |
| time_test_features | 晚視窗特徵 | E3 test 帳戶使用 |

產生 E3 時，僅 E3 config 的 features 改指早視窗；suite 專用的 time_train_features 不殘留在各單次 run 的設定中。
缺少早視窗欄位或與全視窗為同一路徑，會在建立批次輸出目錄前拒絕。
單獨手寫 E3 config 時，features 直接指定早視窗即可，不一定需要 suite 專用欄位。

路徑檢查不是上游時間正確性的證明。A 仍須確保每份表真正遵守視窗，並提供觀測截止時間與版本。

### 7.3 回歸測試與複查

新增三個案例：

1. 正確提供 full／early／late 時，10 份 E1 保留 full，10 份 E3 使用 early，且 E3 的晚視窗路徑為 late。
2. 只有晚視窗但沒有 time_train_features 時，應拒絕且不留下 suite 目錄。
3. time_train_features 誤指 full 時，應拒絕且不留下 suite 目錄。

修正前實際看到三例失敗；修正後 tests/test_suite.py 為 4 passed，完整測試為 30 passed、5 skipped。只讀複查認定原問題已解決，未發現這項修正新增的重大問題。

另修正操作文件的一處可攜性表述：全新 clone 不會自動包含本機的 reports/member_b/results/results.md，必須先執行 aggregate 才產生。

## 8. 本次實際驗證命令、結果與證據

### 8.1 安裝與環境

```bash
# 第一次線上安裝：遇到執行環境 DNS 限制，未完成。
UV_CACHE_DIR=/tmp/aml-uv-cache bash scripts/setup_b.sh

# 使用既有快取，在新的 .venv 成功安裝並執行 CLI help。
UV_OFFLINE=true UV_CACHE_DIR=/tmp/aml-uv-cache bash scripts/setup_b.sh
```

32 個套件安裝成功，Python 3.11.15。執行機器為 macOS ARM64。
系統 Python 與 .venv 相互隔離：直接使用系統 python3 不一定找得到 pandas，操作時應使用腳本或 .venv/bin/python。
本次確認了快取安裝與目前 Mac 環境，未宣稱已驗證 Windows、Linux、所有 Python 版本或完全無快取的網路安裝。

主要套件版本：numpy 2.4.6、pandas 3.1.1、scikit-learn 1.9.1、xgboost 3.2.0、shap 0.51.0、matplotlib 3.11.2、pyarrow 26.0.1、pyyaml 6.0.3、pytest 9.2.0、ruff 0.17.4。完整版本見本機 environment.json。

### 8.2 pytest、lint、shell 與依賴

```bash
bash scripts/check_b.sh -q
bash -n scripts/setup_b.sh scripts/run_b.sh scripts/check_b.sh scripts/smoke_b.sh
UV_CACHE_DIR=/tmp/aml-uv-cache uv pip check --python .venv/bin/python
git diff --check f22ff043303669b666d43f9b44b4b0574345feee
```

| 驗證 | 結果 |
|---|---|
| 修正前完整 pytest | 27 passed、5 skipped |
| 修正後完整 pytest | **30 passed、5 skipped** |
| Ruff | All checks passed |
| shell 語法 | exit 0 |
| uv pip check | 32 個已安裝套件相依相容 |
| diff 空白檢查 | 無錯誤 |

修正後的 30 個案例可分為：指標／切分 9、輸入／模型 3、runner 9、報告／SHAP 5、suite 4。

**5 項跳過的原因：**既有 tests/test_loader.py 整個模組在原始資料缺席時跳過。包含三個資料集結構案例、fraudStep 檢查與 unknown-dataset 案例。最後一個其實不依賴原始資料，但受到原模組標記一起跳過；本次未修改其他組員的既有測試政策。因此不能把 5 skipped 說成 5 passed。

**3 項警告的原因：**SHAP 使用的 Matplotlib colormap set_bad／set_over／set_under 有 PendingDeprecationWarning。不是本次 assertion 失敗，也未隱藏警告。未來更新 SHAP／Matplotlib 時需再驗證。

### 8.3 合成 CLI smoke

```bash
bash scripts/smoke_b.sh reports/member_b/smoke
```

本次重新產生 1,000 個測試帳戶，其中 100 個正樣本；這是程式自建的驗證資料，**不是 AMLSim 正式資料**。

- S1 五 seeds、S2 五 seeds，共 10 個 tune/finalize。
- 每個設定均有 Rule-based 與 XGBoost test 輸出，形成 20 筆模型結果。
- 每個 seed 的合成資料 split 為 600／200／200。
- 每個 XGBoost 比較兩個小型 grid 候選，僅為快速驗證，不能當正式超參數搜尋範圍。
- S1 seed0、S2 seed0 各取 16 個 test 帳戶及 20 個 train 背景帳戶作 SHAP。
- summary 有四列：S1／S2 × Rule-based／XGBoost，每列 n_runs=5、seeds=0,1,2,3,4、five_seeds_complete=true。
- 每個 run 的 purpose=test；預設研究報告成功排除全部十個 run，輸出「尚無正式研究結果」。

SHAP 加總最大絕對誤差（僅驗證數值正確性）：

| 合成 run | 最大誤差 | 驗證門檻 |
|---|---:|---:|
| S1 seed0 | 約 4.6850e-08 | 小於 1e-4，通過 |
| S2 seed0 | 約 6.2501e-08 | 小於 1e-4，通過 |

本日誌不引用這些合成資料的 AP、F1 作為研究發現；數值品質檢查與研究有效性必須分開。

### 8.4 其他 CLI

```bash
bash scripts/run_b.sh --help
bash scripts/run_b.sh aggregate experiments reports/member_b/results
bash scripts/run_b.sh suite configs/member_b.example.yaml /private/tmp/aml-b-review-suite-20261109
```

CLI 可用；正式 experiments 無完成研究 run 時，aggregate 如實輸出 pending。
基本範例有 amount、behavior、structure、temporal 四群，會產生 E1 10 份，加上三個非金額群組各 S1/S2 × 五 seeds 的 E2 共 30 份，總計 **40 份設定**。範例未提供時間資料時不產生 E3，並在 pending.json 說明；不另產生重跑 E1 的 E5。

### 8.5 本機證據位置

相對於 aml-insight-git：

| 檔案／目錄 | 內容 |
|---|---|
| reports/member_b/review_2026-11-09/tests.txt | 新副本基準版本的測試輸出 |
| reports/member_b/review_2026-11-09/tests_after_fix.txt | E3 修正後完整測試輸出 |
| reports/member_b/review_2026-11-09/smoke.txt | 十次 CLI 驗證的命令輸出 |
| reports/member_b/review_2026-11-09/environment.json | Python、平台與套件版本 |
| reports/member_b/review_2026-11-09/github_snapshot_before_fix.json | 複查開始時的 PR／CI／檔案清單快照 |
| reports/member_b/review_2026-11-09/local_files_after_fix.json | 本機初版檔案清單在修正後的 SHA256／blob SHA |
| reports/member_b/smoke/ | 可重建的合成資料、模型與圖，Git 忽略 |
| reports/member_b/results/results.md | 本次 aggregate 產生的正式 pending 報告 |

含本機路徑的執行證據與大量 smoke 產物留在本機，避免把驗證資料混成正式研究輸出；程式、方法與本日誌供小組檢查。

## 9. W1–W8 驗收對照與誠實的完成邊界

| 週次 | 原要求 | 程式／文件狀態 | 真正完成正式驗收尚需 |
|---|---|---|---|
| W1 | PR-AUC、P@K、R@K、F1、矩陣、ROC；60/20/20；E1–E5 協定 | 已實作、手算及五 seeds 測試通過 | 確認 A 正式欄位、資料版本、E4 定義 |
| W2 | 含／不含金額規則、XGBoost 不平衡處理、統一 runner | 已實作；validation 閾值與候選、seed／Git／source 可記錄 | 真實特徵表，正式調參與 final |
| W3 | validation-only 深度／步長／樹數搜尋，test 只測一次 | 已實作與隔離／重試拒絕測試 | 正式候選搜尋、選定設定與一次 final |
| W4 | S1/S2 三模型比較、PR 圖、共同錯誤分析 | B 兩模型及 C 接口已測試；C 測試分數不是實際訓練模型 | C 正式分數及版本、A 子群 metadata |
| W5 | SHAP、E2、忠實度 | 解釋及遮罩測試通過；消融設定可產生 | 正式群組／資料變體、正式模型與 SHAP 圖 |
| W6 | E3 時間切分與隨機比較 | 路徑問題已修；早／晚快照接入測試通過 | A 全／早／晚三份表及觀測聲明，正式成效比較 |
| W7 | 五 seeds 重跑主要模型；mean±SD、最佳值 | 設定與彙整已驗證 | 五個 seeds 的正式 E1，以及 C 對應輸出 |
| W8 | 實驗與結果章節定稿，每個數字可追溯 | 可產生章節草稿、summary 與來源 | 正式結果、限制討論與小組確認，不能以空報告宣稱定稿 |

## 10. 仍需組員提供的具體資料

### 10.1 A 的交付清單

1. 帳戶特徵表：每列唯一 nodeid、二元 isFraud、有限數值特徵。
2. 所有特徵的明確群組；amount 含起始餘額與所有金額衍生特徵。
3. 特徵版本／commit、資料來源與生成命令；補值／標準化如有擬合，需 train-only。
4. 無小額交易分析 metadata：nodeid、has_small_amount（0/1），不得當 S2 模型特徵。
5. E3 全視窗、早視窗及晚視窗特徵表；相同帳戶／靜態標籤，明確時間界線與觀測截止證據。
6. 若 E2 以不同資料變體而非刪欄實現，需列出各變體意義與可比較的帳戶集合。

### 10.2 C 的交付清單

1. 每個 seed／S1/S2／實驗變體各自對應的 validation.csv、test.csv（或 Parquet）。
2. 每份僅含 nodeid、score；帳戶 ID 精確等於 B manifest。
3. metadata：seed、setting、manifest_sha256、feature_file_sha256、feature_version、selected_columns、training_config。
4. E3 另含 time_feature_file_sha256。
5. 實際訓練設定、圖邊／聚合方式、訓練標籤範圍、訓練 seed、版本與無 test 標籤洩漏的記錄。
6. S2 的圖與特徵也需遵守金額控制，不能只從表格移除 amount、卻在圖邊特徵留下相同捷徑。

已有 C 的 Cora 入門 notebook／子圖／首頁骨架，不代表已具備能接入 E1 的 AML 帳戶 GraphSAGE 正式預測。

## 11. 目前限制與尚可改善項目

| 項目 | 等級／性質 | 本階段處理 |
|---|---|---|
| 正式特徵、時間資料與 C 分數缺席 | 正式實驗阻擋 | 明列 A／C 交付，不捏造結果 |
| PR 未審查與未合併 | 小組流程阻擋 | 保留 PR，交有權限組員審查 |
| 無 GitHub check run | CI 能力缺口 | 保存本機通過證據；建議後續建立最小 B CI，未擅自聲稱已部署 CI |
| E4 未定義 | 規格缺口 | 向小組確認，不自行指定 |
| 原始資料缺席導致 loader 測試跳過 | 資料驗證限制 | 正式資料到齊後重跑並處理真正失敗 |
| SHAP 第三方棄用警告 | 維護事項 | 保留警告與鎖檔，版本更新後再驗證 |
| C training_config 若非 mapping，錯誤較晚出現 | 輸入防呆改善 | 已記錄；正常契約要求 JSON object，不是既有正常路徑阻擋 |
| 時間資料僅以路徑／界線／聲明檢查 | 上游可驗證性限制 | 需交易級來源與版本佐證 |
| E2／E3 結果目前是可比較資料表，不是已完成科學結論 | 研究進度 | 待正式資料後逐項寫結論與限制 |
| 支援平台未做矩陣測試 | 可攜性限制 | 本次實測 macOS ARM64＋Python 3.11.15，不保證其他組合 |
| 原始下載資料夾與 Git 副本並存 | 工作目錄混淆風險 | 後續以 aml-insight-git 為主；舊目錄作歷史備份 |

## 12. 下一階段可直接執行的工作順序

1. 小組審查 PR #2 的方法與接入契約；確認本次 E3 修正也在最新分支。
2. 確認主資料集、S1/S2 特徵群組、規則欄位與 E4 規格。
3. A 提供正式特徵及版本後，將範例 YAML 改成實際檔案與欄位。
4. 先輸出各 seed manifest 給 C，不自行更換 split。
5. C 使用相同資料版本、split 與設定訓練並輸出 validation/test 分數及 metadata。
6. 所有要比較的模型輸入到齊，再 tune；只檢視 validation 選參與閾值。
7. 鎖定比較設定後 finalize，每個 run 僅一次，保存完整 artifacts。
8. 對正式 XGBoost 執行 explain，產生 SHAP 與忠實度；勿將 masking 視為因果結論。
9. 依 A 的資料變體／群組做 E2，依全／早／晚表做 E3。
10. 跑齊 E1 五 seeds，E5 直接彙整這些結果，報平均±sample SD、最佳值及有效數。
11. 分析共同漏判與無小額交易詐欺帳戶，保留失敗案例與資料限制。
12. 產生 results.md，對每個表格數字回查 experiments/<id>/metrics.json，完成小組認可的論文段落。
13. 正式資料到齊後再跑既有 loader 測試；如建立 CI，要記錄實際雲端執行結果。

### 可向組員回報的階段說法

> 成員 B 的評估、規則模型、XGBoost、SHAP 與可重現實驗流程已完成程式階段實作，複查後修正了時間實驗設定的全／早視窗混用問題。最新本機測試為 30 項通過、5 項因原始資料缺席跳過，並完成十次合成流程驗證。程式以 PR #2 交付，正式三模型、消融、時間與穩定性研究仍需 A／C 的正式輸入到齊後執行，目前沒有以合成驗證數字充當論文結果。

## 13. 工作目錄與後續操作

- **主要工作目錄：** aml-insight-git（有 Git 紀錄，連至小組 repository）。
- **歷史目錄：** aml-insight-main（原 ZIP 解壓與早期本機驗證，不作之後的同步主目錄）。
- **分支：** feat/member-b-evaluation。
- **整合入口：** PR #2，target main。
- **重現入口：** scripts/setup_b.sh、scripts/check_b.sh、scripts/run_b.sh、scripts/smoke_b.sh。

後續不要把整個 aml-insight-main 再套進 repository；維持既有目錄結構。不要上傳 .venv、快取、原始資料或大型合成模型／快照。修改研究程式後重建 run，保留舊結果為歷史證據，不改舊指標去配合結論。
