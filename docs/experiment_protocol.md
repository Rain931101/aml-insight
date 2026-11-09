# 成員 B 實驗協定（E1–E5）

版本：2026-11-07。實作範圍依使用者確認，只包含成員 B。
本協定與程式是交付成果；尚未取得 A／C 正式輸入，沒有正式研究結果。

## 任務與研究問題

帳戶節點二元分類，標籤 `isFraud`。不以交易弱標籤代替帳戶真實標籤。
主要資料集沿用 `20K_fanin200cycle200`；其他 20K 樣本非獨立資料，不作泛化證據。

| 實驗 | RQ／假設（待驗證） | 自變數 | 應變數 | 固定控制與限制 |
|---|---|---|---|---|
| E1 | RQ1：圖資訊是否優於規則及表格模型？ | Rule-based／XGBoost／C 的 GraphSAGE，S1／S2 | AP、Top-K、F1、ROC-AUC | 相同帳戶切分；C 未提供時只報兩模型，三模型比較 pending |
| E2 | RQ1：哪些特徵群組貢獻辨識力？ | A 定義的 behavior／structure／temporal 群組移除 | 同上及相對完整設定差異 | XGBoost、相同 seed；每個變體使用同一預定 grid 選參；S1 去金額即 S2，不重跑 |
| E3 | RQ3：時間視窗轉移是否改變效能？ | 隨機全視窗 vs 早視窗 train/validation、晚視窗 test | 同上 | 相同帳戶 manifest；A 提供無未來資訊特徵；靜態標籤不能證明未來新詐欺預測 |
| E4 | 原圖未定義 | 未定義 | 未定義 | 不自行發明，pending 規格 |
| E5 | RQ3：主要模型在切分變動下是否穩定？ | seeds 0、1、2、3、4 | 平均、樣本 SD、最佳值 | 直接彙整 E1 已鎖定的五次結果，不額外重測同一 seed |

## 分層切分

以帳戶 ID 排序後依標籤做 60/20/20，固定 seeds `[0,1,2,3,4]`。
保存 `manifest.json`，各 split 互斥且覆蓋全體；每份皆需含兩類、正樣本比例與全體偏差 < 0.005（0.5 個百分點），無法滿足即報錯。
A 的輸入列序變動不改變 split。C 採原計畫 transductive 全圖可見、只有 train 標籤用於 loss；這項訓練約束由 C 的 training_config 留存，B 無法從分數檔驗證訓練是否合規。

## 特徵與防洩漏

- `nodeid` 僅對齊；`isFraud` 僅標籤；`fraudStep` 與 `tx_id` 不可作模型特徵。
- A 必須明確分類所有特徵。S1 所有合法特徵；S2 移除 amount 群組（含餘額與一切金額衍生值）。不憑名稱猜測。
- 上游不得使用標籤衍生、全資料 fit 的標準化／補值等洩漏資訊。B 拒絕 NaN、無窮值，A 若補值需 train-only 擬合並記錄。
- 時間切分不得使用視窗結束後的交易或圖邊。B 驗證時間界線與上游聲明，無法只從聚合表證明上游無洩漏；此限制必須保留在結論。
- `has_small_amount` 為分析 metadata，不納入 S2 特徵。可由 A 依合法交易觀測窗口提供；界線沿用專案 <100 的小額定義。

## 選模與 test 鎖定

`tune` 使用 train 訓練 XGBoost；`scale_pos_weight = train_negative / train_positive`。
只以 validation AP 選超參數，保存所有候選。完全同分時採 ParameterGrid 順序的第一個。
不以 test 指標回選模型；不在 validation 或 test 重新訓練最終模型。

Rule-based 採 config 的 high／low 尾端規則，各特徵門檻只由 validation 分布決定。
high 使用第 95 百分位、low 使用第 5 百分位（等同反向特徵第 95 百分位）。邊界含等號；離散特徵可能因大量 ties 觸發超過 5% 的帳戶。
分數為觸發規則比例，並非校準機率。S1／S2 必須各有明確規則，E2 可停用不適用的規則模型。

每個模型的最終二元閾值只由 validation 最大 F1 決定，F1 同分取最高閾值。
調參與選閾值共用 validation 會有選擇樂觀偏差，正式 test 為獨立保留資料。

`finalize` 驗證 source 與調參 artifacts 指紋後，以排他檔案建立 test 評估鎖。
相同 run 不允許再次 finalize；失敗亦保留鎖，需人工檢視原因，不自動重試測試集。
改 config 必須建立新 run，但這不能把反覆查看相同 test 當作新的獨立證據。
輸出 metrics、預測、PR 圖、錯誤案例後另保存結果指紋；彙整拒絕結果檔變更與重複 seed。

## 時間實驗批次設定的三份輸入

`suite` 的 base config 中，`features` 固定代表全視窗特徵，用於 E1 隨機切分基準。
若要產生 E3，必須另外提供 `time_train_features`（早視窗）與 `time_test_features`（晚視窗）。
產生的 E3 config 會把自己的 `features` 指向早視窗，E1 仍使用全視窗，避免同一路徑被誤當成兩種觀測窗口。
缺少早視窗路徑或與全視窗路徑相同時，在建立批次輸出目錄之前拒絕產生設定。
單獨手寫 E3 config 時，`features` 就直接指定早視窗，不需要 suite 專用欄位。
路徑不同不代表上游計算一定正確，仍需 A 提供交易截止時間與版本證據。

## 指標定義

- PR-AUC 使用 sklearn Average Precision（AP），不是 trapezoidal PR AUC。
- Precision@K = top-K 正樣本 / 有效 K；Recall@K = top-K 正樣本 / 全部正樣本。
- K = 100、500、該 split 正樣本數；超過總數時截至樣本數並保存 requested/effective K。
- 同分依 nodeid 升序，避免輸入列序影響排名；「K=正樣本數」跨 seeds 用共同語意名稱彙整。
- F1 以 validation 鎖定閾值計算；混淆矩陣順序 `[[TN,FP],[FN,TP]]`。
- 無正樣本 AP／Recall 為 null；單類 ROC-AUC 為 null，另存原因；無預測正樣本 F1 採 0。
- 不採 accuracy 作主指標：類別不平衡時全部預測正常仍可很高。

## SHAP 與忠實度

僅解釋鎖定 XGBoost。TreeExplainer 採 interventional、probability 空間，背景樣本只取 train。
seed 固定，預設最多 100 背景、200 個 test 解釋帳戶。報告只代表該抽樣，不假稱整份 test。
保存 mean absolute SHAP、summary plot、第一個抽樣帳戶 waterfall、每個帳戶 SHAP 值／base value／分數及原始特徵。
驗證 base value + SHAP 加總與模型分數誤差 <1e-4。

忠實度以每個帳戶絕對 SHAP 排名，移除 top-1／3／5（超出維度截斷），以 train 特徵中位數替代，不重訓。
保存每帳戶分數下降、平均下降及固定 seed 的隨機特徵替代對照。負值可能表示移除了保護性特徵；相關性與分布外替代會影響結果，不能當作因果證明。

## 錯誤分析與論文報告

保存共同漏判正樣本 ID、各模型分數與預測；未有 C 時只稱兩模型共同漏判。
若有 A 的 metadata，另報「無小額交易」正樣本數與 recall；沒有該輸入時標 pending。

`aggregate` 僅納入 COMPLETE 且指紋通過的研究 run，預設排除 purpose=test。
相同資料 bytes、程式、版本、特徵定義與選參策略才合併；C 的模型另比對訓練設定與特徵版本。
報平均、樣本標準差 ddof=1、最佳值與有效 run 數，單次 SD 為 null，不以 0 冒充穩定。
未滿 5 seeds 標 pending。保留 `summary.csv`、`sources.json`、`results.md`，每列數字可追至 run/metrics.json。
最佳值僅補充，不能只選最好一次。不同 seed 的 test 可能重疊，SD 不是獨立外部資料的不確定性或信賴區間。

## 限制

合成資料具有金額捷徑、靜態帳戶標籤、缺乏交易與 pattern 真實標籤；未見行為的正樣本可能難以辨識。
結果不能推論真實銀行成效或犯罪事實。沒有 A／C 輸入時僅交付流程與驗證紀錄，不寫不存在的實驗數字。
