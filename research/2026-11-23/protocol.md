# 研究補強協定 v1（2026-11-23，先登錄再執行）

本檔定義新增探索性研究，不取代 2026-11-16 的 E1 封存。
舊 test 已檢視；所有 AMLSim 補強均標記 exploratory follow-up，不作新的獨立確認。
範圍擴充已由使用者明確要求；不更改 A/C 原有建模程式、不部署網站。

## 可檢驗問題

RQ-A：S2 排除金額後，GraphSAGE 的鄰居訊息是否仍帶來 AP / 固定預算召回增益？
RQ-B：當聚合特徵不變、圖邊重接或移除時，結果如何改變？
RQ-C：早視窗選模後轉至晚視窗，靜態帳戶分類表現如何改變？
RQ-D：方向性、隨機遮罩與重複解釋種子是否支持案例解釋？
RQ-E：在不同資料生成機制與不同節點語意的 Elliptic，方法比較是否仍成立？
這不是聲稱新演算法；目前定位為可重現的控制比較與反捷徑研究。

## 共同設計（在讀取新增結果前鎖定）

- Seeds 0–4；AMLSim 沿用既有 train/validation/test 帳戶 manifest。
- 補強主設定為 S2（10 個非金額特徵）；舊 S1/S2 皆保留不覆寫。
- GraphSAGE：2 層 mean，dropout .5，Adam、weight_decay .0005，CPU 單線程。
- 固定 grid：hidden32/lr.01、hidden64/lr.01、hidden64/lr.005；每候選 200 epochs。
  以 validation AP 選 grid / epoch，完全同分保留先遇到者。
  200 epochs 是新補強的共同計算預算，不能與舊 500 epochs 結果直接當相同基線。
- XGBoost：max_depth 3/5/7 × learning_rate .05/.1 × n_estimators 100/300；train-only 類別權重。
- 補強規則：train 的95百分位決定 in/out tx count、in_degree門檻；val只選F1 cutoff。
  舊 validation-quantile 規則保留，不覆寫或混合兩種版本。
- AP 為主指標；F1 cutoff僅val選定；同報ROC-AUC、P/R@100/500、1%/5%/10%審查預算。
- 五 seeds 報 mean / sample SD；成對差異逐 seed 公開，不將重疊test當獨立樣本。
- 不因新增 test 分數調整 grid、案例選法、訓練預算或補跑更好 seed。
- 每個 run 在評估前建立排他 TEST_STARTED；失敗保留、不自動重試；預測與metrics封存hash。
- 保存實際模型checkpoint、訓練標準化參數、完整候選val曲線與環境版本。

## E2：30 組 GNN 模型（六條件 × 五 seeds）

1. full：原圖與S2特徵（本輪共同基線）。
2. no_edges：完全移除訊息傳遞邊；仍保留兩層 SAGE 根節點線性路徑。
   這是同架構無鄰居控制，不將它誤稱獨立調參的標準 MLP。
3. rewire：在模型實際使用的簡單無向圖進行 2×E 次成功 double-edge swaps，
   每節點無向度數與邊數完全相同；固定原特徵。seed與splitseed配對。
   不宣稱均勻取樣整個圖空間，報原圖邊重疊率。
4. drop50：無向帳戶對隨機保留50%，再轉雙向訊息邊；固定原特徵。
5. no_structure：原訊息圖，移除5個圖衍生特徵，只留2行為+3時間。
6. early_late：early 1–89 train/val特徵與圖，late90–149 test特徵與圖；
   train標準化參數沿用early，test只用晚窗。不是未來新犯罪或新帳戶偵測。

另XGBoost在S2、去structure、去behavior、去temporal、early_late各五seeds；
圖邊變化且特徵固定不重跑XGB，因為輸入完全相同。

## E4：解釋與失敗案例

- 舊XGBoost鎖定模型：既有200帳戶SHAP加總核對；依預測類別評估方向一致的遮罩，
  train中位數參考、10個固定random control種子；不把保護特徵的負下降判作失敗。
- 背景敏感性：相同200帳戶，train100背景，種子0/1/2；top5 Jaccard與排名相關。
- 補強GraphSAGE full seed0：以相同run的XGB TP/FP/FN/TN分類，各類分數距門檻最近者
  （同分nodeid小者），每類最多2例；這是事先定義的邊界案例，不是代表全體。
- GNNExplainer：model explanation、binary raw logit、edge object mask，100epochs，
  解釋隨機種子0/1/2；使用目標兩跳計算子圖，核對目標logit與全圖一致。
- 報移除top20%邊後原預測類別confidence下降、10個隨機同數量對照、top邊Jaccard。
  特徵保持固定；遮罩結果非因果證明。無邊或未涵蓋類別明列，不補造案例。

## 外部方法複驗：Elliptic（不混同AMLSim任務）

- 官方Elliptic公開資料，本機保存、不redistribute；來源與授權另外登錄。
- 節點為Bitcoin交易，不是銀行帳戶；known licit / illicit二元，unknown不當負例。
- 排除ID、time及72個鄰居聚合特徵；使用time後的93個local features。
- 時間1–29 known train、30–34 known validation、35–49 known test。
- GNN train圖只含time<=29端點，val圖<=34，test圖<=49；不跨未來窗口輸入圖訊息。
  未知標籤節點可作各當期圖的未標記上下文，只有known train loss。
- 同上GNN200epochs/3grid、XGB12grid，另RF（300trees、max_depth None/12、balanced）、
  LogisticRegression（C .1/1/10、balanced、train-only標準化）皆val AP選擇。
- seed0–4只代表初始化/抽樣變異，時間切分不變；非五份獨立外部資料。
- 在所有候選選定、val門檻與選參存檔後，才首次評估外部test。若官方檔案無法取得，
  明列取得阻擋，不改用不明鏡像資料冒充。

## 人工／場域項目

準備使用者測試腳本、計時與量表CSV、專家訪談題綱及AI/成員貢獻表；
沒有真實受試者回覆就不填結果。不能由程式或AI代替受試者與專家背書。

## 訓練前資料稽核補記

原始資料存在自轉帳；本輪所有 GNN 條件共同移除自環，保留 GraphSAGE 根節點路徑，避免自環在雙向圖與重接邊中的度數定義歧義。特徵仍依原始事件計算。此補記於任何新增訓練／test結果產生前記錄；原始事件不刪除。
