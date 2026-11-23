# 外部資料卡：Elliptic v1

- 官方來源：https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- 原論文：https://arxiv.org/abs/1908.02591 （Weber et al., 2019）
- 官方中繼資料：`data/external/elliptic/official_metadata.json`；查得version1、lastUpdated2019-07-31。
- 中繼資料列授權：CC BY-NC-ND 4.0。原始及衍生逐筆資料、模型輸入和外部預測僅存本機ignored目錄。
  本次不重發布資料集；GitHub交付僅含自寫程式、資料卡與彙總研究結果。
- 公開下載網址及zip/CSV雜湊保存在本機receipt；不利用不明鏡像或其他專案的預測結果。
- 本輪目標是Bitcoin交易節點的known illicit/licit二元分類，不是銀行帳戶分類。
- unknown保留為-1，不能當成正常；只用於各當期可見圖上下文。
- 原始166特徵中只使用93個local numeric features；排除識別碼、時間及72個已發布的鄰居聚合特徵。
- train1–29、val30–34、test35–49；每個觀測階段的邊只連接截至該階段的節點。
- 原始公開特徵已有非公開資訊與供應方前處理，我們無法從CSV證明該前處理完全沒有未來資訊；
  再做train-only rescaling並不能消除這個來源限制。
- GNN與表格模型分別在此資料重新訓練，**不是把AMLSim模型直接轉移成功**。
- 不以本結果推定銀行偵測效能、洗錢定罪或全國競賽名次。
