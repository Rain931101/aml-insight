# 文獻對照與貢獻定位

查核日期：2026-11-23。這是有範圍的文獻盤點，不宣稱系統性回顧或窮盡所有研究。
來源以原論文、作者／出版社頁與官方資料庫為主；不能跨不同資料與切分比較榜單分數。

| 原始研究 | 可核對的核心內容 | 對本專題的要求 | 我們不能宣稱的貢獻 |
|---|---|---|---|
| Chen & Guestrin (2016), [XGBoost](https://arxiv.org/abs/1603.02754) | 可擴充的樹提升系統 | 保留有競爭力的表格基準及完整選參 | 我們發明XGBoost |
| Hamilton et al. (2017), [GraphSAGE](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) | 以鄰居聚合學習可用於未見節點的函數 | 名稱為inductive架構不等於本實驗已驗證inductive；需另設資料邊界 | 使用GraphSAGE就是模型創新 |
| Lundberg & Lee (2017), [SHAP](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | 加法特徵歸因框架 | 保存背景、輸出空間、加總與方向性驗證 | 歸因等同因果證明 |
| Weber et al. (2019), [Elliptic](https://arxiv.org/abs/1908.02591) | Bitcoin交易節點、支付流邊、166個特徵；比較LR/RF/MLP/GCN，原研究中RF有競爭力 | 加入RF/LR；unknown不當負例；分清交易與銀行帳戶任務 | GNN必然優於樹模型 |
| Ying et al. (2019), [GNNExplainer](https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html) | 以遮罩尋找對GNN預測重要的子圖與特徵 | 真正保存解釋遮罩並做擾動測試，不能拿普通子圖充當解釋 | 畫出網絡就有可解釋性 |
| Himmelhuber et al. (2021), [Demystifying GNN Explanations](https://arxiv.org/abs/2111.12984) | 指出合成資料、評估指標與呈現可能誤導 | 公開失敗解釋與遮罩限制 | 一張漂亮圖能證明忠實度 |
| Amara et al. (2022), [GraphFramEx](https://proceedings.mlr.press/v198/amara22a.html) | 依不同使用者需求評估解釋充分性／必要性，方法無全面優勢 | 區分模型預測與真實現象、必要性遮罩與穩定性 | 本專題的簡化遮罩分數等同完整GraphFramEx複現 |
| Jensen et al. (2023), [SynthAML](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539331/) | 以實際銀行資料為基礎的合成警示／交易資料，探討AML評估問題 | 是值得後續取得的銀行型資料；授權與任務需另核對 | 舊AMLSim三個樣本涵蓋真實銀行分布 |
| Altman et al. (2023), [AMLworld](https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models) | 發布較完整的合成金融資料生成與GNN基準 | 區分[IBM/AML-Data](https://github.com/IBM/AML-Data)與本次舊AMLSim；資料授權獨立於程式授權 | 舊AMLSim等同此資料或論文benchmark |
| 2024 [Elliptic2原論文](https://arxiv.org/abs/2404.19109) | 將洗錢分析拓展至子圖任務 | 本專題目前是節點分類，不能用節點AP代表整個洗錢網絡被找到 | 本專題已驗證網絡級犯罪案件偵測 |

本文件以原始標題與識別碼追溯；最終投稿前需依競賽格式整理完整書目。

## 與近年更接近工作的邊界

另查得 *Matched Validation Benchmarking for Cryptocurrency Money Laundering Detection:
Separating Encoder Gains From Graph Features and PPO Alert Control*
（[IEEE記錄11741094](https://ieeexplore.ieee.org/abstract/document/11741094)），以及
[2026年SCITEPRESS紀錄148321](https://www.scitepress.org/publishedPapers/2026/148321/pdf/index.html)。
本次原文端點分別拒絕存取，僅核對到搜尋書目／摘要片段，**不以片段推定完整方法或結果**。
須由學校圖書館取得全文後補入比較；在此之前不能宣稱「首次分離圖特徵與編碼器收益」
或「首次提出預算感知AML」。這項限制保留在最終研究定位。

## 本專題的可守住定位

暫定題目：**排除金額捷徑後的金融異常節點辨識：圖訊息、圖衍生特徵與解釋穩定性的控制實驗**。

這是應用研究與可重現性貢獻，並非新GNN演算法。可主張的成果必須落到：

1. 查明舊AMLSim樣本相依、重複事件、巨大連通分量與金額訊號，提供可重建稽核。
2. 在同切分、同圖架構選參預算下，分離鄰居訊息與圖衍生特徵，公開負面結果。
3. 將SHAP的方向性遮罩、背景敏感性及GNN邊遮罩對照附上原始紀錄。
4. 在不同節點語意的Elliptic做獨立來源**方法複驗**，不聲稱直接跨域模型轉移。
5. 將模型輸出接到可追溯的人工檢視流程；使用效益必須另以真實受試者實測。

## 尚未可用的宣傳說法

不得說「首創」「業界可用」「準確抓出洗錢」「證明降低銀行成本」「保證得獎」。
若沒有新方法或場域證據，網站完整度再高也不能取代這些論證。
