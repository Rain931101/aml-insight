"""Write an evidence-linked report from verified summaries, never retrain."""

import json

import pandas as pd
from core import REPORT


def run():
    receipt = json.loads((REPORT / "study_complete.json").read_text())
    if receipt["model_records"] != 85:
        raise ValueError("Study summary missing")
    df = pd.read_csv(REPORT / "all_metrics.csv")
    mean = df.groupby(["dataset", "variant", "model"]).pr_auc.agg(["mean", "std"])

    def ap(ds, variant, model):
        row = mean.loc[ds, variant, model]
        return f"{row['mean']:.4f} ± {row['std']:.4f}"

    comparison = "\n".join(
        f"| {v} | {ap('AMLSim', v, 'graphsage')} |"
        for v in ["full", "no_edges", "rewire", "drop50", "no_structure", "early_late"]
    )
    tabular = "\n".join(
        f"| {v} | {ap('AMLSim', v, 'xgboost')} |"
        for v in ["full", "no_structure", "no_behavior", "no_temporal", "early_late"]
    )
    external = "\n".join(
        f"| {m} | {ap('Elliptic', 'temporal', m)} |"
        for m in ["logistic", "random_forest", "xgboost", "graphsage"]
    )
    sub = pd.read_csv(REPORT / "small_amount_subgroups.csv")
    hard = sub[(sub.setting == "S2") & (sub.group == "no_small")].groupby("model").recall.mean()
    g = pd.read_csv(REPORT / "gnn_explanations/masking.csv")
    st = pd.read_csv(REPORT / "gnn_explanations/stability.csv")
    dup = (
        pd.read_csv(REPORT / "duplicate_sensitivity/results.csv")
        .groupby("model")[["original_ap", "deduplicated_input_ap", "decision_flips"]]
        .mean()
    )
    env = json.loads((REPORT / "environment.json").read_text())
    text = f"""# 排除金額捷徑後，圖訊息究竟帶來多少改善？

AML Insight 研究補強報告｜2026-11-23｜狀態：可由程式完成之實驗與驗證已完成，網站未部署。

## 摘要

本研究對既有AMLSim帳戶分類補入圖與特徵消融、早晚時間窗轉移、子群與重複事件敏感性，
另在不同來源的Elliptic交易節點資料上做時間切分方法複驗。
新增55個AMLSim runs與5個Elliptic runs，共85組model/run指標，皆保存validation選參與一次性test鎖。

研究顯示：在本輪AMLSim設定中，圖衍生特徵的移除造成較大的AP下降，
移除鄰居訊息的影響相對小；跨晚時間窗的下降更大。外部資料上RF優於本輪GraphSAGE。
GNN解釋有個案遮罩效應，但重要邊的跨種子穩定性有限。
這些結果支持更保守的研究原型定位，**不支持GNN必勝、真實銀行成效或犯罪判定**。

## 1. 資料與設計邊界

主要資料為官方AMLSim commit7338a4b，20,000帳戶、120,558交易、1,804正例（9.02%）。
標籤為帳戶層級isFraud，非逐筆犯罪標籤；fraudStep排除。
三份20K樣本各有19,980節點的巨大無向連通分量，另20個孤立節點；不適合偽稱依連通分量做多份獨立驗證。
cycle與fanin樣本共有5,791個不同的完全相同事件tuple，不能當成獨立外部資料。
混合樣本有1,684個重複事件列與15筆自轉帳。重複不等於錯誤，主分析保留原事件。

本輪GNN共同移除自環、保留GraphSAGE根節點路徑；所有圖條件一致處理。
重接邊以實際訊息圖的無向度數為準，全部20,000節點度數完全保持，2×E次成功swap。
圖特徵固定，不假稱每種圖條件的圖衍生特徵都重算；變體重疊率見graph_checks來源。

舊E1的test已檢視，因此所有新增AMLSim結果均為**探索性後續研究**。
協定與plan在新增訓練／指標前鎖定；自環處理補記亦在訓練前加入。
五個seed重用同一帳戶母體，不能當作五個獨立外部資料集。
原E1封存不覆寫；本輪200epochs不能混入原500epochs的平均。

- 完整稽核：[data_audit.json](data_audit.json)、[data_quality.csv](data_quality.csv)、[sample_overlap.csv](sample_overlap.csv)。
- 事先定義：[protocol.md](../../research/2026-11-23/protocol.md)、[鎖定紀錄](../../research/2026-11-23/protocol_lock.json)。
- 文獻與原創性：[literature_review.md](../../research/2026-11-23/literature_review.md)。

## 2. 可重現的方法

S2使用10個非金額特徵：2行為、3時間、5圖結構。train/val/test帳戶60/20/20，seed0–4。
GraphSAGE兩層mean、dropout.5、train-only signed-log標準化、train標籤loss，
3候選×200epochs，每組只用val AP選候選及epoch。XGBoost固定12候選，train類別權重。
F1門檻僅val選。新Rule-based以train95百分位決定規則值，val只選最終F1門檻；
與舊validation-quantile規則分開報告。

表中的±是五個seed的**樣本標準差ddof=1**，不是信賴區間或顯著性檢定。
不同模型的候選數與訓練成本不同，不宣稱全模型算力完全相等；GNN消融各條件預算一致。
核心訓練器與模型來源一致。run_amlsim有兩種純import排序版本，其餘AST相同，
保留在[source_archive](source_archive/formatting_receipt.json)，沒有依test修改算法。

環境：Python {env["python"]}；torch {env["versions"]["torch"]}、PyG {env["versions"]["torch-geometric"]}、
XGBoost {env["versions"]["xgboost"]}、sklearn {env["versions"]["scikit-learn"]}。
每工作程序單線程，最多2個AMLSim與1個外部工作程序；環境與uv.lock雜湊見[environment.json](environment.json)。

## 3. 圖結構與特徵控制實驗

### 3.1 GraphSAGE

| 條件 | Test AP mean ± SD |
|---|---|
{comparison}

![圖消融](figures/graph_ablation.png)

- full→no_edges的平均差僅約0.0116 AP；這是同架構沒有鄰居訊息的控制，並非另行調參的標準MLP。
- 保持度數但改變關係後約下降0.0157 AP；不能宣稱所有收益都是原始鄰居關係帶來。
- 移除5個圖衍生特徵後約下降0.1694 AP，影響大於移除訊息邊。
- 特徵組數量、相關性與模型交互作用不同，這不是嚴格的貢獻百分比分解。
- 每seed只有一個重接／移邊樣本，沒有分離同split內的多重擾動變異。

### 3.2 XGBoost特徵消融

| 條件 | Test AP mean ± SD |
|---|---|
{tabular}

去時間特徵沒有一致改善或惡化的強證據，不做test後選特徵。
圖邊改變而特徵固定時XGBoost輸入不變，沒有重跑以冒充獨立結果。
逐seed的[paired_graph_differences.csv](paired_graph_differences.csv)公開，不將五個重疊split做獨立樣本t檢定。

## 4. 時間視窗轉移

early_late以1–89的特徵／圖訓練與選參，以90–149的特徵／圖評估test帳戶，
late輸入沿用early train的標準化；不使用late標籤選參。
GraphSAGE AP由本輪full的0.8733降至0.6206；XGBoost由0.8300降至0.5188。

這顯示目前隨機全視窗結果不能代表時間轉移能力。
早晚視窗長度、交易量與模式出現時間均不同，不能把全部差距稱為概念漂移或某個模型的因果失敗。
帳戶集合與靜態標籤相同，**不是未見新帳戶、未來新犯罪發生時間預測**。

## 5. 小額交易子群與固定審查預算

以下使用原2026-11-16的500epochs封存結果；不與本輪200epochs基線混合。
S2中沒有參與小於100交易的正例帳戶，五seed平均recall：

- GraphSAGE：{hard["graphsage"]:.2%}
- XGBoost：{hard["xgboost"]:.2%}
- Rule-based：{hard["rule_based"]:.2%}

![困難子群](figures/hard_subgroups.png)

小額關聯子群的recall高得多。因此「整體AP高」不能等同「難以辨識帳戶也處理得好」。
此分組來自全觀測窗交易，只作既有test的事後子群診斷，不納入S2特徵、不是未來可用的訓練標籤。
子群正例數逐seed公開；不得只報較容易的子群。

[review_budgets.csv](review_budgets.csv)另報1%、5%、10%、12.5%固定清單預算的命中、誤報、recall與每正例檢視量。
這是模型清單的計數效益，**不是已實測的人工作業時間節省**。

## 6. 重複事件敏感性與校準

對固定模型、固定門檻，僅將完全相同tuple保留一列後重建行為／時間輸入；帳戶對圖不變。

| 模型 | 原AP | 去重輸入AP | 平均門檻判定改變數／4,000帳戶 |
|---|---:|---:|---:|
| GraphSAGE | {dup.loc["graphsage", "original_ap"]:.4f} | {dup.loc["graphsage", "deduplicated_input_ap"]:.4f} | {dup.loc["graphsage", "decision_flips"]:.1f} |
| XGBoost | {dup.loc["xgboost", "original_ap"]:.4f} | {dup.loc["xgboost", "deduplicated_input_ap"]:.4f} | {dup.loc["xgboost", "decision_flips"]:.1f} |

這是固定模型的輸入敏感性，不是清理後重訓比較；沒有證據這些重複事件是資料錯誤。
結果不能用來挑選更漂亮的資料版本。

[校準診斷](calibration_diagnostics/summary.csv)另保存Brier與固定10等寬bin ECE，不擬合校準器或調閾值。
AMLSim full GNN的平均ECE約0.1023，XGBoost約0.0924，不能把sigmoid分數直接稱為犯罪機率。
ECE受分箱及稀疏bin影響，不當作完整校準有效性的證明。

## 7. 解釋的方向性、穩定性與失敗案例

### 7.1 XGBoost SHAP

原200個抽樣帳戶中各設定188個預測負例、12個預測正例。
改以「模型原判斷類別的信心」評估，只遮罩支持原判斷的正向貢獻特徵，
與10個同數量隨機特徵遮罩對照；遮罩參考仍是train中位數。
方向性分析顯示原先平均分數上升，部分來自移除正常預測的保護性特徵；
不能單憑舊的signed score drop判定SHAP錯誤。

改變train背景抽樣seed0/1/2後，top5特徵平均Jaccard：S1約0.8047、S2約0.8831。
這只涵蓋現有抽樣與背景設定，不能外推所有帳戶與解釋方法。
詳見[方向性遮罩](shap_audit/directional_summary.csv)與[背景穩定性](shap_audit/stability_summary.csv)。
加總吻合只代表局部歸因重建，不是因果正確性或犯罪證據。

### 7.2 真正的GNNExplainer

使用本輪full seed0鎖定checkpoint。以XGB的TP/FP/FN/TN各選2個距val門檻最近案例，
共8個案例，各3個解釋seed，共24份解釋。案例類別按XGB定義，不冒稱也是GNN的相同類別。

- 兩跳計算子圖與全圖目標logit最大誤差：{g.logit_subgraph_error.max():.2e}。
- 移除top20%無向帳戶對，方向成對移除，與10個隨機同數量控制比較。
- 24份解釋平均超出隨機控制的原預測信心下降：約{g.excess_drop.mean():.4f}。
- top20%重要邊跨解釋seed的平均Jaccard：約{st.top20_pair_jaccard.mean():.4f}。

平均遮罩效應不代表每個案例通過。FP案例11466的top邊移除反而提高原預測信心，
FN案例1015亦未優於隨機控制，須作失敗案例保留。
邊集合穩定性偏低，因此不能宣稱GNN解釋已穩定可靠。
遮罩為模型反應診斷，不是交易關係的因果判定；零權重同分以ID排序，也會影響Jaccard。

[case_selection.json](gnn_explanations/case_selection.json)、[完整masking](gnn_explanations/masking.csv)、
[穩定性](gnn_explanations/stability.csv)、[同帳戶XGB SHAP](gnn_explanations/paired_xgb_shap.csv)。

最初一次解釋因dropout模式錯誤被一致性檢查攔下；其暫存結果已隔離，未納入。
修正wrapper.eval與回歸測試後重新產生解釋，沒有重訓模型或改寫test結果。

## 8. 不同來源的外部方法複驗：Elliptic

官方公開Elliptic v1，交易節點，不是銀行帳戶。只用93個local特徵，排除時間、ID與72個鄰居聚合特徵。
known train 26,381（正例2,871），val3,513（591），test16,670（1,083）；unknown不當負例。
時間1–29／30–34／35–49；GNN圖分別只含截至29／34／49的端點，train未來節點輸入歸零。
稽核發現原圖跨time的訊息邊為0；這仍是觀測期間的離線節點分類，不宣稱即時銀行預測。

所有候選與val門檻鎖定後才計算test。五seed使用固定時間切分，SD反映訓練隨機性，不代表資料不確定性。

| 模型 | Test AP mean ± SD |
|---|---|
{external}

![外部比較](figures/external_comparison.png)

在這個固定local-feature與訓練預算設定，RF與XGBoost優於GraphSAGE。
這反駁「GNN在不同金融圖資料上必然較好」的強說法，不代表所有GNN架構都不適合此資料。
我們在外部資料重新訓練，是**方法複驗**，不是AMLSim模型零樣本轉移成功。
公開local features可能含供應方前處理及非公開資訊，無法從CSV獨立證明原始特徵生成沒有未來資訊；
train-only再標準化不能消除這個來源限制。
來源、授權、任務差異見[external_data_card.md](../../research/2026-11-23/external_data_card.md)。
原始／衍生外部逐筆資料僅存本機ignored路徑，不重發布。

## 9. 可重現性與交付

- 60個新增runs、85組model結果：全部COMPLETE、test鎖、結果檔雜湊核對。
- 所有[all_metrics.csv](all_metrics.csv)行可由[source清單](sources.json)回到metrics.json。
- 保留完整val曲線、模型checkpoint、標準化參數、split及選參；失敗不靜默重試。
- 科學邊界測試包括train-only縮放、外部unknown／時間圖邊界、度數保持、結果鎖與解釋eval模式。
- 全套測試與lint實際結果見[verification.md](verification.md)；測試數量不代表研究有效性自動成立。
- 本地推論計時只涵蓋8個子圖、CPU模型forward，不含載入、網頁、網路或人工檢視，不能當完整平台SLA。

## 10. 目前可以／不能提出的結論

可以提出的結論是：本資料與設定下，圖衍生特徵和鄰居訊息帶來不同幅度的收益；
高隨機切分分數隱藏時間轉移、困難子群及解釋穩定性的問題；不同來源資料的模型排序會改變。
這是有界的實證控制研究與可追溯工具，不是新演算法、真實銀行驗證或首創性證明。

使用者已要求略過真人項目，因此沒有受試結果、專家認可、金融業合作或省時聲明。
尚需校方／競賽規則確認、特定場域資料授權或真正新方法時，不能由這套程式代替。
網站仍是原本線上版本；本次沒有push、開PR或部署。
"""
    (REPORT / "research_report.md").write_text(text)
    print("Evidence-linked research report written")


if __name__ == "__main__":
    run()
