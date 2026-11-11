# 資料字典（Data Dictionary）— AMLSim 20K 樣本資料

> 依據實際檔案檢查結果撰寫（`scripts/01_inspect_files.py`、`scripts/02_label_quality_analysis.py`，2026-11-05）。
> 來源：IBM/AMLSim 官方 repo `sample/*.tgz`，commit 記錄於 `data/raw/AMLSIM_COMMIT.txt`。
> 這是 2019-01 產生的**舊版輸出格式**，與 AMLSim 目前 `conf.json` 描述的 `accounts.csv / transactions.csv / alert_transactions.csv` 格式**不同**，不可混用文件說明。

## 1. 資料集一覽

| 資料集 | 帳戶數 | 詐欺帳戶 | 比例 | 交易數 | 洗錢模式 | 本專題用途 |
|---|---:|---:|---:|---:|---|---|
| `20K_fanin200cycle200` | 20,000 | 1,804 | 9.02% | 120,558 | 200 cycle + 200 fan-in | **主要資料集** |
| `20K_cycle200` | 20,000 | 945 | 4.73% | 117,805 | 200 cycle | 模式別補充分析 |
| `20K_fanin200` | 20,000 | 954 | 4.77% | 118,250 | 200 fan-in | 模式別補充分析 |

- 三個資料集由同一設定產生，`init_balance` 逐列 100% 相同，**不是獨立樣本**，不能當作「跨資料集泛化」證據。
- `20K_fanin200cycle200` 的 metadata 寫 1,803 個詐欺帳戶，實際 `nodes.csv` 為 1,804；以檔案為準。

## 2. `nodes.csv`（帳戶 / 節點層級）

| 欄位 | 型別 | 意義 | 可作特徵？ | 說明 |
|---|---|---|---|---|
| `nodeid` | int64 | 帳戶 ID，0..19999 連續且唯一 | ❌（僅識別） | 圖節點索引 |
| `isFraud` | int8 (0/1) | 是否為洗錢模式成員 | ❌ **標籤** | 本專題的預測目標 |
| `init_balance` | float64 | 模擬起始餘額 | ✅（S1 amount 群組） | 單一欄位 AUC ≈ 0.51；依協定 S2 也排除此餘額特徵 |
| `fraudStep` | int64 | 詐欺行為發生的 step；-1 = 無 | ❌ **洩漏** | 非詐欺帳戶全為 -1；≠ -1 時 100% 為詐欺。約 50% 詐欺帳戶也是 -1 |

## 3. `transactions.csv`（交易 / 邊層級）

| 欄位 | 型別 | 意義 | 可作特徵？ | 說明 |
|---|---|---|---|---|
| （無）→ `tx_id` | int64 | 由 loader 以列序產生 | ❌（僅識別） | 原始檔沒有交易 ID |
| `sourceNodeId` | int64 | 轉出帳戶 | 結構 | 邊的起點 |
| `targetNodeId` | int64 | 轉入帳戶 | 結構 | 邊的終點 |
| `value` | float64 | 交易金額 | ✅（需注意捷徑） | 範圍 2.51 ~ 599.91 |
| `time` | int64 | 模擬 step（1 ~ 149） | ✅（只能用切分時點之前） | 沒有對應真實日期；不是秒或日 |

**這個版本沒有的欄位**（不得自行捏造）：交易類型、交易層級標籤、alert ID、pattern ID、客戶資料、國別、分行。

## 3.1 衍生帳戶特徵（`src/aml_insight/features/account.py`）

| 特徵 | 群組 | 定義 | S1 | S2 |
|---|---|---|:---:|:---:|
| `init_balance` | amount | nodes.csv 的模擬起始餘額；S2 依協定排除餘額訊號 | ✓ | — |
| `in_amount_sum` / `out_amount_sum` | amount | 轉入 / 轉出金額總和 | ✓ | — |
| `in_amount_mean` / `out_amount_mean` | amount | 轉入 / 轉出事件平均金額 | ✓ | — |
| `net_flow` | amount | 轉入金額總和減轉出金額總和 | ✓ | — |
| `in_tx_count` / `out_tx_count` | behavior | 符合時間窗的轉入 / 轉出事件列數；重複列仍逐筆計算 | ✓ | ✓ |
| `active_step_count` | temporal | 帳戶作為任一端點的不同 step 數 | ✓ | ✓ |
| `mean_interarrival_steps` / `median_interarrival_steps` | temporal | 帳戶作為任一端點之排序事件 step 差值平均 / 中位數；不足兩筆為 0 | ✓ | ✓ |
| `in_degree` / `out_degree` | structure | 不同轉入來源 / 轉出目標帳戶數 | ✓ | ✓ |
| `neighbor_count` | structure | 入/出鄰居聯集數，自身不計入 | ✓ | ✓ |
| `local_density` | structure | 無向投影中直接鄰居誘導子圖的密度 | ✓ | ✓ |
| `two_hop_neighbor_count` | structure | 距離恰為 2 的無向鄰居數（排除自身與 1-hop） | ✓ | ✓ |

若提供 `min_step` / `max_step`，特徵與圖只使用含端點的時間窗 `min_step <= time <= max_step`；E3 輸出 step 1–89 與 90–149 兩張獨立特徵表。`nodeid` 用於對齊，不作模型輸入；`isFraud` 另存為目標，`fraudStep` 永不輸出為特徵。完整群組清單由 `FEATURE_GROUPS` 定義，建置流程另寫入 `data/processed/feature_groups.json` 供 B 的設定對照。

## 4. 資料品質與前處理決策（主要資料集）

- 無缺值；所有交易端點皆存在於 `nodes.csv`；金額皆 > 0。
- 完全重複的交易列 1,684 筆（約 1.4%，同 sender/receiver/金額/step），同一帳戶對可有多筆交易 → 清理模組保留原始事件列；建圖聚合成一條有向帳戶對邊，並保留 `transaction_count`、金額總和/平均、最早/最晚 step，因此交易 multiplicity 不會遺失。
- self-loop 15 筆：預設保留為有效事件，因原始記錄本身未標示為錯誤。其對帳戶行為計數有貢獻，圖中以 self-edge 保留；鄰居集合不把自身算作鄰居。若後續實驗要排除，`clean_dataset(..., drop_self_loops=True)` 可重現此選項並須另記設定。
- 20 個帳戶完全沒有交易：保留為孤立節點，節點數固定 20,000，行為計數/度數為 0；不捏造邊或刪除帳戶。
- 清理與品質摘要由 `src/aml_insight/data/clean.py` 計算；正式特徵由 `src/aml_insight/features/account.py` 產生。原始列順序形成的 `tx_id` 只作識別欄位，不作特徵。

## 5. 重要發現與限制

1. **標籤只在帳戶層級**。交易層級只能用「兩端皆為詐欺帳戶」推導弱標籤，但詐欺帳戶之間也有大量一般金額交易，弱標籤不等於洗錢交易。
2. **金額捷徑**：所有金額 < 100.04 的交易都發生在兩個詐欺帳戶之間（100%）。規則「帳戶有任一筆 < 100 的交易」即可達 precision 100%、recall 約 68%。這是模擬器的產物，不代表真實世界規律。
3. **約 1/3 詐欺帳戶沒有任何小額交易**，且其 `fraudStep` 全為 -1——這些帳戶可能被指派為模式成員但未執行洗錢交易，是模型難以辨識（甚至不可辨識）的正樣本，需在錯誤分析中單獨討論。
4. 沒有 pattern ID，群組層級只能用連通元件近似；「兩端皆詐欺」的交易把 1,799 個詐欺帳戶連成一個大元件，無法還原 400 個 pattern。
5. 合成資料：任何結論都只適用於此模擬設定。
