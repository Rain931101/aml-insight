# Week 1 – 資料檔案檢查報告

自動產生：`scripts/01_inspect_files.py`

AMLSim commit：`7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca`

## 20K_cycle200

### 檔案

- `alert.png` — 15.7 KB
- `cc.png` — 29.0 KB
- `count.png` — 38.2 KB
- `deg.png` — 23.7 KB
- `diameter.png` — 32.3 KB
- `metadata.txt` — 0.1 KB
- `nodes.csv` — 360.5 KB
- `transactions.csv` — 2,541.1 KB
- `wcc.png` — 11.4 KB

### metadata.txt

```
nodes: 20,000 (945 fraud nodes)
transactions: 117,805
patterns: 200 cycles
```

### nodes.csv（20,000 列）

|              | dtype   |   n_null |   n_unique |    min |      max |
|:-------------|:--------|---------:|-----------:|-------:|---------:|
| nodeid       | int64   |        0 |      20000 |   0    | 19999    |
| isFraud      | int8    |        0 |          2 |   0    |     1    |
| init_balance | float64 |        0 |      15638 | 100.04 |   599.92 |
| fraudStep    | int64   |        0 |        119 |  -1    |   149    |

前 5 列：

|   nodeid |   isFraud |   init_balance |   fraudStep |
|---------:|----------:|---------------:|------------:|
|        0 |         0 |         184.44 |          -1 |
|        1 |         0 |         175.8  |          -1 |
|        2 |         0 |         142.06 |          -1 |
|        3 |         0 |         125.89 |          -1 |
|        4 |         0 |         151.13 |          -1 |

### transactions.csv（117,805 列）

|              | dtype   |   n_null |   n_unique |   min |      max |
|:-------------|:--------|---------:|-----------:|------:|---------:|
| sourceNodeId | int64   |        0 |      19980 | 10    | 19999    |
| targetNodeId | int64   |        0 |      19804 | 12    | 19999    |
| value        | float64 |        0 |      13929 | 10.11 |   599.91 |
| time         | int64   |        0 |        149 |  1    |   149    |

前 5 列：

|   tx_id |   sourceNodeId |   targetNodeId |   value |   time |
|--------:|---------------:|---------------:|--------:|-------:|
|       0 |            280 |          15349 |  147.8  |      1 |
|       1 |            249 |           9015 |  104.06 |      1 |
|       2 |            166 |           9955 |  183.77 |      1 |
|       3 |            371 |           9446 |  103.94 |      1 |
|       4 |            276 |          10800 |  128.94 |      1 |

### 結構檢查

- ✅ nodeid 唯一
- ✅ nodeid 為 0..N-1 連續
- ✅ 所有 sourceNodeId 存在於 nodes
- ✅ 所有 targetNodeId 存在於 nodes
- ✅ 交易金額皆 > 0
- 完全重複交易列：1,420
- 自我轉帳（self-loop）：14
- 不同 (sender, receiver) 帳戶對：116,256
- 同一帳戶對多筆交易的對數：446（最多 11 筆）
- 時間欄位 `time`：1 ~ 149，共 149 個不同 step
- 曾出現在交易中的帳戶：19,980 / 20,000


## 20K_fanin200

### 檔案

- `alert.png` — 15.0 KB
- `cc.png` — 27.9 KB
- `count.png` — 39.2 KB
- `deg.png` — 24.0 KB
- `diameter.png` — 31.1 KB
- `metadata.txt` — 0.1 KB
- `nodes.csv` — 360.5 KB
- `transactions.csv` — 2,549.3 KB
- `wcc.png` — 11.4 KB

### metadata.txt

```
nodes: 20,000 (954 fraud nodes)
transactions: 118,250
patterns: 200 fan-in
```

### nodes.csv（20,000 列）

|              | dtype   |   n_null |   n_unique |    min |      max |
|:-------------|:--------|---------:|-----------:|-------:|---------:|
| nodeid       | int64   |        0 |      20000 |   0    | 19999    |
| isFraud      | int8    |        0 |          2 |   0    |     1    |
| init_balance | float64 |        0 |      15638 | 100.04 |   599.92 |
| fraudStep    | int64   |        0 |        126 |  -1    |   149    |

前 5 列：

|   nodeid |   isFraud |   init_balance |   fraudStep |
|---------:|----------:|---------------:|------------:|
|        0 |         0 |         184.44 |          -1 |
|        1 |         0 |         175.8  |          -1 |
|        2 |         0 |         142.06 |          -1 |
|        3 |         0 |         125.89 |          -1 |
|        4 |         0 |         151.13 |          -1 |

### transactions.csv（118,250 列）

|              | dtype   |   n_null |   n_unique |   min |      max |
|:-------------|:--------|---------:|-----------:|------:|---------:|
| sourceNodeId | int64   |        0 |      19980 | 10    | 19999    |
| targetNodeId | int64   |        0 |      19803 | 13    | 19999    |
| value        | float64 |        0 |      14024 |  2.52 |   599.91 |
| time         | int64   |        0 |        149 |  1    |   149    |

前 5 列：

|   tx_id |   sourceNodeId |   targetNodeId |   value |   time |
|--------:|---------------:|---------------:|--------:|-------:|
|       0 |            360 |          18984 |  103.55 |      1 |
|       1 |            269 |           9237 |  178.96 |      1 |
|       2 |            211 |          19987 |  180.89 |      1 |
|       3 |            265 |           5829 |  158.71 |      1 |
|       4 |            322 |           5431 |  143.11 |      1 |

### 結構檢查

- ✅ nodeid 唯一
- ✅ nodeid 為 0..N-1 連續
- ✅ 所有 sourceNodeId 存在於 nodes
- ✅ 所有 targetNodeId 存在於 nodes
- ✅ 交易金額皆 > 0
- 完全重複交易列：254
- 自我轉帳（self-loop）：14
- 不同 (sender, receiver) 帳戶對：116,521
- 同一帳戶對多筆交易的對數：468（最多 6 筆）
- 時間欄位 `time`：1 ~ 149，共 149 個不同 step
- 曾出現在交易中的帳戶：19,980 / 20,000


## 20K_fanin200cycle200

### 檔案

- `alert.png` — 15.7 KB
- `cc.png` — 25.1 KB
- `count.png` — 39.0 KB
- `deg.png` — 23.9 KB
- `diameter.png` — 33.8 KB
- `metadata.txt` — 0.1 KB
- `nodes.csv` — 360.7 KB
- `transactions.csv` — 2,598.1 KB
- `wcc.png` — 11.4 KB

### metadata.txt

```
nodes: 20,000 (1803 fraud nodes)
transactions: 120,558
patterns: 200 cycles and 200 fan-in
```

### nodes.csv（20,000 列）

|              | dtype   |   n_null |   n_unique |    min |      max |
|:-------------|:--------|---------:|-----------:|-------:|---------:|
| nodeid       | int64   |        0 |      20000 |   0    | 19999    |
| isFraud      | int8    |        0 |          2 |   0    |     1    |
| init_balance | float64 |        0 |      15638 | 100.04 |   599.92 |
| fraudStep    | int64   |        0 |        134 |  -1    |   149    |

前 5 列：

|   nodeid |   isFraud |   init_balance |   fraudStep |
|---------:|----------:|---------------:|------------:|
|        0 |         0 |         184.44 |          -1 |
|        1 |         0 |         175.8  |          -1 |
|        2 |         0 |         142.06 |          -1 |
|        3 |         0 |         125.89 |          -1 |
|        4 |         0 |         151.13 |          -1 |

### transactions.csv（120,558 列）

|              | dtype   |   n_null |   n_unique |   min |      max |
|:-------------|:--------|---------:|-----------:|------:|---------:|
| sourceNodeId | int64   |        0 |      19980 | 10    | 19999    |
| targetNodeId | int64   |        0 |      19807 | 13    | 19999    |
| value        | float64 |        0 |      14170 |  2.51 |   599.91 |
| time         | int64   |        0 |        149 |  1    |   149    |

前 5 列：

|   tx_id |   sourceNodeId |   targetNodeId |   value |   time |
|--------:|---------------:|---------------:|--------:|-------:|
|       0 |            216 |          14730 |  163.3  |      1 |
|       1 |            322 |           5431 |  143.11 |      1 |
|       2 |             78 |          19972 |  192.33 |      1 |
|       3 |            248 |          14820 |  101.86 |      1 |
|       4 |            242 |          18672 |  113.33 |      1 |

### 結構檢查

- ✅ nodeid 唯一
- ✅ nodeid 為 0..N-1 連續
- ✅ 所有 sourceNodeId 存在於 nodes
- ✅ 所有 targetNodeId 存在於 nodes
- ✅ 交易金額皆 > 0
- 完全重複交易列：1,684
- 自我轉帳（self-loop）：15
- 不同 (sender, receiver) 帳戶對：117,341
- 同一帳戶對多筆交易的對數：905（最多 11 筆）
- 時間欄位 `time`：1 ~ 149，共 149 個不同 step
- 曾出現在交易中的帳戶：19,980 / 20,000

