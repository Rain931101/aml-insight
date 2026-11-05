"""AMLSim 20K 樣本資料讀取與基本結構驗證。

資料來源：IBM/AMLSim 官方 repo 內 `sample/*.tgz`（2019-01 產生的舊版輸出格式）。
每個資料集資料夾包含：
    nodes.csv         帳戶（節點）層級：nodeid, isFraud, init_balance, fraudStep
    transactions.csv  交易（邊）層級：sourceNodeId, targetNodeId, value, time
    metadata.txt      資料集摘要（節點數、詐欺節點數、交易數、模式）

注意：本模組只負責「讀取 + 型別 + 結構驗證」，不做任何特徵工程，
也不刪除任何欄位（洩漏欄位的排除在特徵工程階段明確處理）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = ("20K_cycle200", "20K_fanin200", "20K_fanin200cycle200")

NODE_SCHEMA = {
    "nodeid": "int64",
    "isFraud": "int8",
    "init_balance": "float64",
    "fraudStep": "int64",
}
TX_SCHEMA = {
    "sourceNodeId": "int64",
    "targetNodeId": "int64",
    "value": "float64",
    "time": "int64",
}

# 會直接或間接揭露標籤、不得作為模型特徵的欄位
LEAKAGE_COLUMNS = {
    "isFraud": "標籤本身",
    "fraudStep": "只有詐欺帳戶 != -1，等同標籤（且為未來資訊）",
}


class SchemaError(ValueError):
    """資料欄位或型別與預期不符。"""


@dataclass
class AMLSimDataset:
    name: str
    nodes: pd.DataFrame
    transactions: pd.DataFrame
    metadata: str


def _read_csv_strict(path: Path, schema: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。請先執行 scripts/00_fetch_amlsim_samples.sh 解壓資料。"
        )
    df = pd.read_csv(path)
    missing = set(schema) - set(df.columns)
    extra = set(df.columns) - set(schema)
    if missing:
        raise SchemaError(f"{path.name} 缺少欄位：{sorted(missing)}；實際欄位：{list(df.columns)}")
    if extra:
        # 不丟掉未知欄位，只提醒，避免默默遺失資訊
        print(f"[warn] {path.name} 有未預期欄位：{sorted(extra)}")
    if df[list(schema)].isna().any().any():
        nulls = df[list(schema)].isna().sum()
        raise SchemaError(f"{path.name} 有缺值：\n{nulls[nulls > 0]}")
    return df.astype(schema)


def load_dataset(name: str, raw_dir: Path = RAW_DIR) -> AMLSimDataset:
    """讀取單一 AMLSim 樣本資料集。

    >>> ds = load_dataset("20K_fanin200cycle200")
    >>> ds.nodes.shape, ds.transactions.shape
    """
    if name not in DATASETS:
        raise ValueError(f"未知資料集 {name}，可用：{DATASETS}")
    folder = raw_dir / name
    nodes = _read_csv_strict(folder / "nodes.csv", NODE_SCHEMA)
    tx = _read_csv_strict(folder / "transactions.csv", TX_SCHEMA)
    meta_path = folder / "metadata.txt"
    metadata = meta_path.read_text(encoding="utf-8") if meta_path.exists() else ""
    tx = tx.reset_index(names="tx_id")  # 原始檔沒有交易 ID，以列序建立穩定 ID
    return AMLSimDataset(name=name, nodes=nodes, transactions=tx, metadata=metadata)
