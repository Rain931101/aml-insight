"""Week 1 – 步驟 5：資料檔案檢查。

檢查每個資料集的：檔案、筆數、欄位、型別、缺值、重複列、數值範圍、
時間欄位範圍、ID 對應關係（交易端點是否都存在於 nodes）。

用法：
    uv run python scripts/01_inspect_files.py
輸出：
    reports/week1/01_file_inspection.md
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from aml_insight.data.loader import DATASETS, PROJECT_ROOT, RAW_DIR, load_dataset

OUT = PROJECT_ROOT / "reports" / "week1" / "01_file_inspection.md"


def describe_frame(df: pd.DataFrame) -> str:
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_null": df.isna().sum(),
            "n_unique": df.nunique(),
            "min": df.min(numeric_only=True),
            "max": df.max(numeric_only=True),
        }
    )
    return info.to_markdown()


def inspect(name: str) -> str:
    buf = io.StringIO()
    w = lambda s="": buf.write(s + "\n")

    folder = RAW_DIR / name
    ds = load_dataset(name)
    nodes, tx = ds.nodes, ds.transactions

    w(f"## {name}\n")
    w("### 檔案\n")
    for p in sorted(folder.iterdir()):
        w(f"- `{p.name}` — {p.stat().st_size / 1024:,.1f} KB")
    w(f"\n### metadata.txt\n\n```\n{ds.metadata.strip()}\n```\n")

    w(f"### nodes.csv（{len(nodes):,} 列）\n")
    w(describe_frame(nodes))
    w(f"\n前 5 列：\n\n{nodes.head().to_markdown(index=False)}\n")

    w(f"### transactions.csv（{len(tx):,} 列）\n")
    w(describe_frame(tx.drop(columns="tx_id")))
    w(f"\n前 5 列：\n\n{tx.head().to_markdown(index=False)}\n")

    # ---- 結構一致性 ----
    node_ids = set(nodes["nodeid"])
    checks = {
        "nodeid 唯一": nodes["nodeid"].is_unique,
        "nodeid 為 0..N-1 連續": set(range(len(nodes))) == node_ids,
        "所有 sourceNodeId 存在於 nodes": tx["sourceNodeId"].isin(node_ids).all(),
        "所有 targetNodeId 存在於 nodes": tx["targetNodeId"].isin(node_ids).all(),
        "交易金額皆 > 0": (tx["value"] > 0).all(),
    }
    w("### 結構檢查\n")
    for k, v in checks.items():
        w(f"- {'✅' if v else '❌'} {k}")

    n_dup_rows = tx.duplicated(subset=["sourceNodeId", "targetNodeId", "value", "time"]).sum()
    n_self = (tx["sourceNodeId"] == tx["targetNodeId"]).sum()
    pair_counts = tx.groupby(["sourceNodeId", "targetNodeId"]).size()
    w(f"- 完全重複交易列：{n_dup_rows:,}")
    w(f"- 自我轉帳（self-loop）：{n_self:,}")
    w(f"- 不同 (sender, receiver) 帳戶對：{len(pair_counts):,}")
    w(f"- 同一帳戶對多筆交易的對數：{(pair_counts > 1).sum():,}"
      f"（最多 {pair_counts.max()} 筆）")
    w(f"- 時間欄位 `time`：{tx['time'].min()} ~ {tx['time'].max()}，"
      f"共 {tx['time'].nunique()} 個不同 step")
    active = set(tx["sourceNodeId"]) | set(tx["targetNodeId"])
    w(f"- 曾出現在交易中的帳戶：{len(active):,} / {len(nodes):,}\n")
    return buf.getvalue()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# Week 1 – 資料檔案檢查報告\n", "自動產生：`scripts/01_inspect_files.py`\n"]
    commit = RAW_DIR / "AMLSIM_COMMIT.txt"
    if commit.exists():
        parts.append(f"AMLSim commit：`{commit.read_text().strip()}`\n")
    for name in DATASETS:
        parts.append(inspect(name))
    text = "\n".join(parts)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[saved] {Path(OUT).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
