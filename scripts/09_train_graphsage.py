"""成員 C：在 AMLSim 上訓練 GraphSAGE，輸出 B 評估流程可直接接入的分數檔。

用法：
    uv run python scripts/09_train_graphsage.py                       # S1、S2 × seeds 0–4
    uv run python scripts/09_train_graphsage.py --setting S2 --seed 0 # 只跑一組

前置：先執行 scripts/03_build_features_graph.py（成員 A）產生 data/processed/。

每組輸出到 data/processed/C/E1-<setting>-seed<seed>/：
    manifest.json   帳戶切分（與 B 的 tune 以相同函式、相同 seed 產生，位元組相同）
    validation.csv  nodeid,score（validation 帳戶）
    test.csv        nodeid,score（test 帳戶；這裡不計算任何 test 指標）
    metadata.json   B 的 config 中 graphsage.metadata 要求的欄位
    selection.json  每個候選超參數的 validation AP 與學習曲線

研究規則：
- 特徵、帳戶切分、欄位順序全部沿用 B 的 load_features / stratified_split，確保公平比較。
- 只用 validation AP 選超參數與 epoch；test 指標一律交給 B 的 finalize 計算一次。
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # 避免 torch 與其他 OpenMP 函式庫衝突

import argparse
import json
import subprocess
import time

import pandas as pd
import torch

from aml_insight.data.loader import PROJECT_ROOT
from aml_insight.data.split import SEEDS, stratified_split
from aml_insight.evaluation.artifacts import sha256, write_json
from aml_insight.evaluation.inputs import load_features
from aml_insight.models.graphsage import (
    build_edge_index,
    predict_proba,
    preprocess,
    train_and_select,
)

PROCESSED = PROJECT_ROOT / "data" / "processed"
FEATURES = PROCESSED / "features.parquet"
EDGES = PROCESSED / "edges.parquet"
GROUPS = PROCESSED / "feature_groups.json"
OUT_ROOT = PROCESSED / "C"
REPORT = PROJECT_ROOT / "reports" / "week3" / "C_graphsage_validation.json"

GRID = [
    {"hidden": 32, "lr": 0.01},
    {"hidden": 64, "lr": 0.01},
    {"hidden": 64, "lr": 0.005},
]
EPOCHS = 500
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5


def git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def run(setting: str, seed: int, direction: str, epochs: int) -> dict:
    groups = json.loads(GROUPS.read_text())
    df, columns = load_features(FEATURES, {"groups": groups}, setting)
    splits = stratified_split(df.index.to_numpy(), df.isFraud.to_numpy(), seed)

    out = OUT_ROOT / f"E1-{setting}-seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", splits)

    x = torch.tensor(preprocess(df[columns], splits["train"]))
    y = torch.tensor(df.isFraud.to_numpy(), dtype=torch.long)
    edge_index = build_edge_index(pd.read_parquet(EDGES), df.index, direction)
    pos = pd.Series(range(len(df)), index=df.index)
    idx = {k: torch.tensor(pos.loc[v].to_numpy()) for k, v in splits.items()}

    start = time.time()
    best, records = train_and_select(
        x, edge_index, y, idx["train"], idx["validation"], grid=GRID, epochs=epochs,
        seed=seed, weight_decay=WEIGHT_DECAY, dropout=DROPOUT,
    )
    scores = predict_proba(best["model"], x, edge_index)
    for part in ("validation", "test"):
        pd.DataFrame({"nodeid": splits[part], "score": scores[idx[part].numpy()]}).to_csv(
            out / f"{part}.csv", index=False
        )

    build = json.loads((PROCESSED / "build_summary.json").read_text())
    metadata = {
        "seed": seed,
        "setting": setting,
        "manifest_sha256": sha256(out / "manifest.json"),
        "feature_file_sha256": sha256(FEATURES),
        "feature_version": build.get("code_commit") or "unknown",
        "selected_columns": columns,
        "training_config": {
            "architecture": "GraphSAGE",
            "layers": 2,
            "aggr": "mean",
            "edge_direction": direction,
            "edges_file_sha256": sha256(EDGES),
            "edge_attr_used": False,
            "preprocess": "signed log1p + train-only standardization",
            "loss": "BCEWithLogits, pos_weight = train negatives / train positives",
            "optimizer": "Adam",
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "epochs": epochs,
            "grid": GRID,
            "selection": "validation AP over grid x epochs; ties keep earlier",
            "selected_params": best["params"],
            "selected_epoch": best["best_epoch"],
            "train_labels_only": True,
            "transductive": True,
            "code_commit": git_head(),
        },
    }
    write_json(out / "metadata.json", metadata)
    write_json(out / "selection.json", {"candidates": records, "best_params": best["params"],
                                        "best_epoch": best["best_epoch"],
                                        "validation_ap": best["validation_ap"]})
    summary = {"setting": setting, "seed": seed, "n_features": len(columns),
               "best_params": best["params"], "best_epoch": best["best_epoch"],
               "validation_ap": round(best["validation_ap"], 4),
               "seconds": round(time.time() - start, 1)}
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["S1", "S2"], action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--direction", choices=["both", "in"], default="both")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    for path in (FEATURES, EDGES, GROUPS):
        if not path.exists():
            raise SystemExit(f"找不到 {path}，請先執行 scripts/03_build_features_graph.py")
    torch.set_num_threads(1)

    results = [run(s, seed, args.direction, args.epochs)
               for s in (args.setting or ["S1", "S2"]) for seed in (args.seed or SEEDS)]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "note": "Validation AP only (used for model selection). Test metrics are computed once "
                "by member B's finalize, not here.",
        "runs": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
