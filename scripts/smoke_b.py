"""Synthetic CLI verification only. Never use these generated labels as research evidence."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=False)
(root / "SYNTHETIC_TEST_ONLY.txt").write_text(
    "Synthetic integration verification only. Not AMLSim research results.\n"
)
rng = np.random.default_rng(42)
n = 1000
y = np.array([0] * 900 + [1] * 100)
features = pd.DataFrame(
    {
        "nodeid": np.arange(n),
        "isFraud": y,
        "tx_count": rng.poisson(5 + 3 * y),
        "degree": rng.poisson(2 + 2 * y),
        "min_amount": rng.uniform(0, 500, n) - 30 * y,
    }
)
features.to_parquet(root / "features.parquet", index=False)
pd.DataFrame(
    {"nodeid": np.arange(n), "has_small_amount": (features.min_amount < 100).astype(int)}
).to_csv(root / "metadata.csv", index=False)
cfg = {
    "features": "features.parquet",
    "output_root": "runs",
    "analysis_metadata": "metadata.csv",
    "feature_spec": {
        "groups": {"amount": ["min_amount"], "behavior": ["tx_count"], "structure": ["degree"]}
    },
    "rules": {
        "S1": [
            {"feature": "min_amount", "direction": "low"},
            {"feature": "tx_count", "direction": "high"},
        ],
        "S2": [
            {"feature": "tx_count", "direction": "high"},
            {"feature": "degree", "direction": "high"},
        ],
    },
    "xgb_grid": {"n_estimators": [5, 10], "max_depth": [2]},
    "purpose": "test",
    "experiment": "E1",
}


def cli(*parts):
    subprocess.run(
        [sys.executable, "-m", "aml_insight.evaluation", *map(str, parts)],
        check=True,
        env=os.environ.copy(),
    )


for setting in ["S1", "S2"]:
    for seed in range(5):
        name = f"synthetic-{setting}-seed{seed}"
        config = dict(cfg, setting=setting, seed=seed, run_id=name)
        path = root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        cli("tune", path)
        cli("finalize", root / "runs" / name)
        if seed == 0:
            cli("explain", root / "runs" / name, "--samples", 16, "--background", 20)
cli("aggregate", root / "runs", root / "summary", "--include-test")
cli("aggregate", root / "runs", root / "research-filter-check")
assert pd.read_csv(root / "summary/summary.csv").five_seeds_complete.all()
assert "尚無正式研究結果" in (root / "research-filter-check/results.md").read_text()
print(f"Synthetic smoke completed: {root}")
