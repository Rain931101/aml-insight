"""Export sealed, existing E1 evidence for a lightweight read-only platform. Never trains."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/member-b-amlsim-e1-20261116"
DEST = ROOT / "app/assets/research"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    DEST.mkdir(parents=True, exist_ok=True)
    sources = []
    for run in sorted(SOURCE.glob("E1-S*-seed*")):
        for seal in ["integrity.json", "result_integrity.json"]:
            for name, expected in json.loads((run / seal).read_text()).items():
                if digest(run / name) != expected:
                    raise ValueError(f"Changed evidence: {run.name}/{name}")
        target = DEST / run.name
        target.mkdir(exist_ok=True)
        files = [
            "metrics.json",
            "predictions.csv",
            "columns.json",
            "manifest.json",
            "provenance.json",
            "selection.json",
            "COMPLETE.json",
        ]
        if (run / "shap").exists():
            for name, expected in json.loads((run / "shap/integrity.json").read_text()).items():
                if digest(run / "shap" / name) != expected:
                    raise ValueError(f"Changed SHAP: {name}")
            (target / "shap").mkdir(exist_ok=True)
            files += [
                f"shap/{n}"
                for n in [
                    "values.csv",
                    "importance.csv",
                    "local_predictions.csv",
                    "explanation.json",
                    "faithfulness.json",
                ]
            ]
        for name in files:
            shutil.copyfile(run / name, target / name)
        sources.append(
            {
                "run": run.name,
                "metrics_sha256": digest(run / "metrics.json"),
                "source": str(run.relative_to(ROOT)),
            }
        )
    if len(sources) != 10:
        raise ValueError("Expected exactly S1/S2 × five seeds")
    raw = ROOT / "data/raw/20K_fanin200cycle200"
    receipt = {
        "dataset": "20K_fanin200cycle200",
        "date": "2026-11-16",
        "feature_version": "16 features; before min_amount addition",
        "scope": "Existing held-out test results; exploratory browsing, no retraining",
        "raw_sha256": {n: digest(raw / n) for n in ["nodes.csv", "transactions.csv"]},
        "sources": sources,
    }
    receipt["files"] = {
        str(f.relative_to(DEST)): digest(f)
        for f in sorted(DEST.rglob("*"))
        if f.is_file() and f.name != "receipt.json"
    }
    (DEST / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Exported {len(sources)} sealed runs; {len(receipt['files'])} assets")


if __name__ == "__main__":
    export()
