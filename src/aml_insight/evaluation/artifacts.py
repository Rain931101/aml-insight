"""JSON, fingerprints and provenance; no implicit git repository assumptions."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def provenance(root):
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
        "source_sha256": source_fingerprint(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "shap",
                "matplotlib",
                "pyarrow",
                "pyyaml",
            ]
        },
    }


def seal(run, files, manifest="integrity.json"):
    write_json(run / manifest, {name: sha256(run / name) for name in files})


def verify(run, check_source=True, manifest="integrity.json"):
    if not (run / manifest).exists():
        raise ValueError(f"Artifact integrity manifest missing: {manifest}")
    for name, digest in read_json(run / manifest).items():
        if not (run / name).exists() or sha256(run / name) != digest:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    if check_source and read_json(run / "provenance.json")["source_sha256"] != source_fingerprint():
        raise ValueError("Source integrity mismatch; create a new run after code changes")
