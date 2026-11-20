"""下載 AMLSim 20K 樣本（給雲端部署用，不需要 git clone 整個 AMLSim repo）。

與 scripts/00_fetch_amlsim_samples.sh 取得的是同一份檔案：固定在同一個 AMLSim commit，
確保雲端與本機看到的資料完全相同。只下載 3 個約 1.3 MB 的壓縮檔。
"""

from __future__ import annotations

import tarfile
import tempfile
import urllib.request
from pathlib import Path

from aml_insight.data.loader import DATASETS, RAW_DIR

AMLSIM_COMMIT = "7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca"
URL = "https://raw.githubusercontent.com/IBM/AMLSim/{commit}/sample/{name}.tgz"


def ensure_samples(raw_dir: Path = RAW_DIR, datasets=DATASETS, timeout: int = 60) -> list[str]:
    """缺少的資料集才下載並解壓；回傳這次新下載的資料集名稱。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name in datasets:
        if (raw_dir / name / "nodes.csv").exists():
            continue
        url = URL.format(commit=AMLSIM_COMMIT, name=name)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / f"{name}.tgz"
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                archive.write_bytes(resp.read())
            with tarfile.open(archive) as tar:
                tar.extractall(raw_dir, filter="data")  # 擋掉路徑跳脫等不安全項目
        fetched.append(name)
    commit_file = raw_dir / "AMLSIM_COMMIT.txt"
    if not commit_file.exists():
        commit_file.write_text(AMLSIM_COMMIT + "\n")
    return fetched
