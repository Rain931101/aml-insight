#!/usr/bin/env bash
# 取得 IBM AMLSim 官方 repo 附帶的 20K 樣本資料並解壓到 data/raw/
# 用法：bash scripts/00_fetch_amlsim_samples.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${ROOT}/data/external/AMLSim"
RAW="${ROOT}/data/raw"

mkdir -p "${RAW}" "$(dirname "${CACHE}")"

if [ ! -d "${CACHE}/.git" ]; then
  git clone --depth 1 https://github.com/IBM/AMLSim.git "${CACHE}"
else
  echo "[skip] AMLSim 已存在：${CACHE}"
fi

for tgz in "${CACHE}"/sample/*.tgz; do
  echo "[extract] $(basename "${tgz}")"
  tar xzf "${tgz}" -C "${RAW}"
done

# 記錄來源版本，確保實驗可重現
git -C "${CACHE}" rev-parse HEAD > "${RAW}/AMLSIM_COMMIT.txt"
echo "完成。AMLSim commit: $(cat "${RAW}/AMLSIM_COMMIT.txt")"
ls -1 "${RAW}"
