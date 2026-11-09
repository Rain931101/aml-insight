#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/aml-insight-matplotlib"
if [[ "$(uname -s)" == "Darwin" ]]; then
  AML_OMP_DIR="$(.venv/bin/python -c 'import pathlib, sklearn; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')"
  export DYLD_LIBRARY_PATH="${AML_OMP_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi
.venv/bin/python -m pytest "$@"
.venv/bin/ruff check src tests scripts/smoke_b.py
