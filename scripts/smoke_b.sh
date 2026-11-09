#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/aml-insight-matplotlib"
if [[ "$(uname -s)" == "Darwin" ]]; then
  AML_OMP_DIR="$("${ROOT}/.venv/bin/python" -c 'import pathlib, sklearn; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')"
  export DYLD_LIBRARY_PATH="${AML_OMP_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi
exec "${ROOT}/.venv/bin/python" "${ROOT}/scripts/smoke_b.py" "$@"
