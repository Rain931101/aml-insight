#!/usr/bin/env bash
# Local launcher; uses bundled scikit-learn OpenMP on Macs without Homebrew libomp.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${AML_B_PYTHON:-${ROOT}/.venv/bin/python}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/aml-insight-matplotlib}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  AML_OMP_DIR="$("$PYTHON" -c 'import pathlib, sklearn; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')"
  if [[ -f "${AML_OMP_DIR}/libomp.dylib" ]]; then
    export DYLD_LIBRARY_PATH="${AML_OMP_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  fi
fi
exec "$PYTHON" -m aml_insight.evaluation "$@"
