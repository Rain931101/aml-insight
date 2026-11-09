#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/aml-insight-matplotlib}"
exec uv run python scripts/smoke_b.py "$@"
