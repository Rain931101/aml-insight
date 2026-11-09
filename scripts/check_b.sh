#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/aml-insight-matplotlib}"
uv run python -m pytest "$@"
uv run ruff check src tests scripts/smoke_b.py
