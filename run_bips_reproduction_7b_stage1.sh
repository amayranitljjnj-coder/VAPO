#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
exec "${ROOT}/run_bips_reproduction_7b.sh" stage1 "$@"
