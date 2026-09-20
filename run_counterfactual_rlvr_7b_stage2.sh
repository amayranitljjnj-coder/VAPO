#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
exec "${ROOT}/run_counterfactual_rlvr_7b.sh" stage2 "${1:-train}" "${@:2}"
