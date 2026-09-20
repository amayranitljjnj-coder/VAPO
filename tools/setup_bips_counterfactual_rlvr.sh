#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BIPS_ROOT=${BIPS_ROOT:-${ROOT}/vendor/BiPS}
BIPS_REPOSITORY=${BIPS_REPOSITORY:-https://github.com/zss02/BiPS.git}
BIPS_COMMIT=${BIPS_COMMIT:-41d3a60a558c6901a8eeb4d6fbb365dc48379ce6}
PATCH_FILE=${PATCH_FILE:-${ROOT}/patches/bips_counterfactual_rlvr.patch}

[[ -s "${PATCH_FILE}" ]] || { echo "Missing patch: ${PATCH_FILE}" >&2; exit 1; }
if [[ ! -d "${BIPS_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${BIPS_ROOT}")"
  git clone "${BIPS_REPOSITORY}" "${BIPS_ROOT}"
fi

current_commit=$(git -C "${BIPS_ROOT}" rev-parse HEAD)
if [[ "${current_commit}" != "${BIPS_COMMIT}" ]]; then
  [[ -z "$(git -C "${BIPS_ROOT}" status --porcelain)" ]] || {
    echo "BiPS checkout has local changes; refusing to replace them" >&2
    exit 1
  }
  git -C "${BIPS_ROOT}" fetch origin "${BIPS_COMMIT}"
  git -C "${BIPS_ROOT}" checkout --detach "${BIPS_COMMIT}"
fi

if git -C "${BIPS_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Counterfactual RLVR patch is already applied."
elif git -C "${BIPS_ROOT}" apply --check "${PATCH_FILE}"; then
  git -C "${BIPS_ROOT}" apply "${PATCH_FILE}"
  echo "Applied ${PATCH_FILE}"
else
  echo "Patch does not apply cleanly to ${BIPS_ROOT} at ${BIPS_COMMIT}" >&2
  exit 1
fi

echo "BiPS is ready at ${BIPS_ROOT}"
