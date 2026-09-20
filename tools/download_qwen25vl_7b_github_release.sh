#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: $0 OWNER/REPOSITORY [TARGET_DIR]" >&2; exit 2; }
GITHUB_REPO=$1
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}
TARGET_DIR=${2:-${WORKSPACE_ROOT}/models/Qwen2.5-VL-7B-Instruct}
TAG=${TAG:-qwen2.5-vl-7b-instruct-original}
TEMP_ROOT=${TEMP_ROOT:-${WORKSPACE_ROOT}/training_outputs/download_tmp}

command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }
mkdir -p "${TEMP_ROOT}"
DOWNLOAD_DIR=$(mktemp -d "${TEMP_ROOT%/}/qwen25vl7b-download.XXXXXX")
trap 'rm -rf -- "${DOWNLOAD_DIR}"' EXIT
gh release download "${TAG}" --repo "${GITHUB_REPO}" --dir "${DOWNLOAD_DIR}"
(cd "${DOWNLOAD_DIR}" && sha256sum -c RELEASE_PARTS.sha256)
mkdir -p "${TARGET_DIR}"
cat "${DOWNLOAD_DIR}"/Qwen2.5-VL-7B-Instruct.tar.part-* | tar -xf - -C "${TARGET_DIR}"
cp "${DOWNLOAD_DIR}/Qwen2.5-VL-7B-Instruct.sha256" "${TARGET_DIR}/Qwen2.5-VL-7B-Instruct.sha256"
cp "${DOWNLOAD_DIR}/LICENSE.txt" "${TARGET_DIR}/LICENSE"
(cd "${TARGET_DIR}" && sha256sum -c Qwen2.5-VL-7B-Instruct.sha256)
echo "Model restored and verified at ${TARGET_DIR}"
