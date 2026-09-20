#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 OWNER/REPOSITORY" >&2; exit 2; }
GITHUB_REPO=$1
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}
MODEL_PATH=${MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen2.5-VL-7B-Instruct}
TAG=${TAG:-qwen2.5-vl-7b-instruct-original}
PART_SIZE=${PART_SIZE:-1900MiB}
FILE_LIST=${ROOT}/counterfactual_rlvr/model_files.txt
MODEL_HASHES=${ROOT}/counterfactual_rlvr/Qwen2.5-VL-7B-Instruct.sha256
TEMP_ROOT=${TEMP_ROOT:-${WORKSPACE_ROOT}/training_outputs/publish_tmp}

command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }
gh auth status >/dev/null
[[ -d "${MODEL_PATH}" ]] || { echo "Missing model: ${MODEL_PATH}" >&2; exit 1; }
[[ -s "${FILE_LIST}" && -s "${MODEL_HASHES}" ]] || { echo "Missing model manifests" >&2; exit 1; }
(cd "${MODEL_PATH}" && sha256sum -c "${MODEL_HASHES}")

mkdir -p "${TEMP_ROOT}"
PACKAGE_DIR=$(mktemp -d "${TEMP_ROOT%/}/qwen25vl7b-release.XXXXXX")
trap 'rm -rf -- "${PACKAGE_DIR}"' EXIT
echo "Creating transient ${PART_SIZE} release parts in ${PACKAGE_DIR}"
tar -C "${MODEL_PATH}" -cf - --files-from "${FILE_LIST}" | \
  split -b "${PART_SIZE}" -d -a 2 - "${PACKAGE_DIR}/Qwen2.5-VL-7B-Instruct.tar.part-"
cp "${MODEL_HASHES}" "${PACKAGE_DIR}/Qwen2.5-VL-7B-Instruct.sha256"
cp "${ROOT}/licenses/Qwen2.5-VL-7B-Instruct-APACHE-2.0.txt" "${PACKAGE_DIR}/LICENSE.txt"
(cd "${PACKAGE_DIR}" && sha256sum Qwen2.5-VL-7B-Instruct.tar.part-* > RELEASE_PARTS.sha256)
cat >"${PACKAGE_DIR}/RESTORE.txt" <<EOF
Download every Qwen2.5-VL-7B-Instruct.tar.part-* asset, then run:

  cat Qwen2.5-VL-7B-Instruct.tar.part-* | tar -xf - -C /target/model/directory
  (cd /target/model/directory && sha256sum -c Qwen2.5-VL-7B-Instruct.sha256)

The repository's tools/download_qwen25vl_7b_github_release.sh automates this.
EOF

if gh release view "${TAG}" --repo "${GITHUB_REPO}" >/dev/null 2>&1; then
  echo "Release ${TAG} already exists; uploading with --clobber"
  gh release upload "${TAG}" "${PACKAGE_DIR}"/* --clobber --repo "${GITHUB_REPO}"
else
  gh release create "${TAG}" "${PACKAGE_DIR}"/* \
    --repo "${GITHUB_REPO}" \
    --title "Original Qwen2.5-VL-7B-Instruct training model" \
    --notes "Byte-identical training base model, split into sub-2-GiB assets for GitHub Release distribution. Official upstream: Qwen/Qwen2.5-VL-7B-Instruct. Apache-2.0 model license; retain the upstream README and license metadata."
fi
echo "Published https://github.com/${GITHUB_REPO}/releases/tag/${TAG}"
