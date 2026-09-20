#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: $0 OWNER/REPOSITORY [public|private]" >&2; exit 2; }
GITHUB_REPO=$1
VISIBILITY=${2:-private}
[[ "${VISIBILITY}" == public || "${VISIBILITY}" == private ]] || {
  echo "Visibility must be public or private" >&2
  exit 2
}
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${ROOT}"
gh auth status >/dev/null

login=$(gh api user --jq .login)
account_id=$(gh api user --jq .id)
display_name=$(gh api user --jq '.name // .login')
git config user.name "${display_name}"
git config user.email "${account_id}+${login}@users.noreply.github.com"

python3 tools/verify_training_package.py
git add -A
git diff --cached --check
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git commit -m "Add Counterfactual RLVR and BiPS 7B training tracks"
elif ! git diff --cached --quiet; then
  git commit -m "Update Counterfactual RLVR and BiPS 7B training package"
fi

if gh repo view "${GITHUB_REPO}" >/dev/null 2>&1; then
  if [[ -n "$(git ls-remote --heads "https://github.com/${GITHUB_REPO}.git")" ]]; then
    echo "Remote repository already has branches; refusing a history-overwriting push" >&2
    exit 1
  fi
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/${GITHUB_REPO}.git"
  git push -u origin main
else
  gh repo create "${GITHUB_REPO}" --"${VISIBILITY}" --source . --remote origin --push \
    --description "Controlled Counterfactual RLVR and BiPS full fine-tuning for Qwen2.5-VL-7B"
fi

echo "Published https://github.com/${GITHUB_REPO}"
