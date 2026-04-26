#!/bin/bash
# Container-side j-submit. Mirrors ~/bin/j-submit on the host but points at the
# container's bind-mounted jacobian-reports clone. The container hostname
# (claude-sandbox) is recorded in submitted_from so sandbox submissions are
# distinguishable from host submissions.

set -euo pipefail

REPO=/workspaces/jacobian-analyses

if [[ $# -lt 1 ]]; then
  echo "Usage: j-submit <experiment> [override1 override2 ...]"
  exit 1
fi

exp="$1"; shift
overrides=("$@")
ts="$(date -u +%Y%m%dT%H%M%SZ)"
fname="${ts}-${exp}.yaml"
target="$REPO/instructions/pending/$fname"

cd "$REPO"
git pull --rebase --autostash 2>&1 | tail -2 || true

mkdir -p instructions/pending
{
  echo "experiment: $exp"
  if [[ ${#overrides[@]} -gt 0 ]]; then
    echo "overrides:"
    for o in "${overrides[@]}"; do
      printf '  - "%s"\n' "$o"
    done
  else
    echo "overrides: []"
  fi
  echo "submitted_at: $ts"
  echo "submitted_from: $(hostname)"
} > "$target"

git add "$target"
git commit -m "submit: $exp" 2>&1 | tail -2
git push 2>&1 | tail -2

echo
echo "==> Submitted: $fname"
echo "    File: $target"
echo "    The engaging controller will pick it up within ~5 min."
