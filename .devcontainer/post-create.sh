#!/usr/bin/env bash
# Runs once per container create (not on restart). Idempotent.
set -euo pipefail

REPO_JODE=/workspaces/JacobianODE
REPO_ANALYSES=/workspaces/jacobian-analyses

if [[ ! -r /run/secrets/gh-pat ]]; then
  echo "ERROR: /run/secrets/gh-pat not mounted. Create ~/.config/claude-sandbox/gh-pat on the host." >&2
  exit 1
fi

git config --global credential.helper \
  '!f() { echo "username=adamjeisen"; echo "password=$(cat /run/secrets/gh-pat)"; }; f'

# Container has no SSH keys (intentional — see threat model). Rewrite any
# ssh:// or scp-style GitHub git URLs to HTTPS so public deps (e.g. wmtask
# pinned in pyproject.toml as ssh://git@github.com/...) clone without auth.
# Private repos where the PAT has access still work via the credential
# helper above.
# `url.<base>.insteadOf` is multi-valued; --add appends rather than overwrites
# so both rewrite rules coexist.
git config --global --add url."https://github.com/".insteadOf "ssh://git@github.com/"
git config --global --add url."https://github.com/".insteadOf "git@github.com:"

git -C "$REPO_JODE"     remote set-url origin https://github.com/adamjeisen/JacobianODE.git     || true
git -C "$REPO_ANALYSES" remote set-url origin https://github.com/adamjeisen/jacobian-reports.git || true

if [[ -r /run/secrets/wandb-api-key ]]; then
  umask 077
  WANDB_KEY="$(cat /run/secrets/wandb-api-key)"
  printf 'machine api.wandb.ai\n  login user\n  password %s\n' "$WANDB_KEY" > "$HOME/.netrc"
else
  echo "WARN: no wandb-api-key secret; load_run() will fail on wandb API calls." >&2
fi

mkdir -p "$HOME/bin"
install -m 0755 "$REPO_JODE/.devcontainer/_j-submit-impl.sh" "$HOME/bin/j-submit"

mkdir -p "$HOME/.claude/hooks"
ln -sf "$REPO_JODE/.claude/hooks/uv-cu118-guard.sh" "$HOME/.claude/hooks/uv-cu118-guard.sh"
install -m 0644 "$REPO_JODE/.devcontainer/claude-settings.json" "$HOME/.claude/settings.json"

cd "$REPO_JODE"
# Pin Python 3.13: torch 2.7.1+cu118 only has wheels for cp312/cp313/cp313t
# (no cp314 yet). 3.13 is the latest cu118-compatible. UV_PROJECT_ENVIRONMENT
# (set in devcontainer.json containerEnv) puts the venv under the uv-cache
# volume, isolated from the host's .venv in the bind-mounted workspace.
uv sync --python 3.13 --no-group cu128 --group cu118

uv run --no-sync python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA not visible in container'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"

echo "post-create complete."
