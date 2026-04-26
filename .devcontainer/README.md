# Claude Code sandbox (devcontainer)

Kernel-isolated sandbox for running Claude Code autonomously on endeavour, with GPU access, git-based sweep submission to engaging, and read-only access to engaging checkpoints via sshfs. No SSH from inside the container.

See `/home/adameisen/.claude/plans/zazzy-meandering-snowflake.md` for the full design rationale.

## Threat model — read before configuring

The container's kernel firewall, missing `~/` mount, and scoped PAT are designed to limit what a compromised sandbox session (e.g. via prompt injection from training logs or pulled GitHub content) can do. They DO NOT eliminate one residual exfiltration path that depends on operator awareness:

**`adamjeisen/jacobian-reports` is published to the public web** at `adamjeisen.com/jacobian-reports/` (GitHub Pages). The sandbox's PAT has `Contents:write` on that repo because legitimate sweep submission requires it (`j-submit` writes YAML there, and the engaging controller cron pushes analysis output back). An injected agent can commit any file it can read inside the container — including secrets it pulls from `/run/secrets/` if it manages to escalate, or sensitive intermediates from a wandb run dump — and that file is web-readable within one push cycle. The kernel firewall does not catch this because `github.com` is (and must be) allow-listed.

Implications:
- **Never expand the PAT's repo scope to a private repo whose contents you'd be unhappy to see public.** A scope expansion = a new exfil sink. Public repos are no worse than the existing surface; private repos that hold credentials, internal data, etc. would be strictly worse.
- **Never add a writable bind-mount for sensitive data.** The sandbox's only writable host paths must be the two repo clones.
- **Don't instruct the sandbox Claude to "log debug output to jacobian-reports/diagnostics/"** unless the output is intentionally shareable. Stack traces with paths, wandb config dumps, and similar are effectively public pastes the moment they're committed.
- **Rotation discipline:** if the PAT is suspected leaked, revoke it on GitHub immediately; no in-container action can stop a determined session from pushing as long as the PAT is valid and `github.com` is reachable.

## One-time host prerequisites

All run on endeavour, as user `adameisen`.

### 1. Docker + NVIDIA container toolkit

```
sudo pacman -S docker nvidia-container-toolkit
sudo systemctl enable --now docker
sudo usermod -aG docker adameisen
# log out / log back in (or `newgrp docker` for the current shell)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

The last line must print GPU info.

### 2. devcontainer CLI

```
npm i -g @devcontainers/cli
```

### 3. Fine-grained GitHub PAT

At https://github.com/settings/personal-access-tokens/new:
- Resource owner: `adamjeisen`
- Repository access: select ONLY `JacobianODE` and `jacobian-reports`
- Permissions: `Contents: Read and write`, `Metadata: Read`
- Expiration: 90 days (add calendar reminder to rotate)

Save to `~/.config/claude-sandbox/gh-pat`:
```
mkdir -p ~/.config/claude-sandbox
chmod 700 ~/.config/claude-sandbox
# paste the token into the file, then:
chmod 600 ~/.config/claude-sandbox/gh-pat
```

### 4. Wandb service-account API key

Create a dedicated wandb service account (or at minimum a separate login) so sandbox key rotation doesn't affect the user's interactive wandb sessions. Save to `~/.config/claude-sandbox/wandb-api-key` with `chmod 600`.

### 5. Engaging SSH ControlMaster + sshfs mount

Re-enable the existing (currently-disabled) ssh master unit:
```
systemctl --user enable --now engaging-ssh-master.service
ssh engaging true    # Duo push ONCE; master keeps the session alive
```

`ENGAGING_MOUNT_PATH` is the engaging absolute path bind-mounted into the container (read-only). It must be at or above the path wandb records as `save_dir` in run config so checkpoints resolve at the same path inside the container as on engaging. Currently set to the JacobianODE root on engaging — this exposes everything under it (lightning runs, raw data dumps, anything else). Bounded by your engaging-side ACLs (sshfs runs as `eisenaj@engaging`) and read-only at the FUSE layer.

Set it in your shell profile so both the sshfs unit and devcontainer pick it up (already done in `~/.bashrc` if you followed setup):
```
export ENGAGING_MOUNT_PATH=/orcd/data/ekmiller/001/eisenaj/JacobianODE
```

The mount runs as a user-level systemd service unit (a `.service`, not a `.mount` — `.mount` units have rigid filename-must-match-path rules that get ugly with dashes in the path). The unit file lives at `~/.config/systemd/user/engaging-weights.service` and reads ssh config to multiplex through `engaging-ssh-master.service`'s ControlMaster, so no per-mount Duo push.

```
mkdir -p ~/mnt/engaging-weights
systemctl --user daemon-reload
systemctl --user enable --now engaging-weights.service
ls ~/mnt/engaging-weights   # should show JacobianODE subdirs (lightning/, ...)
```

If `engaging-weights.service` fails to start, check that `engaging-ssh-master.service` is `active` first — the mount `Requires=` it.

## Starting a session

```
export ENGAGING_MOUNT_PATH=/orcd/pool/.../sweeps   # if not already in profile
cd ~/Documents/code/JacobianODE
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
# inside container:
claude
```

Unattended overnight (after smoke-test passes):
```
devcontainer exec --workspace-folder ~/Documents/code/JacobianODE \
  bash -lc 'claude --print "<task description>"'
```

## Smoke test (MUST pass before leaving unattended)

Run these inside the container in order. All must succeed.

1. **Filesystem isolation** — host credentials must be invisible:
   ```
   ls ~/.ssh 2>&1              # expect: No such file or directory
   ls /home/adameisen 2>&1      # expect: No such file or directory
   ls ~/.config/gh 2>&1         # expect: No such file or directory
   ```

2. **Network allow-list**:
   ```
   curl -m 5 https://example.com        # MUST fail (not in allow-list)
   curl -m 5 https://github.com         # MUST succeed
   curl -m 5 https://pypi.org           # MUST succeed
   curl -m 5 https://api.wandb.ai       # MUST succeed
   ```

3. **GPU passthrough**:
   ```
   uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # expect: True GeForce GTX 1050 Ti
   ```

4. **cu118 guard hook active**:
   ```
   # bare `uv run` must be BLOCKED by the PreToolUse hook when invoked by claude:
   claude --print 'run `uv run python -c "print(1)"`'   # should refuse
   # explicit --no-sync succeeds:
   uv run --no-sync python -c "print(1)"
   ```

5. **Git push with PAT** in `/workspaces/jacobian-analyses`:
   ```
   git commit --allow-empty -m "sandbox smoke test"
   git push
   git reset --hard HEAD^
   git push --force-with-lease
   ```

6. **j-submit end-to-end**:
   ```
   j-submit <some-existing-experiment>
   ls /workspaces/jacobian-analyses/instructions/pending/   # new YAML should be here
   # cancel before engaging cron picks it up:
   rm /workspaces/jacobian-analyses/instructions/pending/<just-created>.yaml
   cd /workspaces/jacobian-analyses
   git add -u && git commit -m "cancel sandbox smoke-test submission" && git push
   ```

7. **Weights load end-to-end**:
   ```
   uv run --no-sync python -c "
   from JacobianODE.jacobians.checkpoints import load_run
   r, cfg, *_, m = load_run('<project>', '<known-run-id>')
   print(type(m).__name__)
   "
   ```

8. **Read-only enforcement** on engaging mount:
   ```
   touch "$ENGAGING_MOUNT_PATH/sandbox-should-fail"
   # MUST fail with "Read-only file system"
   ```

9. **Unattended smoke**:
   ```
   # from OUTSIDE the container:
   devcontainer exec --workspace-folder ~/Documents/code/JacobianODE \
     bash -lc 'claude --print "list the files in /workspaces/JacobianODE/jacobians/conf/experiment | head -5"'
   ```

If steps 1, 2, or 8 fail, the isolation is broken — DO NOT proceed. Those are the core security invariants.

## Rotation / maintenance

- **GitHub PAT**: 90-day expiry. Rotate by issuing a new token, overwriting `~/.config/claude-sandbox/gh-pat`. No container rebuild needed.
- **Wandb key**: rotate on a schedule you're comfortable with; overwrite the secret file.
- **Image rebuild**: `devcontainer up --workspace-folder . --rebuild` — persistent volumes (`claude-sandbox-uv-cache`, `claude-sandbox-claude`) survive rebuilds, so uv cache and claude auth are preserved.
- **Wiping persistent state**: `docker volume rm claude-sandbox-uv-cache claude-sandbox-claude` (destructive).

## Scope / non-goals

- No SSH from inside the container. The host's ControlMaster + sshfs handle the one engaging↔endeavour data path (read-only weights).
- No HF / OpenAI tokens. Not needed for the current workload; add later if scope grows.
- No systemd timer to auto-start claude nightly. Add as a follow-up once smoke-test passes.
