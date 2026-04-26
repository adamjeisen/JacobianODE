#!/usr/bin/env bash
# Kernel-level outbound allow-list for the Claude Code sandbox.
# Default OUTPUT policy is DROP; only resolved IPs for allow-listed hosts are permitted.
# Runs as root via the NOPASSWD sudoers entry set up in the Dockerfile.
#
# Based on the reference firewall script in
# https://github.com/anthropics/claude-code/tree/main/.devcontainer

set -euo pipefail

iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

ipset destroy allowed-hosts 2>/dev/null || true
ipset create allowed-hosts hash:net

iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

ALLOWED_DOMAINS=(
  # Git + GitHub API
  "github.com"
  "api.github.com"
  "codeload.github.com"
  "raw.githubusercontent.com"
  "objects.githubusercontent.com"
  # Python package index
  "pypi.org"
  "files.pythonhosted.org"
  # PyTorch wheels (cu118)
  "download.pytorch.org"
  # Weights & Biases (run.config + scan_history for load_checkpoint)
  "api.wandb.ai"
  "wandb.ai"
  # Claude Code install + inference
  "api.anthropic.com"
  "claude.ai"
  "registry.anthropic.com"
  "statsig.anthropic.com"
)

for domain in "${ALLOWED_DOMAINS[@]}"; do
  ips=$(dig +short A "$domain" | grep -E '^[0-9]+\.' || true)
  if [[ -z "$ips" ]]; then
    echo "WARN: no A records for $domain (non-fatal; may resolve later)" >&2
    continue
  fi
  while IFS= read -r ip; do
    ipset add allowed-hosts "$ip" 2>/dev/null || true
  done <<< "$ips"
done

# GitHub publishes its IP ranges via /meta. Pull them before locking down so CIDR
# blocks survive DNS changes during the session. Best-effort; failures are non-fatal.
gh_meta=$(curl -sS --max-time 10 https://api.github.com/meta || true)
if [[ -n "$gh_meta" ]]; then
  echo "$gh_meta" | python3 -c '
import json, sys
data = json.load(sys.stdin)
for key in ("web", "api", "git"):
    for cidr in data.get(key, []):
        if ":" in cidr:
            continue
        print(cidr)
' | while IFS= read -r cidr; do
    ipset add allowed-hosts "$cidr" 2>/dev/null || true
  done
fi

iptables -A OUTPUT -m set --match-set allowed-hosts dst -j ACCEPT

iptables -P OUTPUT DROP
iptables -P INPUT DROP
iptables -P FORWARD DROP

echo "firewall initialized; $(ipset list allowed-hosts | grep -c '^[0-9]') entries in allowed-hosts"
