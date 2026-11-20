#!/usr/bin/env bash
# Admit HTTPS (443) only from Cloudflare, so visitors cannot reach the server around Cloudflare's
# DDoS protection. Keeps SSH open. Safe to rerun; setup-vps.sh installs it as a weekly cron job
# because Cloudflare occasionally adds address ranges.
#
# Usage (as root): cloudflare-firewall.sh [ssh-port]
set -euo pipefail

ssh_port="${1:-${AIDOTNET_SSH_PORT:-22}}"
comment="aidotnet-cloudflare"

ranges=$(mktemp)
trap 'rm -f "$ranges"' EXIT
for family in v4 v6; do
  curl -fsS --retry 3 --max-time 30 "https://www.cloudflare.com/ips-$family" >> "$ranges"
  echo >> "$ranges"
done
# A truncated download must never replace a working rule set with an empty one.
count=$(grep -cE '^[0-9a-f:.]+/[0-9]+$' "$ranges" || true)
if [ "$count" -lt 10 ]; then
  echo "Fetched only $count Cloudflare ranges; leaving the firewall unchanged." >&2
  exit 1
fi

# SSH first, so enabling the firewall can never lock the operator out.
ufw allow "$ssh_port/tcp" comment "ssh" >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null

# Add the current ranges (ufw skips ones already present), then drop only ranges Cloudflare no
# longer publishes. Port 443 is never closed in between.
while read -r cidr; do
  ufw allow proto tcp from "$cidr" to any port 443 comment "$comment" >/dev/null
done < <(grep -E '^[0-9a-f:.]+/[0-9]+$' "$ranges")
while read -r cidr; do
  if ! grep -qxF "$cidr" "$ranges"; then
    ufw delete allow proto tcp from "$cidr" to any port 443 >/dev/null
  fi
done < <(ufw status | grep "# $comment" | awk '{print $3}' | sort -u)

ufw --force enable >/dev/null
echo "Port 443 admits $count Cloudflare ranges; SSH stays open on $ssh_port."

# A pre-existing "allow 443/80 from anywhere" rule would let attackers skip Cloudflare.
if ufw status | grep -E '^(443|80)(/tcp)?( \(v6\))? +ALLOW +Anywhere' >/dev/null; then
  echo "WARNING: port 80 or 443 is still open to everyone through an older rule:" >&2
  ufw status | grep -E '^(443|80)(/tcp)?( \(v6\))? +ALLOW +Anywhere' >&2
  echo "Remove it (for example: ufw delete allow 443/tcp) unless another site on this server needs it." >&2
fi
