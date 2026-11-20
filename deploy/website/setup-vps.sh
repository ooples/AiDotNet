#!/usr/bin/env bash
# One-time (and safely repeatable) setup of the VPS that serves aidotnet.dev.
# Run as root on Ubuntu 22.04+/Debian 12+ from a checkout of this directory:
#
#   sudo bash setup-vps.sh --deploy-key /path/to/github-deploy-key.pub [--ssh-port 22]
#
# Before running, place the Cloudflare Origin CA certificate and key (README step 2) at
#   /etc/caddy/certs/aidotnet-origin.pem and /etc/caddy/certs/aidotnet-origin.key
#
# What it does: installs Caddy, rsync and ufw; creates the aidotnet-deploy account that GitHub
# Actions uploads with; creates /srv/aidotnet/{www,test}; installs the Caddy site config and the
# preview secret; restricts port 443 to Cloudflare; schedules the weekly firewall refresh.
# It never deletes existing sites, users or firewall rules.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deploy_key="" ssh_port=22
while [ $# -gt 0 ]; do
  case "$1" in
    --deploy-key) deploy_key="$2"; shift 2 ;;
    --ssh-port) ssh_port="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ "$(id -u)" -eq 0 ] || { echo "Run as root (sudo)." >&2; exit 1; }
[ -n "$deploy_key" ] && [ -f "$deploy_key" ] || { echo "Pass --deploy-key <public key file>." >&2; exit 1; }
grep -qE '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-[a-z0-9-]+) ' "$deploy_key" || { echo "$deploy_key is not an SSH public key." >&2; exit 1; }
for file in /etc/caddy/certs/aidotnet-origin.pem /etc/caddy/certs/aidotnet-origin.key; do
  [ -s "$file" ] || { echo "Missing $file. Create the Cloudflare Origin certificate first (README step 2)." >&2; exit 1; }
done

listener=$(ss -ltnpH 'sport = :443' 2>/dev/null | grep -v '"caddy"' || true)
if [ -n "$listener" ]; then
  echo "Another program already listens on port 443:" >&2
  echo "$listener" >&2
  echo "Stop it, or serve aidotnet.dev from that web server instead; Caddy needs port 443." >&2
  exit 1
fi

echo "== Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl gnupg openssl rsync ufw unattended-upgrades >/dev/null
if ! command -v caddy >/dev/null; then
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy >/dev/null
fi
caddy version

echo "== Deploy account"
if ! id aidotnet-deploy >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash aidotnet-deploy
  passwd --lock aidotnet-deploy >/dev/null
fi
install -d -m 700 -o aidotnet-deploy -g aidotnet-deploy /home/aidotnet-deploy/.ssh
# "restrict" disables port forwarding, agent forwarding and terminals for this key.
printf 'restrict %s\n' "$(cat "$deploy_key")" > /home/aidotnet-deploy/.ssh/authorized_keys
chown aidotnet-deploy:aidotnet-deploy /home/aidotnet-deploy/.ssh/authorized_keys
chmod 600 /home/aidotnet-deploy/.ssh/authorized_keys

echo "== Site directories"
for site in www test; do
  install -d -m 755 -o aidotnet-deploy -g aidotnet-deploy "/srv/aidotnet/$site" "/srv/aidotnet/$site/releases"
done
install -d -m 755 /srv/aidotnet
install -d -m 755 -o caddy -g caddy /var/log/caddy

echo "== Certificates"
chown -R root:caddy /etc/caddy/certs
chmod 750 /etc/caddy/certs
chmod 640 /etc/caddy/certs/aidotnet-origin.pem /etc/caddy/certs/aidotnet-origin.key

echo "== Preview secret"
if [ ! -s /etc/caddy/aidotnet.env ]; then
  # Generate first, write second: a failed openssl must not leave a file holding an empty secret.
  secret="$(openssl rand -hex 32)"
  printf 'AIDOTNET_PREVIEW_SECRET=%s\n' "$secret" > /etc/caddy/aidotnet.env
fi
chown root:caddy /etc/caddy/aidotnet.env
chmod 640 /etc/caddy/aidotnet.env
# The same check runs before every Caddy start (ExecStartPre below), so a later hand edit of the
# file cannot put an empty or malformed secret in front of the preview site.
install -m 755 "$here/check-preview-secret.sh" /usr/local/sbin/aidotnet-check-preview-secret
/usr/local/sbin/aidotnet-check-preview-secret /etc/caddy/aidotnet.env
install -d /etc/systemd/system/caddy.service.d
printf '[Service]\nEnvironmentFile=/etc/caddy/aidotnet.env\nExecStartPre=/usr/local/sbin/aidotnet-check-preview-secret /etc/caddy/aidotnet.env\n' \
  > /etc/systemd/system/caddy.service.d/aidotnet.conf
systemctl daemon-reload

echo "== Caddy config"
install -m 644 "$here/aidotnet.caddy" /etc/caddy/aidotnet.caddy
if ! grep -qxF 'import /etc/caddy/aidotnet.caddy' /etc/caddy/Caddyfile; then
  cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.before-aidotnet.$(date +%s)"
  printf '\nimport /etc/caddy/aidotnet.caddy\n' >> /etc/caddy/Caddyfile
fi
# Validate as the caddy user: validation opens the log files and certificates, so running it as
# root would leave root-owned logs that the service then cannot open.
chown -R caddy:caddy /var/log/caddy
runuser -u caddy -- env "$(grep '^AIDOTNET_PREVIEW_SECRET=' /etc/caddy/aidotnet.env)" \
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
systemctl enable caddy >/dev/null
if systemctl is-active --quiet caddy; then systemctl reload caddy; else systemctl start caddy; fi
systemctl is-active --quiet caddy || { journalctl -u caddy -n 20 --no-pager >&2; exit 1; }

echo "== Firewall"
install -m 755 "$here/cloudflare-firewall.sh" /usr/local/sbin/aidotnet-cloudflare-firewall
/usr/local/sbin/aidotnet-cloudflare-firewall "$ssh_port"
printf '#!/bin/sh\nAIDOTNET_SSH_PORT=%s /usr/local/sbin/aidotnet-cloudflare-firewall >/var/log/aidotnet-cloudflare-firewall.log 2>&1\n' "$ssh_port" \
  > /etc/cron.weekly/aidotnet-cloudflare-firewall
chmod 755 /etc/cron.weekly/aidotnet-cloudflare-firewall

echo
echo "Done. Values for the GitHub repository secrets (README step 5):"
echo "  WEBSITE_SSH_USER        = aidotnet-deploy"
echo "  WEBSITE_SSH_KNOWN_HOSTS = output of: ssh-keyscan -p $ssh_port $(hostname -I | awk '{print $1}')"
echo "  PREVIEW_BYPASS_SECRET   = $(sed -n 's/^AIDOTNET_PREVIEW_SECRET=//p' /etc/caddy/aidotnet.env)"
