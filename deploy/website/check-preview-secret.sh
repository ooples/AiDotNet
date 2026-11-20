#!/usr/bin/env bash
# Refuse to start Caddy unless the preview secret is well formed.
#
# aidotnet.caddy compares the X-AiDotNet-Preview-Bypass header and the aidotnet_preview cookie with
# AIDOTNET_PREVIEW_SECRET. An empty or malformed value must never reach it: an empty placeholder
# would either break the config load or let a request with an empty header or cookie in.
# setup-vps.sh installs this as an ExecStartPre of caddy.service, so it runs on every start and
# restart, including one that follows a hand edit of the file.
#
# Usage: check-preview-secret.sh [env-file]
set -euo pipefail

file="${1:-/etc/caddy/aidotnet.env}"
fail() { echo "$file: $1 Caddy will not start until it holds exactly one line AIDOTNET_PREVIEW_SECRET=<64 lowercase hex characters> (openssl rand -hex 32)." >&2; exit 1; }

[ -r "$file" ] || fail "missing or unreadable."
# systemd uses the last assignment in an EnvironmentFile, so a second line could override a valid first one.
[ "$(grep -c '^[[:space:]]*AIDOTNET_PREVIEW_SECRET=' "$file" || true)" -eq 1 ] || fail "must assign AIDOTNET_PREVIEW_SECRET exactly once."
grep -qE '^AIDOTNET_PREVIEW_SECRET=[0-9a-f]{64}$' "$file" || fail "the preview secret is empty or malformed."
