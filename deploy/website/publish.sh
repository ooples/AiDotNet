#!/usr/bin/env bash
# Upload a built website to the VPS and switch the live site to it atomically.
#
# Usage: publish.sh <dist-directory> <www|test>
# Environment: WEBSITE_SSH_HOST, WEBSITE_SSH_USER, WEBSITE_SSH_KEY (private key text),
#              WEBSITE_SSH_KNOWN_HOSTS (pinned host key lines), optional WEBSITE_SSH_PORT.
#
# The upload goes to a new release directory; only after it is complete does the "current"
# symlink move, so visitors never see a half-uploaded site. Unchanged files are hard-linked
# from the previous release, so each deploy transfers only what changed. The three newest
# releases are kept; to roll back, point "current" at an older one on the server.
set -euo pipefail

dist="${1:?usage: publish.sh <dist-directory> <www|test>}"
site="${2:?usage: publish.sh <dist-directory> <www|test>}"
case "$site" in www|test) ;; *) echo "::error::site must be www or test, got '$site'"; exit 1 ;; esac
: "${WEBSITE_SSH_HOST:?Missing WEBSITE_SSH_HOST}"
: "${WEBSITE_SSH_USER:?Missing WEBSITE_SSH_USER}"
: "${WEBSITE_SSH_KEY:?Missing WEBSITE_SSH_KEY}"
: "${WEBSITE_SSH_KNOWN_HOSTS:?Missing WEBSITE_SSH_KNOWN_HOSTS}"
[ -f "$dist/index.html" ] || { echo "::error::$dist/index.html is missing; refusing to publish"; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
printf '%s\n' "$WEBSITE_SSH_KEY" > "$work/key"
printf '%s\n' "$WEBSITE_SSH_KNOWN_HOSTS" > "$work/known_hosts"
chmod 600 "$work/key"
ssh_command="ssh -p ${WEBSITE_SSH_PORT:-22} -i $work/key -o UserKnownHostsFile=$work/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
target="$WEBSITE_SSH_USER@$WEBSITE_SSH_HOST"
base="/srv/aidotnet/$site"
release="$(date -u +%Y%m%dT%H%M%SZ)-${GITHUB_SHA:-manual}"

rsync -a --delete --link-dest="$base/current/" -e "$ssh_command" "$dist/" "$target:$base/releases/$release/"

$ssh_command "$target" bash -s -- "$base" "$release" <<'REMOTE'
set -euo pipefail
base=$1 release=$2
test -f "$base/releases/$release/index.html"
ln -sfn "releases/$release" "$base/current.next"
mv -Tf "$base/current.next" "$base/current"
# Release names start with a UTC timestamp, so name order is age order; the newest is current.
cd "$base/releases"
ls -1 | sort -r | tail -n +4 | xargs -r rm -rf --
echo "Live: $base/current -> releases/$release"
REMOTE
