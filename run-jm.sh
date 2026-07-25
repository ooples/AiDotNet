#!/bin/bash
# One generated-shard iteration matching CI exactly: rebuild -> serial heavy-shard config -> shard run.
# Shard letters come from $1 (default the full J-M set) so each split shard can be timed on its own,
# e.g. `bash /repo/run-jm.sh M`. Mirrors the sonarcloud.yml filters exactly.
# Logs go to container-local /tmp (immune to Windows bind-mount hiccups); source/build use the /repo mount.
set -o pipefail
cd /repo || { echo "FATAL: /repo bind mount missing"; exit 3; }

if ! ( echo probe > /repo/_mnt_probe && rm -f /repo/_mnt_probe ) 2>/dev/null; then
  echo "FATAL: /repo bind mount not writable -- restart the container"; exit 3
fi

# Accept either a letter set (e.g. "M", "JK") or "RAW:<tag>:<alternation>" for the second-letter
# shards, e.g. RAW:Ma-Ml:FullyQualifiedName~Generated.Ma|FullyQualifiedName~Generated.MA|...
if [ "${1#RAW:}" != "${1:-}" ]; then
  REST="${1#RAW:}"
  TAG="${REST%%:*}"
  ALT="${REST#*:}"
  BLOG=/tmp/loop-build-$TAG.log
  TLOG=/tmp/loop-test-$TAG.log
  RAWMODE=1
fi
LETTERS="${1:-JKLM}"
[ -z "${RAWMODE:-}" ] && TAG="$LETTERS"
[ -z "${RAWMODE:-}" ] && ALT=""
i=0
while [ -z "${RAWMODE:-}" ] && [ $i -lt ${#LETTERS} ]; do
  L=$(printf '%s' "$LETTERS" | cut -c$((i+1)))
  [ -n "$ALT" ] && ALT="$ALT|"
  ALT="${ALT}FullyQualifiedName~Generated.$L"
  i=$((i+1))
done

BLOG=${BLOG:-/tmp/loop-build-$TAG.log}
TLOG=${TLOG:-/tmp/loop-test-$TAG.log}

echo "=== [$(date +%H:%M:%S)] BUILD (shard $TAG) ==="
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 > "$BLOG" 2>&1
cp -f "$BLOG" "/repo/loop-build-$TAG.log" 2>/dev/null || true
if ! grep -q "Build succeeded" "$BLOG"; then
  echo "BUILD FAILED:"; grep -E ": error" "$BLOG" | sort -u | head -20; exit 1
fi
echo "build ok"

# Serial heavy-shard runner config (exactly what CI rewrites for the generated heavy shards).
RUNNER=tests/AiDotNet.Tests/bin/Release/net10.0/xunit.runner.json
cat > "$RUNNER" <<'JSON'
{ "$schema":"https://xunit.net/schema/current/xunit.runner.schema.json","parallelizeAssembly":false,"parallelizeTestCollections":false,"maxParallelThreads":1,"preEnumerateTheories":false,"methodDisplay":"classAndMethod","diagnosticMessages":false }
JSON
[ -s "$RUNNER" ] || { echo "FATAL: could not write $RUNNER (mount drop)"; exit 3; }

echo "=== [$(date +%H:%M:%S)] TEST (shard $TAG, enforcing CI 45min job timeout) ==="
TEST_START=$(date +%s)
DOTNET_gcServer=0 timeout 45m dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release --framework net10.0 --no-build --no-restore \
  --filter "(FullyQualifiedName~ModelFamilyTests.Generated&($ALT))&Category!=HeavyTimeout" \
  --blame-hang-timeout 5min --blame-hang-dump-type none --logger "console;verbosity=normal" > "$TLOG" 2>&1
TEST_EXIT=$?; TEST_MIN=$(( ($(date +%s) - TEST_START) / 60 ))
cp -f "$TLOG" "/repo/loop-test-$TAG.log" 2>/dev/null || true
echo "=== [$(date +%H:%M:%S)] RESULT shard $TAG (${TEST_MIN} min, exit ${TEST_EXIT}) ==="
[ "$TEST_EXIT" = "124" ] && echo ">>> TIMED OUT at 45min -- CI would CANCEL this shard <<<"
grep -E "^(Passed|Failed)! " "$TLOG" | tail -1
echo "--- failing classes (count) ---"
grep -oE "^  Failed AiDotNet\.Tests\.ModelFamilyTests\.Generated\.[A-Za-z0-9_]+Tests" "$TLOG" \
  | sed "s/.*Generated\.//;s/Tests$//" | sort | uniq -c
echo "--- heavy (OOM/timeout) classes ---"
grep -B8 -E "OutOfMemoryException|timed out after" "$TLOG" \
  | grep -oE "Generated\.[A-Za-z0-9_]+Tests" | sed "s/Generated\.//;s/Tests$//" | sort -u
echo "=== [$(date +%H:%M:%S)] DONE shard $TAG ==="
