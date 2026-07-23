#!/usr/bin/env bash
set -euo pipefail

CONFIGURATION="${CONFIGURATION:-Release}"
FILTER="${FILTER:-Category!=GPU&Category!=Integration}"
NO_BUILD="${NO_BUILD:-0}"
RUN_SERVING_IN_PARALLEL="${RUN_SERVING_IN_PARALLEL:-0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$NO_BUILD" != "1" ]]; then
  echo "Building AiDotNet.sln ($CONFIGURATION)..."
  dotnet build AiDotNet.sln -c "$CONFIGURATION"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
results_root="$repo_root/TestResults/sharded/$timestamp"
mkdir -p "$results_root"

run_shard() {
  local name="$1"; shift
  local log="$results_root/$name.log"
  echo "Starting $name..."
  ( dotnet "$@" --results-directory "$results_root" --logger "trx;LogFileName=$name.trx" --filter "$FILTER" >"$log" 2>&1 ) &
  echo $!
}

pids=()
pids+=("$(run_shard AiDotNetTests_net8 test tests/AiDotNet.Tests/AiDotNetTests.csproj -c "$CONFIGURATION" --framework net8.0 --no-build)")
pids+=("$(run_shard AiDotNetTests_net471 test tests/AiDotNet.Tests/AiDotNetTests.csproj -c "$CONFIGURATION" --framework net471 --no-build)")
pids+=("$(run_shard AiDotNetTensorsTests_net8 test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c "$CONFIGURATION" --framework net8.0 --no-build)")

if [[ "$RUN_SERVING_IN_PARALLEL" == "1" ]]; then
  pids+=("$(run_shard AiDotNetServingTests_net8 test tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj -c "$CONFIGURATION" --framework net8.0 --no-build)")
fi

echo "Waiting for parallel shards (logs: $results_root)..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$RUN_SERVING_IN_PARALLEL" != "1" ]]; then
  echo "Running Serving tests (sequential)..."
  dotnet test tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj -c "$CONFIGURATION" --framework net8.0 --no-build --results-directory "$results_root" --logger "trx;LogFileName=AiDotNetServingTests_net8.trx" --filter "$FILTER"
fi

if [[ "$failed" == "1" ]]; then
  echo "One or more shards failed. See logs under: $results_root" >&2
  exit 1
fi

echo "All shards passed. Logs: $results_root"
