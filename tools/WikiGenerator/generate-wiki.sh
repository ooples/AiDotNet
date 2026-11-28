#!/usr/bin/env bash
# Generates the API reference wiki. Run from the repository root:
#
#   bash tools/WikiGenerator/generate-wiki.sh [output-directory]
#
# The reference is generated from the library built from this checkout. When that build fails (for
# example while master waits on an unpublished dependency), it is generated from the latest stable
# AiDotNet package on nuget.org instead, so a website build or deploy does not depend on the library
# compiling. The fallback is announced as a workflow warning and in the step summary; it never hides a
# failure of the generator itself.
set -euo pipefail

out="${1:-website/src/content/docs/reference/wiki}"

if dotnet build src/AiDotNet.csproj -c Debug -f net8.0; then
  dotnet run --project tools/WikiGenerator -c Debug -- src/bin/Debug/net8.0/AiDotNet.xml "$out"
  exit 0
fi

index_url="https://api.nuget.org/v3-flatcontainer/aidotnet/index.json"
version="$(curl -fsS --retry 3 --connect-timeout 10 --max-time 60 "$index_url" \
  | jq -r '[.versions[] | select(test("-") | not)] | last // empty')"
if [ -z "$version" ]; then
  echo "::error::src/AiDotNet.csproj did not build, and no stable AiDotNet package version could be read from $index_url."
  exit 1
fi

message="src/AiDotNet.csproj did not build, so the API reference was generated from the published AiDotNet $version package instead of this checkout."
echo "::warning title=API reference generated from the published package::$message"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  printf '### API reference\n\n%s\n' "$message" >> "$GITHUB_STEP_SUMMARY"
fi

# The generator loads AiDotNet.dll from its own output directory, so it is rebuilt against the package.
dotnet build tools/WikiGenerator -c Debug -p:PublishedAiDotNetVersion="$version"

packages="$(dotnet nuget locals global-packages --list | sed -e 's/^global-packages: //' -e 's/[[:space:]]*$//')"
xml="$packages/aidotnet/$version/lib/net10.0/AiDotNet.xml"
if [ ! -f "$xml" ]; then
  echo "::error::The AiDotNet $version package has no XML documentation at $xml."
  exit 1
fi

dotnet run --project tools/WikiGenerator -c Debug --no-build -- "$xml" "$out"
