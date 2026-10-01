<#
.SYNOPSIS
  Fails when a test class in a sharded project is selected by no shard in .github/test-shards.yml.

.DESCRIPTION
  check-shard-coverage.ps1 guards the generated ModelFamily classes only. Hand-written classes had no
  guard, and on 2026-09-30 269 of them matched no shard filter and never ran anywhere. This runs
  TypeImpact --inventory over the assemblies the Build job just produced: every test method is
  evaluated against every shard filter of its project, and a class with a method no filter selects
  (and no deliberately excluded category) fails the step with the clause that assigns it.

  Run it locally after a Release build:
    pwsh tools/TestImpact/Assert-ShardInventory.ps1
#>
[CmdletBinding()]
param(
    [string] $Repository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string] $OutFile = 'shard-inventory.json'
)

$ErrorActionPreference = 'Stop'
$manifest = Join-Path ([IO.Path]::GetTempPath()) 'shard-inventory-shards.json'
$all = @(yq -o=json -I=0 '.shard' (Join-Path $Repository '.github/test-shards.yml') | ConvertFrom-Json)
if ($all.Count -eq 0) { throw 'test-shards.yml yielded no shards' }
ConvertTo-Json -InputObject @($all) -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8

$tool = Join-Path $Repository 'tools/TestImpact/TypeImpact/TypeImpact.csproj'
& dotnet build $tool -c Release --nologo -v quiet
if ($LASTEXITCODE -ne 0) { throw "building TypeImpact failed (exit $LASTEXITCODE)" }

$arguments = @(
    '--inventory', '--repo', $Repository,
    '--bin', (Join-Path $Repository 'tests/AiDotNet.Tests/bin/Release/net10.0'),
    '--project', 'tests/AiDotNet.Tests/AiDotNetTests.csproj=AiDotNetTests')
$serving = Join-Path $Repository 'tests/AiDotNet.Serving.Tests/bin/Release/net10.0'
if (Test-Path -LiteralPath $serving) {
    $arguments += @('--bin', $serving, '--project', 'tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj=AiDotNet.Serving.Tests')
}
# Invoke-Shard.ps1 appends these exclusions to every shard filter, so no manifest filter names them.
$arguments += @('--excluded-category', 'HeavyTimeout', '--excluded-category', 'ModelPerformanceCensus',
    '--shards', $manifest, '--out', $OutFile)

& dotnet run --project $tool -c Release --no-build -- @arguments
exit $LASTEXITCODE
