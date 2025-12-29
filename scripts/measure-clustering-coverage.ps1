<#
.SYNOPSIS
Runs clustering integration tests with coverage and reports line coverage for src\Clustering.

.DESCRIPTION
Executes AiDotNet.Tests clustering integration tests using the coverlet collector and
summarizes line coverage for the src\Clustering\ subtree based on OpenCover sequence points.

.PARAMETER Configuration
Build configuration to test (default: Debug).

.PARAMETER Framework
Target framework to test (default: net8.0).

.PARAMETER Filter
Optional dotnet test filter (default: FullyQualifiedName~AiDotNet.Tests.IntegrationTests.Clustering).

.PARAMETER MinimumLineCoverage
Optional minimum line coverage percent threshold. Set to 0 to disable enforcement.
#>

[CmdletBinding()]
param(
  [string]$Configuration = "Debug",
  [string]$Framework = "net8.0",
  [string]$Filter = "FullyQualifiedName~AiDotNet.Tests.IntegrationTests.Clustering",
  [int]$MinimumLineCoverage = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectPath = Join-Path $repoRoot "tests\AiDotNet.Tests\AiDotNetTests.csproj"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultsDir = Join-Path $repoRoot ("TestResults\ClusteringCoverage\" + $timestamp)
New-Item -Path $resultsDir -ItemType Directory -Force | Out-Null

$args = @(
  "test", $projectPath,
  "-c", $Configuration,
  "-f", $Framework,
  "--no-build",
  "--nologo",
  "--logger", "trx;LogFileName=test-results.trx",
  "--results-directory", $resultsDir,
  "--settings", (Join-Path $repoRoot "coverlet.runsettings"),
  "--collect", "XPlat Code Coverage"
)

if ($Filter -and $Filter.Trim().Length -gt 0) {
  $args += @("--filter", $Filter)
}

Write-Host "dotnet $($args -join ' ')"
& dotnet @args 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) {
  throw "dotnet test failed with exit code $LASTEXITCODE."
}

$coverageFile = Get-ChildItem -Path $resultsDir -Recurse -Filter "coverage.opencover.xml" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $coverageFile) {
  throw "coverage.opencover.xml not found under $resultsDir."
}

[xml]$coverage = Get-Content $coverageFile.FullName
$files = @{}
foreach ($file in $coverage.SelectNodes("//File")) {
  $files[$file.uid] = $file.fullPath
}

$total = 0
$visited = 0
foreach ($seq in $coverage.SelectNodes("//SequencePoint")) {
  $fileId = $seq.fileid
  if (-not $files.ContainsKey($fileId)) {
    continue
  }

  $path = $files[$fileId]
  if ($path -like "*\src\Clustering\*") {
    $total++
    if ([int]$seq.vc -gt 0) {
      $visited++
    }
  }
}

if ($total -eq 0) {
  Write-Warning "No sequence points found for src\\Clustering."
  $coveragePercent = 0
} else {
  $coveragePercent = [Math]::Round(100.0 * $visited / $total, 2)
}

Write-Host ("Clustering coverage (line/sequence points): {0:N2}% ({1}/{2})" -f $coveragePercent, $visited, $total)

if ($MinimumLineCoverage -gt 0 -and $coveragePercent -lt $MinimumLineCoverage) {
  throw ("Coverage {0:N2}% is below the minimum threshold {1:N2}%." -f $coveragePercent, $MinimumLineCoverage)
}
