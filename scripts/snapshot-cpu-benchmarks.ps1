param(
  [string]$Source = "BenchmarkDotNet.Artifacts/results",
  [string]$DestinationRoot = "BenchmarkDotNet.Artifacts/results/snapshots",
  [string]$Stamp = (Get-Date -Format "yyyyMMdd-HHmmss")
)

if (-not (Test-Path -Path $Source)) {
  Write-Error "Source path not found: $Source"
  exit 1
}

New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
$destination = Join-Path -Path $DestinationRoot -ChildPath $Stamp
New-Item -ItemType Directory -Path $destination -Force | Out-Null

$patterns = @(
  "AiDotNetBenchmarkTests.*CpuComparisonBenchmarks-report.csv",
  "AiDotNetBenchmarkTests.*CpuComparisonBenchmarks-report.html",
  "AiDotNetBenchmarkTests.*CpuComparisonBenchmarks-report-github.md"
)

$files = foreach ($pattern in $patterns) {
  Get-ChildItem -Path $Source -Filter $pattern -File -ErrorAction SilentlyContinue
}

if (-not $files) {
  Write-Error "No benchmark reports found in $Source"
  exit 1
}

Copy-Item -Path $files.FullName -Destination $destination -Force
Write-Output $destination
