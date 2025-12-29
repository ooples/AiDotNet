<#
.SYNOPSIS
Runs meta-learning integration tests with coverage and reports method coverage.
#>

[CmdletBinding()]
param(
  [string]$Configuration = "Release",
  [string]$Framework = "net471",
  [string]$Project = "tests\\AiDotNet.Tests\\AiDotNetTests.csproj",
  [string]$Filter = "FullyQualifiedName~IntegrationTests.MetaLearning",
  [string]$ResultsDirectory = "TestResults\\MetaLearningCoverage",
  [string]$NamespacePrefix = "AiDotNet.MetaLearning"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectPath = Join-Path $repoRoot $Project
$resultsDir = Join-Path $repoRoot $ResultsDirectory
New-Item -Path $resultsDir -ItemType Directory -Force | Out-Null

$settingsPath = Join-Path $repoRoot "coverlet.runsettings"

$args = @(
  "test", $projectPath,
  "-c", $Configuration,
  "-f", $Framework,
  "--collect", "XPlat Code Coverage",
  "--results-directory", $resultsDir,
  "--nologo"
)

if (Test-Path $settingsPath) {
  $args += @("--settings", $settingsPath)
}

if ($Filter -and $Filter.Trim().Length -gt 0) {
  $args += @("--filter", $Filter)
}

Write-Host "dotnet $($args -join ' ')"
& dotnet @args 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$coverageFile = Get-ChildItem -Path $resultsDir -Recurse -Filter "coverage.opencover.xml" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $coverageFile) {
  Write-Error "Coverage report not found under $resultsDir."
  exit 1
}

[xml]$coverage = Get-Content $coverageFile.FullName

$classes = $coverage.SelectNodes("//Class")
$metaClasses = New-Object System.Collections.Generic.List[System.Xml.XmlElement]

foreach ($class in $classes) {
  $fullName = $class.fullName
  if (-not $fullName) {
    $fullName = $class.fullname
  }
  if ($fullName -and $fullName.StartsWith($NamespacePrefix)) {
    $metaClasses.Add($class) | Out-Null
  }
}

$methods = New-Object System.Collections.Generic.List[System.Xml.XmlElement]
foreach ($class in $metaClasses) {
  $methodNodes = $class.Methods.Method
  if ($methodNodes) {
    foreach ($method in $methodNodes) {
      $methods.Add($method) | Out-Null
    }
  }
}

if ($methods.Count -eq 0) {
  Write-Warning "No methods found for namespace prefix '$NamespacePrefix'."
  Write-Host "Coverage file: $($coverageFile.FullName)"
  exit 0
}

$covered = 0
foreach ($method in $methods) {
  $sequencePoints = $method.SequencePoints.SequencePoint
  $hit = $false

  if ($sequencePoints) {
    foreach ($sp in $sequencePoints) {
      if ([int]$sp.vc -gt 0) {
        $hit = $true
        break
      }
    }
  } elseif ($method.Summary) {
    if ([int]$method.Summary.visitedSequencePoints -gt 0) {
      $hit = $true
    }
  }

  if ($hit) {
    $covered++
  }
}

$total = $methods.Count
$percent = if ($total -gt 0) { [math]::Round(($covered * 100.0) / $total, 2) } else { 0 }
Write-Host ("Meta-learning method coverage ({0}): {1}/{2} ({3}%)" -f $NamespacePrefix, $covered, $total, $percent)
Write-Host "Coverage file: $($coverageFile.FullName)"
