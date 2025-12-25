<# 
.SYNOPSIS
Runs AiDotNet tests locally using lightweight sharding for faster feedback.

.DESCRIPTION
This script is intended for local development where you want to saturate CPU cores by running
multiple `dotnet test` invocations in parallel. It defaults to the "fast" suite:
- Excludes Category=GPU, Category=Integration, Category=Stress
- Targets net8.0
- Skips build (assumes you built already)

.PARAMETER Configuration
Build configuration to test (default: Release).

.PARAMETER Framework
Target framework to test (default: net8.0).

.PARAMETER MaxParallel
Maximum concurrent `dotnet test` processes (default: number of logical processors).

.PARAMETER IncludeIntegration
Include integration tests (Category=Integration).

.PARAMETER IncludeGpu
Include GPU tests (Category=GPU). Note: requires GPU hardware.

.PARAMETER IncludeStress
Include stress tests (Category=Stress). Note: slow and may be flaky on shared machines.

.PARAMETER Build
Build the test projects before running tests.

.PARAMETER UseRunSettings
Use `tests/local.runsettings` (MaxCpuCount) to increase in-process parallelism.
#>

[CmdletBinding()]
param(
  [string]$Configuration = "Release",
  [string]$Framework = "net8.0",
  [int]$MaxParallel = [Environment]::ProcessorCount,
  [switch]$IncludeIntegration,
  [switch]$IncludeGpu,
  [switch]$IncludeStress,
  [switch]$Build,
  [switch]$UseRunSettings
)

$ErrorActionPreference = "Stop"

function New-TestShard {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Project,
    [Parameter(Mandatory = $true)][string]$Filter
  )

  [pscustomobject]@{
    Name = $Name
    Project = $Project
    Filter = $Filter
  }
}

function Get-CategoryFilter {
  $parts = @()

  if (-not $IncludeGpu) { $parts += "Category!=GPU" }
  if (-not $IncludeIntegration) { $parts += "Category!=Integration" }
  if (-not $IncludeStress) { $parts += "Category!=Stress" }

  if ($parts.Count -eq 0) { return "" }
  return ($parts -join "&")
}

function Join-FilterParts {
  param([string[]]$Parts)

  $filtered = @(
    $Parts |
      Where-Object { $_ -and $_.Trim().Length -gt 0 } |
      ForEach-Object { $_.Trim() }
  )

  if ($filtered.Count -eq 0) { return "" }
  return ($filtered -join "&")
}

function New-UnitCoreFilter {
  param(
    [string]$CategoryFilter,
    [string[]]$Roots,
    [string[]]$ExcludeSegments
  )

  $filters = New-Object System.Collections.Generic.List[string]
  foreach ($root in $Roots) {
    $parts = @($CategoryFilter, "FullyQualifiedName~$root")
    foreach ($segment in $ExcludeSegments) {
      $parts += "FullyQualifiedName!~$root.$segment"
    }

    $filters.Add((Join-FilterParts -Parts $parts))
  }

  return ($filters -join "|")
}

function New-UnitShardFilter {
  param(
    [string]$CategoryFilter,
    [string[]]$Roots,
    [string[]]$Segments
  )

  $filters = New-Object System.Collections.Generic.List[string]
  foreach ($root in $Roots) {
    foreach ($segment in $Segments) {
      $filters.Add((Join-FilterParts -Parts @($CategoryFilter, "FullyQualifiedName~$root.$segment")))
    }
  }

  return ($filters -join "|")
}

function New-NamespaceFilter {
  param(
    [string]$CategoryFilter,
    [string[]]$Namespaces
  )

  $filters = New-Object System.Collections.Generic.List[string]
  foreach ($ns in $Namespaces) {
    $filters.Add((Join-FilterParts -Parts @($CategoryFilter, "FullyQualifiedName~$ns")))
  }

  return ($filters -join "|")
}

function Invoke-DotNetTest {
  param(
    [Parameter(Mandatory = $true)]$Shard,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$Configuration,
    [Parameter(Mandatory = $true)][string]$Framework,
    [switch]$UseRunSettings
  )

  $projectPath = Join-Path $RepoRoot $Shard.Project
  $resultsDir = Join-Path $RepoRoot ("TestResults\LocalShards\" + ($Shard.Name -replace "[:/\\]", "-"))
  New-Item -Path $resultsDir -ItemType Directory -Force | Out-Null

  $args = @(
    "test", $projectPath,
    "-c", $Configuration,
    "-f", $Framework,
    "--no-build",
    "--nologo",
    "--logger", "trx;LogFileName=test-results.trx",
    "--results-directory", $resultsDir
  )

  if ($UseRunSettings) {
    $runsettings = Join-Path $RepoRoot "tests\local.runsettings"
    if (Test-Path $runsettings) {
      $args += @("--settings", $runsettings)
    }
  }

  if ($Shard.Filter -and $Shard.Filter.Trim().Length -gt 0) {
    $args += @("--filter", $Shard.Filter)
  }

  Write-Host "[$($Shard.Name)] dotnet $($args -join ' ')"
  & dotnet @args 2>&1 | Out-Host
  return [pscustomobject]@{ Name = $Shard.Name; ExitCode = $LASTEXITCODE }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
  if ($Build) {
    Write-Host "Building test projects..."
    dotnet build "$repoRoot\tests\AiDotNet.Tests\AiDotNetTests.csproj" -c $Configuration -f $Framework --nologo
    dotnet build "$repoRoot\tests\AiDotNet.Tensors.Tests\AiDotNet.Tensors.Tests.csproj" -c $Configuration -f $Framework --nologo
    if ($Framework -like "net8*") {
      dotnet build "$repoRoot\tests\AiDotNet.Serving.Tests\AiDotNet.Serving.Tests.csproj" -c $Configuration -f $Framework --nologo
    }
  }

  $categoryFilter = Get-CategoryFilter

  # Shards: keep these broad and cheap to maintain. Adjust if you see one shard dominating runtime.
  $shards = New-Object System.Collections.Generic.List[object]

  # AiDotNet.Tests (large) - shard UnitTests by sub-areas for faster local runs.
  $unitNamespaceRoots = @(
    "AiDotNet.Tests.UnitTests",
    "AiDotNetTests.UnitTests"
  )
  $unitExcludedSegments = @(
    "ActivationFunctions",
    "Attention",
    "Autodiff",
    "AutoML",
    "Data",
    "Diagnostics",
    "Diffusion",
    "Encoding",
    "FeatureSelectors",
    "FitDetectors",
    "FitnessCalculators",
    "Genetics",
    "Helpers",
    "Inference",
    "Interpretability",
    "JitCompiler",
    "KnowledgeDistillation",
    "LearningRateSchedulers",
    "LinearAlgebra",
    "Logging",
    "LossFunctions",
    "MetaLearning",
    "MixedPrecision",
    "ModelCompression",
    "NestedLearning",
    "NeuralNetworks",
    "Optimizers",
    "RAG",
    "Regularization",
    "ReinforcementLearning",
    "RetrievalAugmentedGeneration",
    "Serving",
    "TimeSeries",
    "Tokenization",
    "TransferLearning"
  )
  $unitCoreFilter = New-UnitCoreFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -ExcludeSegments $unitExcludedSegments
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 01 Core" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter $unitCoreFilter))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 02 Activation/Attention" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("ActivationFunctions", "Attention"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 03 Autodiff/AutoML/Data" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("Autodiff", "AutoML", "Data"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 04 Diagnostics/Diffusion/Encoding" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("Diagnostics", "Diffusion", "Encoding"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 05 Feature/Fit/Fitness" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("FeatureSelectors", "FitDetectors", "FitnessCalculators"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 06 Genetics/Helpers/Inference" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("Genetics", "Helpers", "Inference"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 07 Interpretability/JIT/KD" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("Interpretability", "JitCompiler", "KnowledgeDistillation"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 08 Schedulers/LA/Logging/Loss" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("LearningRateSchedulers", "LinearAlgebra", "Logging", "LossFunctions"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 09 Meta/Mixed/Compression" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("MetaLearning", "MixedPrecision", "ModelCompression"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 10 NN/Optimizers/RAG" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("NeuralNetworks", "Optimizers", "RAG"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 11 Regularization/RL/RAG2" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("Regularization", "ReinforcementLearning", "RetrievalAugmentedGeneration"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Unit - 12 Serving/TimeSeries/Token/Transfer" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-UnitShardFilter -CategoryFilter $categoryFilter -Roots $unitNamespaceRoots -Segments @("Serving", "TimeSeries", "Tokenization", "TransferLearning"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 13 InferenceOptimization" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter "$categoryFilter&FullyQualifiedName~AiDotNet.Tests.InferenceOptimization"))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 14 PromptEngineering" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter "$categoryFilter&FullyQualifiedName~AiDotNet.Tests.PromptEngineering"))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 15 Recovery/Concurrency" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter "$categoryFilter&FullyQualifiedName~AiDotNet.Tests.Concurrency|$categoryFilter&FullyQualifiedName~AiDotNet.Tests.Recovery"))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 16 ActivationFunctions" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-NamespaceFilter -CategoryFilter $categoryFilter -Namespaces @("AiDotNet.Tests.ActivationFunctions"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 17 Factories" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-NamespaceFilter -CategoryFilter $categoryFilter -Namespaces @("AiDotNet.Tests.Factories"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 18 MetaLearning" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-NamespaceFilter -CategoryFilter $categoryFilter -Namespaces @("AiDotNet.Tests.MetaLearning"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 19 Pruning" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-NamespaceFilter -CategoryFilter $categoryFilter -Namespaces @("AiDotNet.Tests.Pruning"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 20 Tokenization" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-NamespaceFilter -CategoryFilter $categoryFilter -Namespaces @("AiDotNet.Tests.Tokenization"))))
  $shards.Add((New-TestShard -Name "AiDotNet.Tests - Other - 21 EndToEndTests" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter (New-NamespaceFilter -CategoryFilter $categoryFilter -Namespaces @("AiDotNet.Tests.EndToEndTests"))))

  if ($IncludeIntegration) {
    $shards.Add((New-TestShard -Name "AiDotNet.Tests - Integration" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter "FullyQualifiedName~AiDotNet.Tests.IntegrationTests"))
  }

  if ($IncludeGpu) {
    $shards.Add((New-TestShard -Name "AiDotNet.Tests - GPU" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter "Category=GPU"))
  }

  if ($IncludeStress) {
    $shards.Add((New-TestShard -Name "AiDotNet.Tests - Stress" -Project "tests\AiDotNet.Tests\AiDotNetTests.csproj" -Filter "Category=Stress"))
  }

  # Other test projects
  $shards.Add((New-TestShard -Name "AiDotNet.Tensors.Tests" -Project "tests\AiDotNet.Tensors.Tests\AiDotNet.Tensors.Tests.csproj" -Filter $categoryFilter))
  if ($Framework -like "net8*") {
    $shards.Add((New-TestShard -Name "AiDotNet.Serving.Tests" -Project "tests\AiDotNet.Serving.Tests\AiDotNet.Serving.Tests.csproj" -Filter $categoryFilter))
  }

  $shards = $shards.ToArray()

  # Remove redundant leading/trailing '&' when categoryFilter is empty.
  foreach ($s in $shards) {
    if ($null -ne $s.Filter) {
      $s.Filter = ($s.Filter -replace "^[&]+", "") -replace "[&]+$", ""
      $s.Filter = ($s.Filter -replace "&&+", "&").Trim()
      if ([string]::IsNullOrWhiteSpace($s.Filter)) {
        $s.Filter = $null
      }
    }
  }

  Write-Host "Running $($shards.Count) shard(s) with MaxParallel=$MaxParallel ..."

  if (-not $Build) {
    $uniqueProjects = $shards | Select-Object -ExpandProperty Project -Unique
    foreach ($project in $uniqueProjects) {
      $projectPath = Join-Path $repoRoot $project
      if (-not (Test-Path $projectPath)) {
        continue
      }

      $projectName = Split-Path $projectPath -Leaf
      $outputDir = Join-Path (Split-Path $projectPath -Parent) ("bin\\$Configuration\\$Framework")
      $expectedDll = Join-Path $outputDir ($projectName -replace "\\.csproj$", ".dll")

      if (-not (Test-Path $expectedDll)) {
        Write-Host "Pre-building missing test output for $project ..."
        dotnet build $projectPath -c $Configuration -f $Framework --nologo
      }
    }
  }

  if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Warning "PowerShell 7+ is recommended for parallel sharding. Running shards sequentially (install 'pwsh' for parallel runs)."
    $results = @()
    foreach ($shard in $shards) {
      if ($UseRunSettings) {
        $results += Invoke-DotNetTest -Shard $shard -RepoRoot $repoRoot -Configuration $Configuration -Framework $Framework -UseRunSettings
      } else {
        $results += Invoke-DotNetTest -Shard $shard -RepoRoot $repoRoot -Configuration $Configuration -Framework $Framework
      }
    }
  } else {
    # Note: ForEach-Object -Parallel runs in separate runspaces where functions like
    # Invoke-DotNetTest are not available. The code below intentionally duplicates
    # the test invocation logic from Invoke-DotNetTest for parallel execution.
    $results = $shards | ForEach-Object -Parallel {
      $projectPath = Join-Path $using:repoRoot $_.Project
      $resultsDir = Join-Path $using:repoRoot ("TestResults\LocalShards\" + ($_.Name -replace "[:/\\]", "-"))
      New-Item -Path $resultsDir -ItemType Directory -Force | Out-Null

      $args = @(
        "test", $projectPath,
        "-c", $using:Configuration,
        "-f", $using:Framework,
        "--no-build",
        "--nologo",
        "--logger", "trx;LogFileName=test-results.trx",
        "--results-directory", $resultsDir
      )

      if ($using:UseRunSettings) {
        $runsettings = Join-Path $using:repoRoot "tests\local.runsettings"
        if (Test-Path $runsettings) {
          $args += @("--settings", $runsettings)
        }
      }

      if ($_.Filter -and $_.Filter.Trim().Length -gt 0) {
        $args += @("--filter", $_.Filter)
      }

      Write-Host "[$($_.Name)] dotnet $($args -join ' ')"
      & dotnet @args 2>&1 | Out-Host
      [pscustomobject]@{ Name = $_.Name; ExitCode = $LASTEXITCODE }
    } -ThrottleLimit $MaxParallel
  }

  $ordered = @($results) | Where-Object { $_ -is [psobject] -and $_.PSObject.Properties.Match('Name').Count -gt 0 -and $_.PSObject.Properties.Match('ExitCode').Count -gt 0 } | Sort-Object Name
  $failed = $ordered | Where-Object { $_.ExitCode -ne 0 }

  Write-Host ""
  Write-Host "Shard results:"
  $ordered | ForEach-Object { Write-Host ("- {0}: exit {1}" -f $_.Name, $_.ExitCode) }

  if ($failed.Count -gt 0) {
    Write-Error ("{0} shard(s) failed." -f $failed.Count)
  }
} finally {
  Pop-Location
}
