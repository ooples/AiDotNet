# Shared execution and verification policy. Lists, coverage rules and defaults are unchanged.
enum CoverageDisposition {
  Instrument
  Carried
  RunWithoutCoverage
}

function Get-ShardCoveragePolicy {
    param([string] $ShardName, [bool] $Heavy, [string] $EventName, [string] $ForceCoverage,
        [string] $CarriedJson, [string] $RunWithoutCoverageJson, [string] $RollingJson, [switch] $Quiet)
    $heavyShards = @(
      'ModelFamily - Diffusion A-C',
      'ModelFamily - Diffusion D-I',
      'ModelFamily - Diffusion J-K',
      'ModelFamily - Diffusion L',
      'ModelFamily - Diffusion M A-Mi',
      'ModelFamily - Diffusion M Mochi-Model',
      'ModelFamily - Diffusion M Motion-MZ',
      'ModelFamily - Diffusion N-R',
      'ModelFamily - Diffusion SA-SD',
      'ModelFamily - Diffusion SE-SP',
      'ModelFamily - Diffusion Stable',
      'ModelFamily - Diffusion Step-Sync',
      'ModelFamily - Diffusion T-Z',
      'ModelFamily - Generated Layers A',
      'ModelFamily - Generated Layers B',
      'ModelFamily - Generated Layers C',
      'ModelFamily - Generated Layers D',
      'ModelFamily - Generated Layers E',
      'ModelFamily - Generated Layers F',
      'ModelFamily - Generated Layers G',
      'ModelFamily - Generated Layers H-I',
      'ModelFamily - Generated Layers J-L',
      'ModelFamily - Generated Layers M A-Medi',
      'ModelFamily - Generated Layers M MedS-Meta',
      'ModelFamily - Generated Layers M MetaV-Mu',
      'ModelFamily - Generated Layers N',
      'ModelFamily - Generated Layers O',
      'ModelFamily - Generated Layers P A,C,I,L,N,P',
      'ModelFamily - Generated Layers P E,H,O,R,S,U,W,Y',
      'ModelFamily - Generated Layers Q-R1',
      'ModelFamily - Generated Layers R2-SAI',
      'ModelFamily - Generated Layers SAL-SAM',
      'ModelFamily - Generated Layers SAN-Sig',
      'ModelFamily - Generated Layers Sil-Sq',
      'ModelFamily - Generated Layers St-Su',
      'ModelFamily - Generated Layers Sv-Sy',
      'ModelFamily - Generated Layers T A-Tac',
      'ModelFamily - Generated Layers T Tem-Tri',
      'ModelFamily - Generated Layers U',
      'ModelFamily - Generated Layers VA-VF',
      'ModelFamily - Generated Layers Video',
      'ModelFamily - Generated Layers Vi-VM',
      'ModelFamily - Generated Layers Vo-VR',
      'ModelFamily - Generated Layers W',
      'ModelFamily - Generated Layers X',
      'ModelFamily - Generated Layers Y-Z',
      'ModelFamily - NeuralNetworks A-F',
      'ModelFamily - NeuralNetworks G-L',
      'ModelFamily - NeuralNetworks M-N',
      'ModelFamily - NeuralNetworks O-R',
      'ModelFamily - NeuralNetworks S',
      'ModelFamily - NeuralNetworks T',
      'ModelFamily - NeuralNetworks U-Z',
      # TimeSeries/Activation/Loss: the double-precision Transformer/CNN forecasters
      # (Autoformer, Chronos, DeepANT, LSTM-VAE, ...) each fit the 60 s [Fact] budget in
      # isolation (~15 s) but TIME OUT under the shard's default 4-way collection parallelism —
      # their managed-engine forward parallelizes over all cores, so N classes in flight
      # oversubscribe the runner's 4 cores and stall well past 60 s. Serialize so each runs
      # uncontended with the whole machine (the contention is CPU, not paper-scale heaviness, so
      # these stay in the default gate rather than HeavyTimeout). #1706/#1305.
      'ModelFamily - TimeSeries/Activation/Loss',
      'ModelFamily - Code/Forecast/Segment/Survival',
      'Unit - 03a Diffusion Core/Schedulers/Encoding',
      'Unit - 03b1 Diffusion Preprocessors',
      'Unit - 03b2 Diffusion DDPM',
      'Unit - 03b3 Diffusion Control Guidance/Adapters',
      'Unit - 03b4 Diffusion ControlNet Core A',
      'Unit - 03b5 Diffusion ControlNet Core B',
      'Unit - 03c1 Diffusion Contracts A-C',
      'Unit - 03c2 Diffusion Contracts D-I',
      'Unit - 03c3 Diffusion Contracts J-R',
      'Unit - 03c4 Diffusion Contracts S',
      'Unit - 03c5 Diffusion Contracts T-Z',
      'Unit - 03c6 Editing Contracts A-I',
      'Unit - 03c7 Editing Contracts J-Z',
      'Unit - 03d1 Diffusion FastGen Image/VAE',
      'Unit - 03d2 Diffusion FastGen Consistency/Turbo',
      'Unit - 03d3 Diffusion FastGen Flow/AR',
      'Unit - 03d4 Diffusion FastGen Schedulers',
      'Unit - 03d5 Diffusion FastGen SDXL State',
      'Unit - 03d6 Diffusion FastGen Flux State',
      'Unit - 03d7 Diffusion FastGen Distillation',
      'Unit - 03d8 Diffusion New Conditioners',
      # Integration shards that instantiate paper-scale models (Document AI,
      # Finance, ComputerVision, NeuralNetworks, Diffusion, Video, MetaLearning,
      # etc.) — serialize like the ModelFamily shards to stay under the runner
      # memory envelope. The light integration letter-shards run parallel.
      'Integration C - ComputerVision Detection',
      'Integration C - ComputerVision Segmentation Models',
      'Integration C - ComputerVision Segmentation Contracts',
      # This clone-heavy shard previously exhausted a 16 GiB runner.
      'Integration C - Core',
      'Integration D',
      'Integration E-G',
      'Integration M',
      'Integration N-O',
      'Integration P-Q',
      'Integration R',
      'Integration S',
      'Integration T-Z',
      'Integration - Serial Perf',
      'TopLevel - FederatedLearning',
      'TopLevel - Audio/Onnx/Tokenization',
      'TTS Paper Tests'
    )
    $heavyShard = ($heavyShards -contains $shardName) -or ($Heavy)

    # Coverage instrumentation distorts the CPU/wall ratios asserted by timing shards.
    $performanceShards = @(
      'Integration - Serial Perf',
      'Integration - CPU Parallelism Probe'
    )
    $measuresTiming = $performanceShards -contains $shardName
    # EVERY heavy shard serializes. D-F used to be carved out of this on the grounds that
    # it needed "the repository's four-thread xUnit default" to fit the 45-minute budget, but
    # that default no longer exists: tests/AiDotNet.Tests/xunit.runner.json now sets
    # parallelizeTestCollections=false, so the carve-out bought nothing and only hid the fact
    # that the shard was running one-at-a-time anyway -- straight into the OOM killer. D-F is
    # now split into D/E/F, each serialized like every other heavy shard and each given a
    # 60-minute ceiling to pay for the lost concurrency.
    $serializeShard = $heavyShard
    # Heavy and timing shards normally skip coverage - the first for memory, the second
    # because instrumentation distorts the ratios under test. That is right for PR
    # validation. A dedicated map run instruments them only until it has positive evidence
    # that a shard cannot produce a digest safely; a previous-map alwaysRun classification
    # is that evidence, and keeps the complete correctness run mandatory without repeating
    # the instrumentation failure.
    #
    # The memory ceiling belongs to the PR matrix, not to a nightly job, so a dispatch can
    # ask for coverage everywhere. That run feeds the map; it is not meant to gate a PR.
    # A shard whose digest select-shards carried forward is not re-instrumented: its map
    # entry comes from the carried artifact, and instrumenting it anyway would only spend
    # the 3.21x overhead re-measuring what is byte-identical. Any parse failure of the
    # carried or run-without-instrumentation list degrades to Instrument, which is the
    # pre-incremental behaviour and is selection-safe.


    $carriedSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $runWithoutInstrumentationSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    try {
      $rawCarried = $CarriedJson
      if ($rawCarried) {
        foreach ($c in @($rawCarried | ConvertFrom-Json)) { [void] $carriedSet.Add([string] $c) }
      }
    }
    catch {
      $carriedSet.Clear()
      if (-not $Quiet) { Write-Host "::warning::carried-shard list unreadable - instrumenting this shard" }
    }

    try {
      $rawRunWithoutInstrumentation = $RunWithoutCoverageJson
      if ($rawRunWithoutInstrumentation) {
        foreach ($c in @($rawRunWithoutInstrumentation | ConvertFrom-Json)) {
          [void] $runWithoutInstrumentationSet.Add([string] $c)
        }
      }
    }
    catch {
      $runWithoutInstrumentationSet.Clear()
      if (-not $Quiet) { Write-Host "::warning::run-without-instrumentation list unreadable - instrumenting this shard" }
    }

    if ($carriedSet.Contains($shardName) -and $runWithoutInstrumentationSet.Contains($shardName)) {
      throw "coverage disposition for '$shardName' is both Carried and RunWithoutCoverage"
    }

    $coverageDisposition = [CoverageDisposition]::Instrument
    if ("$($ForceCoverage)" -eq 'true') {
      if ($carriedSet.Contains($shardName)) {
        $coverageDisposition = [CoverageDisposition]::Carried
      }
      elseif ($runWithoutInstrumentationSet.Contains($shardName)) {
        $coverageDisposition = [CoverageDisposition]::RunWithoutCoverage
      }
    }

    if ($coverageDisposition -ne [CoverageDisposition]::Instrument -and
        -not ($heavyShard -or $measuresTiming)) {
      throw "coverage disposition '$coverageDisposition' is invalid for coverage-producing shard '$shardName'"
    }

    $forcedCoverageRun = "$($ForceCoverage)" -eq 'true'
    # Pull requests never instrument: the test-impact map is fed only by the weekly coverage-everywhere
    # run, SonarCloud's coverage comes from master, and coverlet slowed every PR shard for a report
    # nobody gates on. Master pushes keep collecting it for Sonar.
    $pullRequest = "$($EventName)" -eq 'pull_request'
    $collectCoverage = if ($forcedCoverageRun) {
      $coverageDisposition -eq [CoverageDisposition]::Instrument
    } elseif ($pullRequest) {
      $false
    } else {
      -not ($heavyShard -or $measuresTiming)
    }

    # Rolling map refresh (tools/TestImpact/RollingShardMap.ps1). On the scheduled nightly, select-shards may
    # name a budgeted handful of heavy shards to instrument tonight, so the map's per-shard coverage is days
    # old rather than a week. Only then: an unset, empty or unreadable COVERAGE_ROLLING leaves the decision
    # above exactly as it was, as does every other event. A timing shard is never instrumented here - this
    # run is its correctness net - and a forced coverage run keeps its own dispositions.
    $rollingCoverage = $false
    if ($RollingJson -and -not $forcedCoverageRun -and -not $measuresTiming -and
        "$($EventName)" -eq 'schedule') {
      try {
        $rollingCoverage = @($RollingJson | ConvertFrom-Json | ForEach-Object { [string] $_ }) -ccontains $shardName
      }
      catch {
        if (-not $Quiet) { Write-Host '::warning::rolling coverage list unreadable - this shard runs without instrumentation, as on any other night' }
      }
    }
    if ($rollingCoverage -and -not $collectCoverage) {
      $collectCoverage = $true
      if (-not $Quiet) { Write-Host 'rolling map refresh: coverage on for this heavy shard tonight (TEST_IMPACT_ROLLING_MAP)' }
    }

    if ($forcedCoverageRun -and $coverageDisposition -eq [CoverageDisposition]::Instrument -and
        ($heavyShard -or $measuresTiming)) {
      if (-not $Quiet) { Write-Host 'collect_coverage_everywhere: coverage forced on for this shard to feed the test-impact map' }
    }
    elseif ($coverageDisposition -eq [CoverageDisposition]::Carried) {
      if (-not $Quiet) { Write-Host 'coverage carried forward from the previous map - not instrumenting this shard' }
    }
    elseif ($coverageDisposition -eq [CoverageDisposition]::RunWithoutCoverage) {
      if (-not $Quiet) { Write-Host 'previous map classified this memory/timing-bound shard as always-run - executing correctness tests without instrumentation' }
    }
    return [pscustomobject]@{ HeavyShard = $heavyShard; MeasuresTiming = $measuresTiming; CollectCoverage = [bool]$collectCoverage
        CoverageDisposition = $coverageDisposition.ToString(); ForcedCoverageRun = $forcedCoverageRun; RollingCoverage = $rollingCoverage
        PullRequest = [bool]$pullRequest }
}
