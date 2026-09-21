<# Sharded test execution, extracted from sonarcloud.yml.
   The inline run block hit GitHub's 21000-character expression limit (HTTP 422 on dispatch).
   Matrix values arrive as SHARD_* environment variables instead of being interpolated.
#>
Set-StrictMode -Version Latest

# Sanitize shard name: remove/replace all characters invalid in file paths
$shardName = "$($env:SHARD_NAME)"
$sanitizedShardName = $shardName -replace '[\\/:*?"<>|\s-]+', '_'
$results = Join-Path "TestResults" $sanitizedShardName
New-Item -Path $results -ItemType Directory -Force | Out-Null

# Pre-test resource snapshot. Cancelled-runner shards (Diffusion
# S-prefix / T-Z, Unit-03 diffusion, ModelFamily-NN,
# Generated Layers A-M/N-Z)
# die with `The runner has received a shutdown signal` 2-6 min into
# test execution. The dotnet test step exits before producing TRX
# output so we have no idea what was running at OOM time. Dump
# memory + disk + CPU info on entry so the next cancellation has
# forensic data.
Write-Host "=== Pre-test resource snapshot ==="
Write-Host "Memory:"
free -h
Write-Host "Disk:"
df -h /
Write-Host "Processors:"
nproc

# Serialize only the known multi-GB model shards; other shards retain default parallelism.
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
  'ModelFamily - Generated Layers N-O',
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
  'TopLevel - Audio/Onnx/Tokenization'
)
$heavyShard = ($heavyShards -contains $shardName) -or ($env:SHARD_HEAVY -eq 'true')

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
enum CoverageDisposition {
  Instrument
  Carried
  RunWithoutCoverage
}

$carriedSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
$runWithoutInstrumentationSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
try {
  $rawCarried = $env:COVERAGE_CARRIED
  if ($rawCarried) {
    foreach ($c in @($rawCarried | ConvertFrom-Json)) { [void] $carriedSet.Add([string] $c) }
  }
}
catch {
  $carriedSet.Clear()
  Write-Host "::warning::carried-shard list unreadable - instrumenting this shard"
}

try {
  $rawRunWithoutInstrumentation = $env:COVERAGE_RUN_WITHOUT_INSTRUMENTATION
  if ($rawRunWithoutInstrumentation) {
    foreach ($c in @($rawRunWithoutInstrumentation | ConvertFrom-Json)) {
      [void] $runWithoutInstrumentationSet.Add([string] $c)
    }
  }
}
catch {
  $runWithoutInstrumentationSet.Clear()
  Write-Host "::warning::run-without-instrumentation list unreadable - instrumenting this shard"
}

if ($carriedSet.Contains($shardName) -and $runWithoutInstrumentationSet.Contains($shardName)) {
  throw "coverage disposition for '$shardName' is both Carried and RunWithoutCoverage"
}

$coverageDisposition = [CoverageDisposition]::Instrument
if ("$($env:SHARD_FORCE_COVERAGE)" -eq 'true') {
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

$forcedCoverageRun = "$($env:SHARD_FORCE_COVERAGE)" -eq 'true'
$collectCoverage = if ($forcedCoverageRun) {
  $coverageDisposition -eq [CoverageDisposition]::Instrument
} else {
  -not ($heavyShard -or $measuresTiming)
}

if ($forcedCoverageRun -and $coverageDisposition -eq [CoverageDisposition]::Instrument -and
    ($heavyShard -or $measuresTiming)) {
  Write-Host 'collect_coverage_everywhere: coverage forced on for this shard to feed the test-impact map'
}
elseif ($coverageDisposition -eq [CoverageDisposition]::Carried) {
  Write-Host 'coverage carried forward from the previous map - not instrumenting this shard'
}
elseif ($coverageDisposition -eq [CoverageDisposition]::RunWithoutCoverage) {
  Write-Host 'previous map classified this memory/timing-bound shard as always-run - executing correctness tests without instrumentation'
}
Write-Host "Running shard '$shardName' (serialized: $serializeShard; coverage: $collectCoverage)"

# Build the argument list as a PowerShell array so the `--`
# separator and the runner args reach `dotnet test` as distinct
# tokens. Earlier we joined them into one string and pwsh's
# token splitter parsed `--` as a standalone switch that MSBuild
# then rejected with `MSB1001: Unknown switch`.
$dotnetArgs = @(
  'test', "$($env:SHARD_PROJECT)",
  '-c', 'Release',
  '--framework', "$($env:SHARD_FRAMEWORK)",
  '--no-build', '--no-restore',
  # HeavyTimeout and ModelPerformanceCensus have dedicated long-running/artifact lanes.
  '--filter', '($($env:SHARD_FILTER))&Category!=HeavyTimeout&Category!=ModelPerformanceCensus'
)
if ($collectCoverage) {
  $runSettings = 'coverlet.runsettings'
  if ($env:SHARD_COVERAGE_INCLUDE) {
    $runSettings = Join-Path $results 'coverlet.shard.runsettings'
    & ./.github/scripts/New-CoverageRunSettings.ps1 -Base coverlet.runsettings -IncludeDirectory (Join-Path (Get-Location).Path $env:SHARD_COVERAGE_INCLUDE) -OutFile $runSettings
    if ($LASTEXITCODE -ne 0) { throw "coverage runsettings for '$shardName' could not be written" }
    # IncludeDirectory alone does not instrument the worker; see Connect-WorkerCoverage.ps1.
    # Degrade, never abort: losing the whole map is worse than one always-run shard.
    try {
      $parentOut = Join-Path (Split-Path -Parent $env:SHARD_PROJECT) "bin/Release/$($env:SHARD_FRAMEWORK)"
      & ./tools/TestImpact/Connect-WorkerCoverage.ps1 -ParentDirectory $parentOut -WorkerDirectory $env:SHARD_COVERAGE_INCLUDE
      if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
    }
    catch {
      Write-Host "::warning::worker coverage not linked for '$shardName': $($_.Exception.Message)"
    }
  }
  $dotnetArgs += @(
    '--collect:XPlat Code Coverage',
    '--settings', $runSettings
  )
} else {
  # Coverage collection instruments every loaded assembly and keeps
  # extra per-test state in the VSTest process. On these serialized
  # big-model shards that overhead is the remaining memory multiplier
  # after xUnit parallelism and Server GC were disabled, so run them
  # as correctness-only shards and collect coverage from the lighter
  # matrix entries.
  if ($heavyShard) {
    Write-Host "Heavy shard: XPlat Code Coverage disabled to keep the runner under its memory envelope"
  } else {
    Write-Host "Timing shard: XPlat Code Coverage disabled because instrumentation would distort the ratio under test"
  }
}
# Detailed VSTest output still prints individual tests only on completion with the current
# xUnit adapter. The opt-in AIDOTNET_TEST_TRACE_FILE below is written by every source-generated
# test constructor before its model/component factory runs and streamed by the sampler, so an OOM has
# an attributable [test-start] class marker in the live log.
$consoleVerbosity = if ($heavyShard) { 'detailed' } else { 'normal' }
$hangTimeout = if ($env:SHARD_HANG_TIMEOUT) { $env:SHARD_HANG_TIMEOUT } else { '5min' }
if ($hangTimeout -cnotmatch '^[1-9][0-9]{0,2}(min|s|h)$') { throw "hangTimeout '$hangTimeout' is not a duration like 5min, 90s or 1h" }
$dotnetArgs += @(
  '--logger', 'trx;LogFileName=test-results.trx',
  '--logger', "console;verbosity=$consoleVerbosity",
  '--results-directory', $results,
  # Crash collection writes a Sequence file naming the in-flight test even when the host
  # dies before producing TRX. Mini dumps retain faulting stacks without exhausting disk;
  # they stay on the runner because this public workflow carries a license secret.
  '--blame-crash',
  '--blame-crash-dump-type', 'mini',
  '--blame-hang-timeout', $hangTimeout,
  '--blame-hang-dump-type', 'none'
)

# Use an absolute path because the test host runs from its output directory, not from the
# workspace. A relative path writes outside the artifact glob and silently loses the GPU
# launch journal and buffer-residency evidence.
& ./.github/scripts/Set-ShardEnvironment.ps1 -Json $env:SHARD_ENV
if ($LASTEXITCODE -ne 0) { throw "shard environment for '$shardName' was rejected" }

$diagDir = Join-Path (Get-Location).Path $results
New-Item -ItemType Directory -Force -Path $diagDir | Out-Null
$env:AIDOTNET_GPU_DIAGNOSTICS_DUMP = Join-Path $diagDir 'gpu-diagnostics.txt'
# Serialize heavy shards through xunit.runner.json; adapter CLI args are ignored.
if ($serializeShard) {
  $projDir = Split-Path -Parent "$($env:SHARD_PROJECT)"
  $runnerJson = Join-Path $projDir 'bin/Release/$($env:SHARD_FRAMEWORK)/xunit.runner.json'
  if (Test-Path $runnerJson) {
    $cfg = Get-Content $runnerJson -Raw | ConvertFrom-Json
    $cfg.parallelizeTestCollections = $false
    $cfg.maxParallelThreads = 1
    # Don't materialize every [Theory]'s data rows at discovery time —
    # the assembly discovers ~64k cases and pre-enumeration inflates the
    # retained-case baseline before a single test even runs.
    $cfg | Add-Member -NotePropertyName preEnumerateTheories -NotePropertyValue $false -Force
    # Silence xUnit discovery diagnostics on heavy shards. --TestCaseFilter is applied
    # AFTER xUnit discovers the whole assembly, so every shard enumerates all ~66.8k test
    # cases; with diagnosticMessages the runner prints one "Discovered [execution] test
    # case" line each. That output alone exceeds GitHub's per-job log cap: the
    # "NeuralNetworks A-F" job (90049783747) produced 57,102 discovery lines and its
    # entire 16.9 MB log covered just the first 2m44s of a 55-minute job, with ZERO test
    # results retained — the shard became impossible to diagnose from CI. The messages
    # are pure noise for a filtered shard run; keep them off here and leave the default
    # (on) in the checked-in xunit.runner.json for local debugging.
    $cfg.diagnosticMessages = $false
    $cfg.internalDiagnosticMessages = $false
    $cfg | ConvertTo-Json -Depth 10 | Set-Content $runnerJson -Encoding utf8
    Write-Host "Serialized heavy-shard runner config: $runnerJson"
  } else {
    # These shards are flagged heavy precisely because they OOM without
    # serialization. If the rewrite can't be applied, fail fast rather
    # than fall through to a parallel run that will OOM and waste the runner.
    Write-Host "ERROR: runner config not found at $runnerJson — cannot serialize heavy shard"
    exit 1
  }
  # The global Server GC (DOTNET_gcServer=1) reserves a heap segment per
  # core and collects lazily — high steady-state footprint chosen for
  # throughput under PARALLEL collections. This shard is now serial, so
  # we don't need that throughput; switch it to Workstation GC, which has
  # a much smaller footprint and collects eagerly under memory pressure.
  # (Confirmed necessary: with serialization alone the shard still OOM'd
  # ~70s into serial execution — the footprint, not parallelism, was the
  # remaining cause.)
  $env:DOTNET_gcServer = '0'
  Write-Host "Heavy shard: Workstation GC (DOTNET_gcServer=0) + serial collections"
}

# Stream memory and active-test markers in-process so evidence survives a runner OOM;
# a Start-Process child does not share the captured stdout handle.
$memSampler = $null
if ($heavyShard) {
  $testTrace = Join-Path $results 'active-test.trace'
  $env:AIDOTNET_TEST_TRACE_FILE = $testTrace
  $memSampler = [powershell]::Create()
  [void]$memSampler.AddScript({
    param($tracePath)

    # A HELD READ POSITION, NOT A RE-READ. `Get-Content | Select-Object -Skip N` still
    # opens the file and enumerates every line before discarding the first N, so each
    # two-second tick re-read the whole trace and the sampler's own I/O grew quadratically
    # across a long heavy shard -- on the one job already documented as memory-bound.
    #
    # The reader is opened once with FileShare.ReadWrite so the test host can keep
    # appending, and each tick reads only what has arrived since. Both handles are disposed
    # in the finally below, which runs when the runspace is stopped.
    $traceStream = $null
    $traceReader = $null
    try {
    while ($true) {
      # A THROW HERE ENDS THE SAMPLER SILENTLY. The runspace's IAsyncResult is discarded
      # at BeginInvoke and nobody calls EndInvoke, so any exception is swallowed whole:
      # sampling stops for the rest of the shard and the log reads like a clean run right
      # up to the OOM kill -- defeating the entire purpose of a sampler that exists to
      # show what memory did BEFORE the kill.
      #
      # Both known throwers are in this loop. The test host APPENDS to $tracePath while
      # this reads it, so a read can hit a sharing violation; and Select-String returns
      # $null when there is no match, making .Line a property access on $null.
      try {
        if ($null -eq $traceReader -and (Test-Path $tracePath)) {
          # ReadWrite sharing: the test host holds this file open for append.
          $traceStream = [System.IO.FileStream]::new(
            $tracePath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite)
          $traceReader = [System.IO.StreamReader]::new($traceStream)
        }

        if ($null -ne $traceReader) {
          # StreamReader returns null at end-of-stream and resumes from the same position
          # on the next tick once more has been appended, so this reads each line once.
          while ($null -ne ($line = $traceReader.ReadLine())) {
            [Console]::Out.WriteLine($line)
          }
        }
        $memLine = Select-String -Path /proc/meminfo -Pattern '^MemAvailable:' -ErrorAction Stop | Select-Object -First 1
        $avail = if ($memLine) { ($memLine.Line -replace '\s+', ' ').Trim() } else { 'MemAvailable: (unreadable)' }
        [Console]::Out.WriteLine("[memdiag $([DateTime]::UtcNow.ToString('HH:mm:ss'))] $avail")
        [Console]::Out.Flush()
      } catch {
        # Reported, never fatal: a sampler that dies quietly is worse than one that says
        # it stumbled and carries on.
        [Console]::Out.WriteLine("[memdiag] sampler iteration failed: $($_.Exception.Message)")
        [Console]::Out.Flush()
      }
      Start-Sleep -Seconds 2
    }
    }
    finally {
      # The runspace is torn down with Stop(), so this is where the handles are released.
      if ($null -ne $traceReader) { try { $traceReader.Dispose() } catch { } }
      if ($null -ne $traceStream) { try { $traceStream.Dispose() } catch { } }
    }
  })
  [void]$memSampler.AddArgument($testTrace)
  [void]$memSampler.BeginInvoke()
  Write-Host "Heavy shard: started live test-start + memory sampler (runspace)"
}

& dotnet @dotnetArgs
$exitCode = $LASTEXITCODE

if ($memSampler) { try { $memSampler.Stop(); $memSampler.Dispose() } catch {} }

# Post-test resource snapshot. If the runner survives this point,
# the test step finished naturally and the snapshot tells us what
# the high-water mark looked like.
Write-Host "=== Post-test resource snapshot ==="
Write-Host "Memory:"
free -h
Write-Host "Disk:"
df -h /
exit $exitCode
