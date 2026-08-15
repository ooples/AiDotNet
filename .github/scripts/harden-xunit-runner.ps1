<#
.SYNOPSIS
    Serializes a test job's xUnit execution and shrinks its GC footprint so the job
    survives a 16 GB GitHub runner.

.DESCRIPTION
    THE SINGLE SOURCE OF THIS FIX. It used to live inline in the test-net10-sharded
    job only, which is why every heavy job added afterwards was born broken: the
    parameter sweeps and the model shape conformance shards were both authored
    without it and both died on their runs with

        ##[error]The runner has received a shutdown signal.
        ##[error]Process completed with exit code 143.

    ~85 seconds into execution, against 45-60 minute budgets. Exit 143 with a
    shutdown signal, well inside the time budget, is the runner being killed for
    memory -- not a timeout and not a cancellation. Any job that constructs models
    from this library must call this script.

    Four separate multipliers have to come off, and each one was individually
    confirmed necessary on a real failing job:

    1. COLLECTION PARALLELISM. xUnit defaults maxParallelThreads to ProcessorCount,
       so it runs many test CLASSES at once and each big-model class holds a
       multi-GB model plus Adam state for the whole class. Note the inline
       `-- xunit.MaxParallelThreads=1` VSTest argument is SILENTLY IGNORED by the
       xunit.runner.visualstudio adapter -- only xunit.runner.json in the build
       output is honored, which is why this script rewrites the file.

    2. THEORY PRE-ENUMERATION. Materializing every [Theory]'s data rows at
       discovery inflates the retained-case baseline before a single test runs.

    3. DISCOVERY DIAGNOSTICS. --filter is applied AFTER xUnit discovers the whole
       assembly, so every job enumerates all ~72k cases and prints one line each.
       That output alone exceeds GitHub's per-job log cap: one job produced 57,102
       discovery lines and its entire 16.9 MB log covered just the first 2m44s,
       with ZERO test results retained -- the shard became impossible to diagnose.

    4. SERVER GC. DOTNET_gcServer=1 reserves a heap segment per core and collects
       lazily -- a high steady-state footprint chosen for throughput under PARALLEL
       collections. A serialized job does not need that throughput. This one is not
       optional: with serialization alone a shard still died ~70s into serial
       execution, so the footprint, not the parallelism, was the remaining cause.

.PARAMETER RunnerJson
    Path to xunit.runner.json in the BUILD OUTPUT (bin/<config>/<tfm>/), not the
    checked-in copy. Rewriting the source file would change local debugging runs.

.PARAMETER SkipWorkstationGc
    Leave DOTNET_gcServer alone. For a job that is already known to fit in memory
    and wants Server GC throughput.

.OUTPUTS
    Exits 1 if RunnerJson is absent. That is deliberate: a caller only invokes this
    because it OOMs without it, so falling through to a parallel run would waste a
    runner to reach the same failure.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $RunnerJson,

    [switch] $SkipWorkstationGc
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $RunnerJson)) {
    Write-Host "ERROR: runner config not found at $RunnerJson -- cannot serialize this job."
    Write-Host "       Expected it in the build output. Did the build step run with the same"
    Write-Host "       configuration and target framework this path was built from?"
    exit 1
}

$cfg = Get-Content $RunnerJson -Raw | ConvertFrom-Json

# 1. one test collection in flight at a time
$cfg | Add-Member -NotePropertyName parallelizeTestCollections -NotePropertyValue $false -Force
$cfg | Add-Member -NotePropertyName maxParallelThreads -NotePropertyValue 1 -Force

# 2. do not materialize [Theory] rows at discovery
$cfg | Add-Member -NotePropertyName preEnumerateTheories -NotePropertyValue $false -Force

# 3. keep ~72k discovery lines out of the log so results survive the cap.
#    Left ON in the checked-in xunit.runner.json for local debugging.
$cfg | Add-Member -NotePropertyName diagnosticMessages -NotePropertyValue $false -Force
$cfg | Add-Member -NotePropertyName internalDiagnosticMessages -NotePropertyValue $false -Force

$cfg | ConvertTo-Json -Depth 10 | Set-Content $RunnerJson -Encoding utf8
Write-Host "[OK] Serialized xUnit runner config: $RunnerJson"

# 4. Workstation GC. Exported two ways on purpose: the process env var covers a
#    caller that runs dotnet test later in THIS same pwsh process (env vars are
#    process-wide, so an inline `& harden-xunit-runner.ps1` still gets it), and
#    GITHUB_ENV covers a caller whose test run is a SEPARATE step, where the
#    process var would be gone.
if (-not $SkipWorkstationGc) {
    $env:DOTNET_gcServer = '0'
    if ($env:GITHUB_ENV) {
        Add-Content -Path $env:GITHUB_ENV -Value 'DOTNET_gcServer=0'
    }
    Write-Host '[OK] Workstation GC (DOTNET_gcServer=0) + serial collections'
}
