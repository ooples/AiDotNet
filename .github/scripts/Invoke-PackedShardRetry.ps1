<#
.SYNOPSIS
    The targeted retry of PR-new failures, for a packed job: once per failed slice, with that slice's own
    environment and results directory.

.DESCRIPTION
    An unpacked job retries in three workflow steps (select the failures absent from the base commit's ledger,
    rerun them once, record the outcome). A packed job (TEST_IMPACT_REPACK=on; ShardPacking.ps1) holds several
    slices, and a retry must not mix them: a Conformance or Sweep slice retried without its environment window
    would walk the whole inventory in one test host. So for every slice whose tests failed this

      1. runs find-pr-new-failures.ps1 over that slice's TestResults/slot-N alone, writing its
         rerun-candidates.json there;
      2. reruns the candidates in a new process that first applies the slice's environment
         (Set-ShardEnvironment.ps1), into TestResults/slot-N/<slug>/retry - the layout an unpacked retry leaves;
      3. records rerunCandidateCount and rerunStepOutcome in the slice's shard-metadata.json.

    Only the compact ledger baseline is supported: it is one download for the whole job. With a legacy per-shard
    TRX baseline the workflow does not run this, and the slices' first results stand - more failures reported,
    never fewer, which is the safe direction the unpacked path takes when it has no baseline.

    Exits 1 when a retry ran and failed, as the unpacked rerun step does; the workflow step is
    continue-on-error, and the regression analysis reads the retry's TRX, not this exit code.
#>
[CmdletBinding(DefaultParameterSetName = 'Pack')]
param(
    [Parameter(ParameterSetName = 'Pack')] [string] $BaselineLedgerDirectory = 'retry-baseline-ledger',
    [Parameter(ParameterSetName = 'Pack')] [string] $ResultsRoot = 'TestResults',
    # One slice's rerun, in its own process so its environment reaches nothing else.
    [Parameter(Mandatory, ParameterSetName = 'Rerun')] [switch] $Rerun,
    [Parameter(Mandatory, ParameterSetName = 'Rerun')] [string] $Project,
    [Parameter(Mandatory, ParameterSetName = 'Rerun')] [string] $Framework,
    [Parameter(Mandatory, ParameterSetName = 'Rerun')] [string] $ResultsDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Rerun) {
    # Filter, name and environment arrive as environment variables: they hold quotes, pipes and parentheses.
    & (Join-Path $PSScriptRoot 'Set-ShardEnvironment.ps1') -Json "$($env:SHARD_ENV)"
    if ($LASTEXITCODE -ne 0) { throw "shard environment for '$($env:SHARD_NAME)' was rejected" }
    New-Item -Path $ResultsDirectory -ItemType Directory -Force | Out-Null
    # The same invocation as the workflow's "Rerun PR-new failures once" step.
    $retryArgs = @(
        'test', $Project,
        '--no-build', '--no-restore',
        '--configuration', 'Release',
        '--framework', $Framework,
        '--filter', "$($env:RETRY_FILTER)",
        '--results-directory', $ResultsDirectory,
        '--logger', 'trx;LogFileName=rerun-new-failures.trx',
        '--', 'xunit.parallelizeAssembly=false', 'xunit.parallelizeTestCollections=false'
    )
    & dotnet @retryArgs
    exit $LASTEXITCODE
}

$segments = @("$($env:SHARD_SEGMENTS)" | ConvertFrom-Json)
$ledger = Get-ChildItem -LiteralPath $BaselineLedgerDirectory -Recurse -Filter ledger.json -File -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $ledger) { throw "The downloaded retry baseline in '$BaselineLedgerDirectory' contains no ledger.json." }

$self = [Environment]::ProcessPath
$failedRetries = 0
for ($slot = 0; $slot -lt $segments.Count; $slot++) {
    $segment = $segments[$slot]
    $root = Join-Path $ResultsRoot "slot-$slot"
    $metadataFile = Join-Path $root 'shard-metadata.json'
    if (-not (Test-Path -LiteralPath $metadataFile -PathType Leaf)) { continue }
    $metadata = Get-Content -LiteralPath $metadataFile -Raw | ConvertFrom-Json
    if ([string] $metadata.testStepOutcome -cne 'failure') { continue }

    Write-Host "::group::Targeted retry for packed slice $($slot + 1): $($segment.name)"
    try {
        # The selector publishes rerun_count and filter as step outputs; read them from a file of this slice's own.
        $outputs = Join-Path $root 'retry-outputs.txt'
        Set-Content -LiteralPath $outputs -Value '' -Encoding utf8
        $savedOutput = $env:GITHUB_OUTPUT
        $env:GITHUB_OUTPUT = (Resolve-Path -LiteralPath $outputs).Path
        try {
            & $self -NoProfile -File (Join-Path $PSScriptRoot 'find-pr-new-failures.ps1') `
                -CurrentResultsPath $root -BaselineLedgerPath $ledger.FullName `
                -OutputFile (Join-Path $root 'rerun-candidates.json')
            $selected = $LASTEXITCODE
        }
        finally { $env:GITHUB_OUTPUT = $savedOutput }
        if ($selected -ne 0) {
            Write-Host "::warning::could not select PR-new failures for '$($segment.name)'; its first result stands"
            continue
        }
        $count = 0
        $filter = ''
        foreach ($line in @(Get-Content -LiteralPath $outputs)) {
            if ($line -cmatch '^rerun_count=(\d+)$') { $count = [int] $Matches[1] }
            elseif ($line -cmatch '^filter=(.*)$') { $filter = $Matches[1] }
        }
        Remove-Item -LiteralPath $outputs -Force
        $metadata.rerunCandidateCount = [string] $count
        if ($count -gt 0 -and $filter) {
            $optional = { param($Key) if ($segment.PSObject.Properties[$Key] -and $null -ne $segment.$Key) { $segment.$Key } else { $null } }
            $sliceEnv = & $optional 'env'
            $slug = [string] $segment.name -replace '[\\/:*?"<>|\s-]+', '_'
            $env:SHARD_NAME = [string] $segment.name
            $env:SHARD_ENV = if ($null -ne $sliceEnv) { ConvertTo-Json -InputObject $sliceEnv -Depth 4 -Compress } else { 'null' }
            $env:RETRY_FILTER = $filter
            & $self -NoProfile -File $PSCommandPath -Rerun -Project ([string] $segment.project) `
                -Framework ([string] $segment.framework) -ResultsDirectory (Join-Path (Join-Path $root $slug) 'retry')
            $metadata.rerunStepOutcome = if ($LASTEXITCODE -eq 0) { 'success' } else { $failedRetries++; 'failure' }
        }
        $metadata | ConvertTo-Json | Set-Content -LiteralPath $metadataFile -Encoding utf8
        Write-Host "$($segment.name): $count PR-new failing method(s); rerun $(if ($metadata.rerunStepOutcome) { $metadata.rerunStepOutcome } else { 'not needed' })"
    }
    catch {
        Write-Host "::warning::targeted retry for '$($segment.name)' did not complete ($($_.Exception.Message)); its first result stands"
    }
    finally { Write-Host '::endgroup::' }
}
exit $(if ($failedRetries -gt 0) { 1 } else { 0 })
