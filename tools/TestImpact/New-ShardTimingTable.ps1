<#
.SYNOPSIS
    Builds the shard timing table that Layer 2 (ShardPacking.ps1) predicts a slice's test time from.

.DESCRIPTION
    The table holds seconds per test class per shard, and a wall factor per shard, as the median over the
    newest test-outcome ledgers. The regression analysis of every pipeline run already publishes that ledger
    (artifact test-outcome-ledger-<sha>, kept 90 days) with each test's name, shard and TRX duration, so this
    downloads a few megabytes and reads nothing else from a run but its job list.

    Which runs. The newest completed sonarcloud.yml runs that published a ledger:
      * pull_request and merge_group runs first, up to -MaxLedgers. They are what gets repacked and they never
        collect coverage, so their durations are the ones a packed job will see;
      * push and schedule runs, up to -MaxInstrumentedLedgers, only to time what no pull request run has: their
        light shards run under coverlet and are slower.
    workflow_dispatch runs are never used (a coverage-everywhere run instruments everything).

    Wall factor. TRX durations leave out what a test host does between tests (fixtures, model construction in
    a constructor): a shard's test step took between 1.0 and 2.5 times its TRX sum in run 43612258685. The
    factor comes from the duration of the 'Run tests (sharded) with coverage' step of the shard's job, or from
    testWallSeconds in the ledger where a packed job recorded it per slice.

    Runs daily from .github/workflows/shard-timing-table.yml, which saves the table to the Actions cache of
    the default branch; a pull request's Build job restores it from there in a second or two.

.PARAMETER SourcesFile
    Offline: a JSON array of { ledger, jobs, instrumented, source } naming files already on disk (jobs is
    optional), newest first, instead of asking GitHub. For tests and for measuring against saved runs.
#>
[CmdletBinding(DefaultParameterSetName = 'Online')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Online')] [string] $Repository,
    [Parameter(ParameterSetName = 'Online')] [string] $Workflow = 'sonarcloud.yml',
    [Parameter(ParameterSetName = 'Online')] [ValidateRange(1, 20)] [int] $MaxLedgers = 5,
    [Parameter(ParameterSetName = 'Online')] [ValidateRange(0, 20)] [int] $MaxInstrumentedLedgers = 2,
    [Parameter(ParameterSetName = 'Online')] [ValidateRange(1, 1000)] [int] $MaxRunsScanned = 400,
    [Parameter(ParameterSetName = 'Online')] [string] $WorkDirectory = '',
    [Parameter(Mandatory, ParameterSetName = 'Offline')] [string] $SourcesFile,
    [string] $OutFile = 'shard-timing-table.json',
    [string] $SummaryFile = $env:GITHUB_STEP_SUMMARY
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/ShardPacking.ps1"

function Get-ShardWallSeconds {
    <# Seconds of each unpacked test job's test step, by shard slug. #>
    param([AllowNull()] [object[]] $Jobs)
    $wall = @{}
    $timings = Get-ShardJobTimings -Jobs @($Jobs)
    foreach ($name in $timings.Keys) {
        if ($null -ne $timings[$name].Step) { $wall[(ConvertTo-ShardSlug $name)] = $timings[$name].Step * 60 }
    }
    return $wall
}

$sources = [Collections.Generic.List[object]]::new()
if ($PSCmdlet.ParameterSetName -ceq 'Offline') {
    foreach ($entry in @(Get-Content -LiteralPath $SourcesFile -Raw | ConvertFrom-Json)) {
        $jobs = if ($entry.PSObject.Properties['jobs'] -and $entry.jobs) { @(Get-Content -LiteralPath ([string] $entry.jobs) -Raw | ConvertFrom-Json) } else { @() }
        $sources.Add([pscustomobject]@{
                Ledger = [string] $entry.ledger; Jobs = $jobs
                Instrumented = [bool] ($entry.PSObject.Properties['instrumented'] -and $entry.instrumented)
                Source = [string] $entry.source
            })
    }
}
else {
    if (-not $WorkDirectory) { $WorkDirectory = Join-Path ([IO.Path]::GetTempPath()) "shard-timing-$([Guid]::NewGuid().ToString('N'))" }
    New-Item -ItemType Directory -Path $WorkDirectory -Force | Out-Null
    $runsJson = & gh run list --repo $Repository --workflow $Workflow --status completed --limit $MaxRunsScanned `
        --json 'databaseId,event,conclusion,createdAt'
    if ($LASTEXITCODE -ne 0) { throw "could not list $Workflow runs of $Repository" }
    $plain = 0
    $instrumented = 0
    foreach ($run in @($runsJson | ConvertFrom-Json)) {
        if ($plain -ge $MaxLedgers -and $instrumented -ge $MaxInstrumentedLedgers) { break }
        $isPlain = [string] $run.event -cin $script:ShardPackEvents
        if (-not $isPlain -and [string] $run.event -cnotin @('push', 'schedule')) { continue }
        # A cancelled pull request run was superseded before its regression analysis ran; it has no ledger to
        # offer, and there are too many of them to ask each one. A push or scheduled run is asked whatever became
        # of it: the nightly usually ends cancelled after publishing its ledger, and a shard that did not finish
        # is left out of an observation anyway.
        if ($isPlain -and [string] $run.conclusion -cnotin @('success', 'failure')) { continue }
        if ($isPlain -and $plain -ge $MaxLedgers) { continue }
        if (-not $isPlain -and $instrumented -ge $MaxInstrumentedLedgers) { continue }

        $id = [string] $run.databaseId
        $directory = Join-Path $WorkDirectory $id
        & gh run download $id --repo $Repository --pattern 'test-outcome-ledger-*' --dir $directory 2>$null | Out-Null
        $ledger = if ($LASTEXITCODE -eq 0) {
            Get-ChildItem -LiteralPath $directory -Recurse -Filter ledger.json -File -ErrorAction SilentlyContinue | Select-Object -First 1
        }
        else { $null }
        $global:LASTEXITCODE = 0
        if (-not $ledger) { continue }

        # 50 a page: a job list of 100 from a run this size has answered 502.
        $jobs = @()
        try {
            $pages = & gh api --paginate --slurp "repos/$Repository/actions/runs/$id/jobs?per_page=50" | ConvertFrom-Json
            if ($LASTEXITCODE -eq 0) { $jobs = @($pages | ForEach-Object { $_.jobs }) }
        }
        catch { Write-Host "run $id job list unavailable ($($_.Exception.Message)); its shards get no wall factor from it" }
        $global:LASTEXITCODE = 0

        $sources.Add([pscustomobject]@{ Ledger = $ledger.FullName; Jobs = $jobs; Instrumented = (-not $isPlain)
                Source = "run $id ($($run.event), $(([DateTimeOffset] $run.createdAt).ToString('yyyy-MM-dd')))" })
        if ($isPlain) { $plain++ } else { $instrumented++ }
        Write-Host "using $($sources[$sources.Count - 1].Source)"
    }
}

if ($sources.Count -eq 0) {
    # No table is a state the selection handles (nothing is packed); an empty one would only hide that.
    Write-Host '::warning::no test-outcome ledger was found; no timing table was written'
    exit 0
}

$observations = foreach ($source in $sources) {
    try {
        Read-ShardTimingObservation -LedgerPath $source.Ledger -WallSeconds (Get-ShardWallSeconds $source.Jobs) `
            -Instrumented $source.Instrumented -Source $source.Source
    }
    catch { Write-Host "::warning::$($source.Source) was left out: $($_.Exception.Message)" }
}
$observations = @($observations | Where-Object { $null -ne $_ -and $_.shards.Count -gt 0 })
if ($observations.Count -eq 0) {
    Write-Host '::warning::no ledger held a finished shard; no timing table was written'
    exit 0
}

$table = New-ShardTimingTable -Observations $observations
ConvertTo-Json -InputObject $table -Depth 6 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8
$classes = 0
$withFactor = 0
foreach ($shard in $table.shards.Values) {
    $classes += $shard.classes.Count
    if ($shard.Contains('wallFactor')) { $withFactor++ }
}
$lines = @(
    '### Shard timing table', '',
    "**$($table.shards.Count)** shard(s) and $classes test class timing(s) from $($observations.Count) ledger(s); $withFactor shard(s) have a measured wall factor, the rest use $($table.defaultWallFactor).", '',
    '| Source | Instrumented | Shards |', '|---|---|---|'
)
foreach ($observation in $observations) { $lines += "| $($observation.source) | $($observation.instrumented) | $($observation.shards.Count) |" }
if ($SummaryFile) { ($lines -join "`n") | Out-File -FilePath $SummaryFile -Append -Encoding utf8 }
Write-Host "shard timing table: $($table.shards.Count) shard(s), $classes class timing(s), $($observations.Count) ledger(s) -> $OutFile"
exit 0
