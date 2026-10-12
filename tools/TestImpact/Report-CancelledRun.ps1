<#
.SYNOPSIS
    Explains a cancelled validation run in one place a reader can trust.
.DESCRIPTION
    A cancelled job renders exactly like a failed one in the PR checks list, so an evicted run
    reads as a branch in ruins (#2131: ~22 genuine failures against ~759 cancellations across 20
    open PRs, fourteen of which had zero real failures). The run that is being cancelled cannot
    describe itself - GitHub stops its jobs too - so the report is published from OUTSIDE, after
    the run concludes, as a check run on the same head SHA.

    The report separates the two causes the issue identifies: a deliberate cancellation after a
    build failure (there is a failing job to point at) and concurrency eviction by a newer run
    (there is not, and the newer run is named instead).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, ParameterSetName = 'Report')] [string] $Repository,
    [Parameter(Mandatory, ParameterSetName = 'Report')] [long] $RunId,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-JobConclusionCounts {
    <#
        Groups a run's jobs by conclusion. Takes the job list already materialized by the
        caller so the counting is testable without the API.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Jobs)

    $counts = @{}
    foreach ($job in $Jobs) {
        $key = if ([string]::IsNullOrWhiteSpace($job.conclusion)) { 'none' } else { $job.conclusion }
        if (-not $counts.ContainsKey($key)) { $counts[$key] = 0 }
        $counts[$key] = $counts[$key] + 1
    }
    return $counts
}

function Get-CancellationSummary {
    <#
        Builds the check-run title and body from the run's jobs plus whichever newer run
        superseded it, if any. Pure string work: no API, no side effects.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Jobs,
        [Parameter(Mandatory)] [AllowEmptyString()] [string] $RunUrl,
        [AllowNull()] [object] $SupersededBy
    )

    $counts = Get-JobConclusionCounts -Jobs $Jobs
    $failed = @($Jobs | Where-Object { $_.conclusion -eq 'failure' })
    $cancelledCount = if ($counts.ContainsKey('cancelled')) { $counts['cancelled'] } else { 0 }

    $lines = [System.Collections.Generic.List[string]]::new()
    if ($failed.Count -gt 0) {
        $title = "$($failed.Count) job(s) failed; $cancelledCount cancelled on purpose"
        $lines.Add("This run was cancelled **because a job failed**. The cancelled checks are not")
        $lines.Add('independent failures - they were stopped to stop paying for the rest of the matrix.')
        $lines.Add('')
        $lines.Add('Fix these, in this order:')
        $lines.Add('')
        foreach ($job in $failed) {
            $lines.Add("- [``$($job.name)``]($($job.html_url))")
        }
    }
    elseif ($null -ne $SupersededBy) {
        $title = "Superseded, not broken: 0 failures, $cancelledCount cancelled"
        $lines.Add('Nothing in this run failed. A **newer run** of this workflow started on the same branch')
        $lines.Add('before this one was cancelled, the pattern concurrency eviction leaves; every cancelled')
        $lines.Add('check was stopped mid-flight.')
        $lines.Add('')
        $lines.Add("Read the newer run instead: [run $($SupersededBy.id) on ``$($SupersededBy.head_sha.Substring(0, [Math]::Min(9, $SupersededBy.head_sha.Length)))``]($($SupersededBy.html_url)).")
    }
    else {
        $title = "Cancelled with 0 failures: $cancelledCount checks stopped"
        $lines.Add('This run was cancelled and **no job in it failed**. The red-looking checks are')
        $lines.Add('cancellations, not failures. No newer run was found for this branch, so the')
        $lines.Add('cancellation came from outside the run (a manual cancel, or the queue).')
    }

    $lines.Add('')
    $lines.Add('| conclusion | jobs |')
    $lines.Add('|---|---:|')
    foreach ($key in ($counts.Keys | Sort-Object)) {
        $lines.Add("| $key | $($counts[$key]) |")
    }
    if (-not [string]::IsNullOrWhiteSpace($RunUrl)) {
        $lines.Add('')
        $lines.Add("Cancelled run: $RunUrl")
    }

    return [pscustomobject]@{
        Title   = $title
        Summary = ($lines -join "`n")
        Failed  = $failed.Count
        Cancelled = $cancelledCount
    }
}

function Invoke-SelfTest {
    <#
        The reporter only ever runs on a cancelled run, which is awkward to produce on demand, so
        the branch selection is exercised directly. Each arm asserts the title a reader sees.
    #>
    [CmdletBinding()]
    param()

    $deliberate = Get-CancellationSummary -RunUrl 'https://example/run/1' -SupersededBy $null -Jobs @(
        [pscustomobject]@{ name = 'Build'; conclusion = 'failure'; html_url = 'https://example/job/1' },
        [pscustomobject]@{ name = 'Tests A'; conclusion = 'cancelled'; html_url = 'https://example/job/2' },
        [pscustomobject]@{ name = 'Tests B'; conclusion = 'cancelled'; html_url = 'https://example/job/3' }
    )
    if ($deliberate.Title -ne '1 job(s) failed; 2 cancelled on purpose') {
        throw "Deliberate-cancellation title wrong: $($deliberate.Title)"
    }
    if ($deliberate.Summary -notmatch '\[`Build`\]') {
        throw 'The failing job must be named in the summary.'
    }

    $newer = [pscustomobject]@{
        id = 42; head_sha = 'abcdef1234567890'; html_url = 'https://example/run/42'
    }
    $evicted = Get-CancellationSummary -RunUrl 'https://example/run/2' -SupersededBy $newer -Jobs @(
        [pscustomobject]@{ name = 'Tests A'; conclusion = 'cancelled'; html_url = 'https://example/job/4' },
        [pscustomobject]@{ name = 'Build'; conclusion = 'success'; html_url = 'https://example/job/5' }
    )
    if ($evicted.Title -ne 'Superseded, not broken: 0 failures, 1 cancelled') {
        throw "Eviction title wrong: $($evicted.Title)"
    }
    if ($evicted.Summary -notmatch 'run 42') {
        throw 'The superseding run must be named in the summary.'
    }

    $orphan = Get-CancellationSummary -RunUrl '' -SupersededBy $null -Jobs @(
        [pscustomobject]@{ name = 'Tests A'; conclusion = 'cancelled'; html_url = 'https://example/job/6' }
    )
    if ($orphan.Title -ne 'Cancelled with 0 failures: 1 checks stopped') {
        throw "Orphan-cancellation title wrong: $($orphan.Title)"
    }

    Write-Host '[OK] Report-CancelledRun self-test: deliberate, evicted and orphaned arms all report distinctly.'
}

if ($SelfTest) {
    Invoke-SelfTest
    return
}

# ConvertFrom-Json turns GitHub's ISO-8601 timestamps into DateTime values; a string round-trip
# through the current culture would be locale-dependent, so compare instants directly.
function ConvertTo-Instant([object] $Value) {
    if ($Value -is [datetime]) { return [datetimeoffset] $Value.ToUniversalTime() }
    return [datetimeoffset]::Parse([string] $Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$runJson = gh api "repos/$Repository/actions/runs/$RunId" 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($runJson)) {
    Write-Host "::warning::Could not read run $RunId; no cancellation report published."
    return
}
$run = $runJson | ConvertFrom-Json

# Jobs are paginated at 100 and this repository's validation run has well over that, so a
# single page would undercount the cancellations the report exists to explain.
$jobsJson = gh api --paginate "repos/$Repository/actions/runs/$RunId/jobs?per_page=100" `
    --jq '.jobs[] | {name, conclusion, html_url}' 2>$null
# A failed request is missing evidence, not an empty run: reporting "0 failures" from it would hide
# the failed jobs this report exists to point at.
if ($LASTEXITCODE -ne 0) {
    Write-Host "::warning::Could not read jobs for run $RunId; no cancellation report published."
    return
}
$jobs = @()
if (-not [string]::IsNullOrWhiteSpace($jobsJson)) {
    $jobs = @($jobsJson -split "`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json })
}

# Concurrency eviction looks like this from outside: a newer run of the same workflow on the same
# branch that STARTED before this run was cancelled (eviction cancels the old run when the new one
# is queued). A run started after the cancellation cannot have caused it, so a later re-run of a
# manually cancelled run is not reported as its successor. The API exposes no concurrency group, so
# timing is the evidence; the summary words it as a newer run, not a proven cause.
$supersededBy = $null
$branch = [uri]::EscapeDataString([string] $run.head_branch)
$cancelledAt = ConvertTo-Instant $run.updated_at
$startedAt = ConvertTo-Instant $run.created_at
$candidatesJson = gh api `
    "repos/$Repository/actions/workflows/$($run.workflow_id)/runs?branch=$branch&per_page=20" `
    --jq '.workflow_runs[] | {id, head_sha, html_url, run_number, created_at}' 2>$null
if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($candidatesJson)) {
    $candidates = @($candidatesJson -split "`n" |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json })
    $supersededBy = $candidates |
        Where-Object {
            $created = ConvertTo-Instant $_.created_at
            $_.run_number -gt $run.run_number -and $created -ge $startedAt -and $created -le $cancelledAt
        } |
        Sort-Object run_number |
        Select-Object -First 1
}

$summary = Get-CancellationSummary -Jobs $jobs -RunUrl ([string] $run.html_url) -SupersededBy $supersededBy

Write-Host "Cancellation report for run ${RunId}: $($summary.Title)"
$summary.Summary | Out-Host
if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_STEP_SUMMARY)) {
    "## $($summary.Title)`n`n$($summary.Summary)" | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8
}

# The check run is the part a reviewer actually sees: it sits in the same checks list as the
# cancelled jobs it explains, on the same head SHA, and is neutral so it gates nothing.
$payload = [ordered]@{
    name       = 'Cancellation report'
    head_sha   = [string] $run.head_sha
    status     = 'completed'
    conclusion = 'neutral'
    output     = [ordered]@{
        title   = $summary.Title
        summary = $summary.Summary
    }
}
$payloadPath = Join-Path ([System.IO.Path]::GetTempPath()) "cancel-report-$RunId.json"
($payload | ConvertTo-Json -Depth 5) | Out-File -FilePath $payloadPath -Encoding utf8

gh api "repos/$Repository/check-runs" -X POST --input $payloadPath | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host '::warning::Could not publish the cancellation check run; the step summary still carries it.'
}
