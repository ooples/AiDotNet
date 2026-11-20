<#
.SYNOPSIS
    After the test matrix: what repacking predicted against what the jobs actually took, and - for a packed
    run - the job list rewritten so the tools that read jobs by shard name still can.

.DESCRIPTION
    Report (the default). Reads the shard-pack-plan the Build job published (Select-AffectedTests.ps1) and the
    run's jobs, and writes predicted against actual test minutes per job to the step summary and to -OutFile.
    With TEST_IMPACT_REPACK=report the run was one job per shard, so a would-be packed job's actual time is the
    sum of its slices' test steps and the packed runner-minutes are an estimate, labelled as one. This is how
    the prediction is judged before anything depends on it. Report-only: it exits 0 whatever it finds.

    -ExpandJobs. Rewrites -JobsFile with every packed job replaced by one job per slice
    (Expand-PackedShardJobs), keeping the original beside it as -RawJobsFile. A file with no packed job is left
    byte for byte as it was.
#>
[CmdletBinding(DefaultParameterSetName = 'Report')]
param(
    [Parameter(Mandatory)] [string] $JobsFile,
    [Parameter(Mandatory, ParameterSetName = 'Report')] [string] $PlanFile,
    [Parameter(ParameterSetName = 'Report')] [string] $OutFile = 'shard-pack-report.json',
    [Parameter(ParameterSetName = 'Report')] [string] $SummaryFile = $env:GITHUB_STEP_SUMMARY,
    [Parameter(Mandatory, ParameterSetName = 'Expand')] [switch] $ExpandJobs,
    [Parameter(ParameterSetName = 'Expand')] [string] $RawJobsFile = 'current-jobs-raw.json'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/ShardPacking.ps1"

function Read-RunJobs {
    param([string] $Path)
    $document = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($null -ne $document -and $document -isnot [array] -and $document.PSObject.Properties['jobs']) { return @($document.jobs) }
    return @($document)
}

if ($ExpandJobs) {
    $jobs = @(Read-RunJobs $JobsFile)
    $expanded = @(Expand-PackedShardJobs -Jobs $jobs)
    $slices = @($expanded | Where-Object { $_.PSObject.Properties['packedIn'] })
    if ($slices.Count -eq 0) {
        Write-Host 'no packed job reported its slices; the job list is unchanged'
        exit 0
    }
    Copy-Item -LiteralPath $JobsFile -Destination $RawJobsFile -Force
    ConvertTo-Json -InputObject $expanded -Depth 8 | Set-Content -LiteralPath $JobsFile -Encoding utf8
    Write-Host "expanded $(@($slices | ForEach-Object { $_.packedIn } | Sort-Object -Unique).Count) packed job(s) into $($slices.Count) slice job(s); the original list is $RawJobsFile"
    exit 0
}

try {
    if (-not (Test-Path -LiteralPath $PlanFile -PathType Leaf)) {
        Write-Host "no shard-pack-plan at '$PlanFile': this run computed no packing"
        exit 0
    }
    $plan = Get-Content -LiteralPath $PlanFile -Raw | ConvertFrom-Json
    $report = Get-ShardPackReport -Plan $plan -Jobs @(Read-RunJobs $JobsFile)
    ConvertTo-Json -InputObject $report -Depth 6 | Set-Content -LiteralPath $OutFile -Encoding utf8
    $summary = Get-ShardPackReportSummary -Report $report
    if ($SummaryFile) { "`n$summary" | Out-File -FilePath $SummaryFile -Append -Encoding utf8 }
    Write-Host ($summary -replace '\*\*', '')
}
catch { Write-Host "::warning::shard repacking report unavailable: $($_.Exception.Message)" }
exit 0
