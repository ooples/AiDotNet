<#
.SYNOPSIS
    Executes the typed validation and complete gate in every reuse mode.
#>
[CmdletBinding()]
param([string] $Gate = (Join-Path $PSScriptRoot 'Assert-CiGate.ps1'))

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$failures = [System.Collections.Generic.List[string]]::new()

function Invoke-GateCase {
    param(
        [string] $Name,
        [string] $Stage = 'Complete',
        [string] $ReuseScope = 'None',
        [string] $RequiresValidation = 'true',
        [string] $RequiresTests = 'true',
        [string] $Source = 'success',
        [string] $Select = 'success',
        [string] $Build = 'success',
        [string] $BuildCompat = 'success',
        [string] $Tests = 'success',
        [string] $Regression = 'success',
        [string] $Verdict = 'true',
        [string] $Aggregate = 'success',
        [string] $SizeCheck = 'success',
        [string] $Promotion = 'skipped',
        [string] $CodeQL = 'success',
        [string] $Sonar = 'success',
        [string] $ValidationGate = 'success',
        [int] $ExpectedExit
    )

    & $Gate -Stage $Stage -ReuseScope $ReuseScope -SourceResult $Source `
        -RequiresValidation $RequiresValidation -SelectResult $Select `
        -RequiresTests $RequiresTests `
        -BuildResult $Build -BuildCompatResult $BuildCompat -TestsResult $Tests `
        -RegressionAnalysisResult $Regression -VerdictEnforced $Verdict `
        -AggregateAnalysisResult $Aggregate -SizeCheckResult $SizeCheck `
        -PromotionResult $Promotion -CodeQLResult $CodeQL -SonarResult $Sonar `
        -ValidationGateResult $ValidationGate *> $null
    if ($LASTEXITCODE -ne $ExpectedExit) {
        [void] $failures.Add("$Name expected exit $ExpectedExit, got $LASTEXITCODE")
    }
}

# Validation evidence is independent of quality jobs.
Invoke-GateCase -Name validation_runtime_success -Stage Validation -ExpectedExit 0
Invoke-GateCase -Name validation_selective_auxiliary_skip -Stage Validation -ExpectedExit 0
# The parameter-enumeration sweep and the model-shape conformance windows used to be bespoke jobs
# the gate required by name. They are shards now, so a missing or failed one is caught where every
# other shard is: Import-PullRequestShardArtifacts reports a listed shard with no artifact, and
# New-ShardMapCertificate refuses outcomes that do not cover the map's shard universe. Both land on
# test-regression-analysis, which this gate still requires to succeed.
Invoke-GateCase -Name validation_auxiliary_only -Stage Validation `
    -RequiresTests false -Tests skipped -Verdict false -ExpectedExit 0
Invoke-GateCase -Name validation_required_tests_missing -Stage Validation `
    -Tests skipped -Verdict false -ExpectedExit 1
Invoke-GateCase -Name validation_ignores_quality_failure -Stage Validation `
    -CodeQL failure -Sonar failure -ExpectedExit 0
Invoke-GateCase -Name validation_known_test_failure -Stage Validation `
    -Tests failure -Verdict true -ExpectedExit 0
Invoke-GateCase -Name validation_unenforced_test_failure -Stage Validation `
    -Tests failure -Verdict false -ExpectedExit 1
Invoke-GateCase -Name validation_build_failure -Stage Validation -Build failure -ExpectedExit 1
Invoke-GateCase -Name validation_non_runtime -Stage Validation -RequiresValidation false `
    -Build skipped -BuildCompat skipped -Tests skipped `
    -Regression skipped -Aggregate skipped -SizeCheck skipped -Verdict false -ExpectedExit 0

# No reuse requires current validation and current CodeQL.
Invoke-GateCase -Name complete_current_success -ExpectedExit 0
Invoke-GateCase -Name complete_current_validation_failure -ValidationGate failure -ExpectedExit 1
Invoke-GateCase -Name complete_current_codeql_failure -CodeQL failure -ExpectedExit 1
Invoke-GateCase -Name complete_current_codeql_cancelled -CodeQL cancelled -ExpectedExit 1
# SonarCloud is advisory: no Sonar outcome may block, and none may stand in for a required job.
foreach ($sonarOutcome in 'failure', 'cancelled', 'timed_out', 'skipped') {
    Invoke-GateCase -Name "complete_current_sonar_$sonarOutcome" -Sonar $sonarOutcome -ExpectedExit 0
}
Invoke-GateCase -Name complete_current_sonar_success_does_not_cover_codeql -CodeQL failure -ExpectedExit 1

# Validation-only reuse skips the matrix but still requires CodeQL and runtime promotion.
Invoke-GateCase -Name partial_reuse_success -ReuseScope Validation -Promotion success `
    -ValidationGate skipped -ExpectedExit 0
Invoke-GateCase -Name partial_reuse_quality_failure -ReuseScope Validation -Promotion success `
    -ValidationGate skipped -CodeQL failure -ExpectedExit 1
Invoke-GateCase -Name partial_reuse_sonar_failure_is_advisory -ReuseScope Validation -Promotion success `
    -ValidationGate skipped -Sonar failure -ExpectedExit 0
Invoke-GateCase -Name partial_reuse_missing_promotion -ReuseScope Validation -Promotion skipped `
    -ValidationGate skipped -ExpectedExit 1
Invoke-GateCase -Name partial_non_runtime_reuse -ReuseScope Validation -RequiresValidation false `
    -Promotion skipped -ValidationGate skipped -ExpectedExit 0

# Complete reuse needs no repeated quality job, but runtime data still has to be promoted.
Invoke-GateCase -Name complete_reuse_success -ReuseScope Complete -Promotion success `
    -CodeQL skipped -Sonar skipped -ValidationGate skipped -ExpectedExit 0
Invoke-GateCase -Name complete_reuse_missing_promotion -ReuseScope Complete -Promotion skipped `
    -CodeQL skipped -Sonar skipped -ValidationGate skipped -ExpectedExit 1
Invoke-GateCase -Name complete_non_runtime_reuse -ReuseScope Complete -RequiresValidation false `
    -Promotion skipped -CodeQL skipped -Sonar skipped -ValidationGate skipped -ExpectedExit 0
Invoke-GateCase -Name source_failure_always_blocks -ReuseScope Complete -Source failure `
    -Promotion success -CodeQL skipped -Sonar skipped -ValidationGate skipped -ExpectedExit 1

# #2131: a select job that never got a runner reports the undocumented conclusion 'abandoned'. The gate
# must fail on it (no job ran) without throwing, and so must any conclusion it does not recognise.
Invoke-GateCase -Name validation_select_abandoned -Stage Validation -Select abandoned -ExpectedExit 1
Invoke-GateCase -Name validation_select_unrecognized -Stage Validation -Select mystery_state -ExpectedExit 1
Invoke-GateCase -Name complete_codeql_abandoned -CodeQL abandoned -ExpectedExit 1

# The summary must say when nothing failed and jobs merely did not run, and must not say it when one failed.
$summaryPath = [System.IO.Path]::GetTempFileName()
try {
    & $Gate -Stage Validation -ReuseScope None -SourceResult success -RequiresValidation true `
        -SelectResult abandoned -BuildResult skipped -BuildCompatResult skipped -TestsResult skipped `
        -RegressionAnalysisResult skipped -AggregateAnalysisResult skipped -SizeCheckResult skipped `
        -SummaryFile $summaryPath *> $null
    if ($LASTEXITCODE -ne 1) {
        [void] $failures.Add("summary_abandoned_select: expected exit 1, got $LASTEXITCODE")
    }
    $text = Get-Content -LiteralPath $summaryPath -Raw
    if ($text -notmatch 'No required job reported a failure') {
        [void] $failures.Add('summary_names_incomplete: an abandoned select job was not described as not having run')
    }
    & $Gate -Stage Validation -ReuseScope None -SourceResult success -RequiresValidation true `
        -SelectResult success -BuildResult failure -SummaryFile $summaryPath *> $null
    if ($LASTEXITCODE -ne 1) {
        [void] $failures.Add("summary_build_failure: expected exit 1, got $LASTEXITCODE")
    }
    $text = Get-Content -LiteralPath $summaryPath -Raw
    if ($text -match 'No required job reported a failure') {
        [void] $failures.Add('summary_names_incomplete: a real build failure was described as not having run')
    }
    # A Skipped required job with nothing incomplete upstream is a real failure (required work not done), so
    # it must not be excused. The abandoned-select case above cannot show this: there the select job is the
    # incomplete one, and the skipped jobs downstream of it are excused for that reason.
    & $Gate -Stage Validation -ReuseScope None -SourceResult success -RequiresValidation true `
        -SelectResult success -BuildResult skipped -BuildCompatResult success -TestsResult success `
        -RegressionAnalysisResult success -VerdictEnforced true -AggregateAnalysisResult success `
        -SizeCheckResult success -SummaryFile $summaryPath *> $null
    if ($LASTEXITCODE -ne 1) {
        [void] $failures.Add("summary_skipped_required_job: expected exit 1, got $LASTEXITCODE")
    }
    $text = Get-Content -LiteralPath $summaryPath -Raw
    if ($text -match 'No required job reported a failure') {
        [void] $failures.Add('summary_skipped_required_job: a skipped required job was described as not having run')
    }
    # An empty result means the gate was called without that job's input: a wiring fault. It fails, it is not
    # excused as a job that did not run, and the summary says what it is.
    & $Gate -Stage Validation -ReuseScope None -SourceResult success -RequiresValidation true `
        -SelectResult '' -BuildResult skipped -BuildCompatResult skipped -TestsResult skipped `
        -RegressionAnalysisResult skipped -AggregateAnalysisResult skipped -SizeCheckResult skipped `
        -SummaryFile $summaryPath *> $null
    if ($LASTEXITCODE -ne 1) {
        [void] $failures.Add("summary_empty_result: expected exit 1, got $LASTEXITCODE")
    }
    $text = Get-Content -LiteralPath $summaryPath -Raw
    if ($text -match 'No required job reported a failure') {
        [void] $failures.Add('summary_empty_result: a missing result was excused as a job that did not run')
    }
    if ($text -notmatch 'select-shards reported no result') {
        [void] $failures.Add('summary_empty_result: the summary did not name the job with no result')
    }
    # A timeout can be a hang the change introduced, and a startup failure an invalid workflow it edited:
    # neither may be excused as "did not run".
    foreach ($implicated in 'timed_out', 'startup_failure') {
        & $Gate -Stage Validation -ReuseScope None -SourceResult success -RequiresValidation true `
            -SelectResult success -BuildResult $implicated -BuildCompatResult success -TestsResult success `
            -RegressionAnalysisResult success -VerdictEnforced true -AggregateAnalysisResult success `
            -SizeCheckResult success -SummaryFile $summaryPath *> $null
        # Every other job passed, so the exit code is decided by the build's outcome alone.
        if ($LASTEXITCODE -ne 1) {
            [void] $failures.Add("validation_build_${implicated}: expected exit 1, got $LASTEXITCODE")
        }
        $text = Get-Content -LiteralPath $summaryPath -Raw
        if ($text -match 'No required job reported a failure') {
            [void] $failures.Add("summary_names_incomplete: a build that reported '$implicated' was excused as not having run")
        }
    }
    & $Gate -Stage Validation -ReuseScope None -SourceResult success -RequiresValidation true `
        -SelectResult mystery_state -BuildResult skipped -BuildCompatResult skipped -TestsResult skipped `
        -RegressionAnalysisResult skipped -AggregateAnalysisResult skipped -SizeCheckResult skipped `
        -SummaryFile $summaryPath *> $null
    if ($LASTEXITCODE -ne 1) {
        [void] $failures.Add("summary_names_unrecognized: expected exit 1, got $LASTEXITCODE")
    }
    $text = Get-Content -LiteralPath $summaryPath -Raw
    if ($text -notmatch '\*\*Gate FAILED\*\*' -or $text -notmatch "select-shards reported an unrecognized conclusion 'mystery_state', treated as not passing") {
        [void] $failures.Add('summary_names_unrecognized: an unrecognized select conclusion was not named in a failed summary')
    }
}
finally {
    Remove-Item -LiteralPath $summaryPath -ErrorAction SilentlyContinue
}

Invoke-GateCase -Name deferred_validation_is_not_passing -Source failure -Select skipped `
    -Build skipped -BuildCompat skipped -Tests skipped `
    -Regression skipped -Aggregate skipped -SizeCheck skipped -CodeQL skipped -Sonar skipped `
    -ValidationGate skipped -ExpectedExit 1

if ($failures.Count -gt 0) {
    Write-Host 'CI Gate mode proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host 'CI Gate mode proof passed (validation/current/partial/complete and fail-closed controls).'
exit 0
