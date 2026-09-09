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
        [string] $Source = 'success',
        [string] $Select = 'success',
        [string] $Build = 'success',
        [string] $BuildCompat = 'success',
        [string] $Tests = 'success',
        [string] $ParameterSweep = 'success',
        [string] $ModelShape = 'success',
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
        -BuildResult $Build -BuildCompatResult $BuildCompat -TestsResult $Tests `
        -ParameterSweepResult $ParameterSweep -ModelShapeResult $ModelShape `
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
Invoke-GateCase -Name validation_ignores_quality_failure -Stage Validation `
    -CodeQL failure -Sonar failure -ExpectedExit 0
Invoke-GateCase -Name validation_known_test_failure -Stage Validation `
    -Tests failure -Verdict true -ExpectedExit 0
Invoke-GateCase -Name validation_unenforced_test_failure -Stage Validation `
    -Tests failure -Verdict false -ExpectedExit 1
Invoke-GateCase -Name validation_build_failure -Stage Validation -Build failure -ExpectedExit 1
Invoke-GateCase -Name validation_non_runtime -Stage Validation -RequiresValidation false `
    -Build skipped -BuildCompat skipped -Tests skipped -ParameterSweep skipped -ModelShape skipped `
    -Regression skipped -Aggregate skipped -SizeCheck skipped -Verdict false -ExpectedExit 0

# No reuse requires both current validation and current quality.
Invoke-GateCase -Name complete_current_success -ExpectedExit 0
Invoke-GateCase -Name complete_current_validation_failure -ValidationGate failure -ExpectedExit 1
Invoke-GateCase -Name complete_current_codeql_failure -CodeQL failure -ExpectedExit 1
Invoke-GateCase -Name complete_current_sonar_failure -Sonar failure -ExpectedExit 1

# Validation-only reuse skips the matrix but still requires quality and runtime promotion.
Invoke-GateCase -Name partial_reuse_success -ReuseScope Validation -Promotion success `
    -ValidationGate skipped -ExpectedExit 0
Invoke-GateCase -Name partial_reuse_quality_failure -ReuseScope Validation -Promotion success `
    -ValidationGate skipped -Sonar failure -ExpectedExit 1
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

if ($failures.Count -gt 0) {
    Write-Host 'CI Gate mode proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host 'CI Gate mode proof passed (validation/current/partial/complete and fail-closed controls).'
exit 0
