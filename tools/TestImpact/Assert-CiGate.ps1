<#
.SYNOPSIS
    Evaluates the validation-only or complete CI gate with typed states.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [ValidateSet('Validation', 'Complete')] [string] $Stage,
    [Parameter(Mandatory)] [ValidateSet('None', 'Validation', 'Complete')] [string] $ReuseScope,
    [Parameter(Mandatory)] [string] $SourceResult,
    [Parameter(Mandatory)] [string] $RequiresValidation,
    [string] $SelectResult = 'skipped',
    [string] $BuildResult = 'skipped',
    [string] $BuildCompatResult = 'skipped',
    [string] $TestsResult = 'skipped',
    [string] $ParameterSweepResult = 'skipped',
    [string] $ModelShapeResult = 'skipped',
    [string] $RegressionAnalysisResult = 'skipped',
    [string] $VerdictEnforced = 'false',
    [string] $AggregateAnalysisResult = 'skipped',
    [string] $SizeCheckResult = 'skipped',
    [string] $PromotionResult = 'skipped',
    [string] $CodeQLResult = 'skipped',
    [string] $SonarResult = 'skipped',
    [string] $ValidationGateResult = 'skipped',
    [string] $SummaryFile = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum CiGateStage {
    Validation
    Complete
}

enum CiValidationReuseScope {
    None
    Validation
    Complete
}

enum CiJobConclusion {
    Unknown
    Success
    Failure
    Cancelled
    Skipped
    Neutral
    TimedOut
    ActionRequired
    StartupFailure
    Stale
}

function ConvertTo-CiJobConclusion {
    param([AllowEmptyString()] [string] $Value, [string] $Name)
    switch -CaseSensitive ($Value) {
        'success' { return [CiJobConclusion]::Success }
        'failure' { return [CiJobConclusion]::Failure }
        'cancelled' { return [CiJobConclusion]::Cancelled }
        'skipped' { return [CiJobConclusion]::Skipped }
        'neutral' { return [CiJobConclusion]::Neutral }
        'timed_out' { return [CiJobConclusion]::TimedOut }
        'action_required' { return [CiJobConclusion]::ActionRequired }
        'startup_failure' { return [CiJobConclusion]::StartupFailure }
        'stale' { return [CiJobConclusion]::Stale }
        '' { return [CiJobConclusion]::Unknown }
        default { throw "$Name has unsupported job conclusion '$Value'" }
    }
}

function ConvertTo-RequiredBoolean {
    param([string] $Value, [string] $Name)
    if ($Value -ceq 'true') { return $true }
    if ($Value -ceq 'false') { return $false }
    throw "$Name must be the JSON boolean text true or false"
}

function Add-RequiredSuccess {
    param(
        [System.Collections.Generic.List[object]] $Requirements,
        [string] $Name,
        [CiJobConclusion] $Conclusion
    )
    [void] $Requirements.Add([pscustomobject]@{ Name = $Name; Conclusion = $Conclusion })
}

$gateStage = [CiGateStage] $Stage
$reuse = [CiValidationReuseScope] $ReuseScope
$requiresRuntimeValidation = ConvertTo-RequiredBoolean $RequiresValidation 'RequiresValidation'
$verdictIsEnforced = ConvertTo-RequiredBoolean $VerdictEnforced 'VerdictEnforced'

$source = ConvertTo-CiJobConclusion $SourceResult 'SourceResult'
$select = ConvertTo-CiJobConclusion $SelectResult 'SelectResult'
$build = ConvertTo-CiJobConclusion $BuildResult 'BuildResult'
$buildCompat = ConvertTo-CiJobConclusion $BuildCompatResult 'BuildCompatResult'
$tests = ConvertTo-CiJobConclusion $TestsResult 'TestsResult'
$parameterSweep = ConvertTo-CiJobConclusion $ParameterSweepResult 'ParameterSweepResult'
$modelShape = ConvertTo-CiJobConclusion $ModelShapeResult 'ModelShapeResult'
$regression = ConvertTo-CiJobConclusion $RegressionAnalysisResult 'RegressionAnalysisResult'
$aggregate = ConvertTo-CiJobConclusion $AggregateAnalysisResult 'AggregateAnalysisResult'
$sizeCheck = ConvertTo-CiJobConclusion $SizeCheckResult 'SizeCheckResult'
$promotion = ConvertTo-CiJobConclusion $PromotionResult 'PromotionResult'
$codeql = ConvertTo-CiJobConclusion $CodeQLResult 'CodeQLResult'
$sonar = ConvertTo-CiJobConclusion $SonarResult 'SonarResult'
$validationGate = ConvertTo-CiJobConclusion $ValidationGateResult 'ValidationGateResult'

$requirements = [System.Collections.Generic.List[object]]::new()
Add-RequiredSuccess $requirements 'validation-source' $source

if ($gateStage -eq [CiGateStage]::Validation) {
    if ($reuse -ne [CiValidationReuseScope]::None) {
        throw 'the validation gate cannot mint new evidence from reused evidence'
    }
    Add-RequiredSuccess $requirements 'select-shards' $select
    if ($requiresRuntimeValidation) {
        Add-RequiredSuccess $requirements 'build' $build
        Add-RequiredSuccess $requirements 'build-compat' $buildCompat
        Add-RequiredSuccess $requirements 'parameter-enumeration-sweep' $parameterSweep
        Add-RequiredSuccess $requirements 'model-shape-conformance-windows' $modelShape
        Add-RequiredSuccess $requirements 'test-regression-analysis' $regression
        Add-RequiredSuccess $requirements 'ci-test-analysis' $aggregate
        Add-RequiredSuccess $requirements 'size-check' $sizeCheck
        if (-not $verdictIsEnforced) {
            Add-RequiredSuccess $requirements 'test-net10-sharded' $tests
        }
    }
}
else {
    switch ($reuse) {
        ([CiValidationReuseScope]::None) {
            Add-RequiredSuccess $requirements 'validation-gate' $validationGate
            Add-RequiredSuccess $requirements 'codeql' $codeql
            Add-RequiredSuccess $requirements 'sonarcloud' $sonar
        }
        ([CiValidationReuseScope]::Validation) {
            if ($requiresRuntimeValidation) {
                Add-RequiredSuccess $requirements 'promote-ci-test-analysis' $promotion
            }
            Add-RequiredSuccess $requirements 'codeql' $codeql
            Add-RequiredSuccess $requirements 'sonarcloud' $sonar
        }
        ([CiValidationReuseScope]::Complete) {
            if ($requiresRuntimeValidation) {
                Add-RequiredSuccess $requirements 'promote-ci-test-analysis' $promotion
            }
        }
    }
}

$failed = @($requirements | Where-Object Conclusion -ne ([CiJobConclusion]::Success))
$mode = if ($gateStage -eq [CiGateStage]::Validation) {
    if ($requiresRuntimeValidation) { 'runtime validation' } else { 'non-runtime validation' }
}
else {
    switch ($reuse) {
        ([CiValidationReuseScope]::None) { 'current-run validation and quality' }
        ([CiValidationReuseScope]::Validation) { 'reused validation; current-run quality' }
        ([CiValidationReuseScope]::Complete) { 'reused complete CI' }
    }
}

$summary = [System.Collections.Generic.List[string]]::new()
[void] $summary.Add("## $Stage CI Gate")
[void] $summary.Add('')
[void] $summary.Add("Mode: $mode")
[void] $summary.Add('')
[void] $summary.Add('| Required job | Result |')
[void] $summary.Add('|---|---|')
foreach ($requirement in $requirements) {
    [void] $summary.Add("| $($requirement.Name) | $($requirement.Conclusion.ToString()) |")
}
[void] $summary.Add('')
[void] $summary.Add($(if ($failed.Count -eq 0) { '**Gate PASSED**' } else { '**Gate FAILED**' }))

if ($SummaryFile) {
    $summary -join "`n" | Set-Content -LiteralPath $SummaryFile -Encoding utf8
}
else {
    $summary | Write-Host
}

if ($failed.Count -gt 0) {
    foreach ($failure in $failed) {
        Write-Host "::error title=$Stage CI Gate::Required job '$($failure.Name)' reported '$($failure.Conclusion)' (expected Success)."
    }
    exit 1
}
exit 0
