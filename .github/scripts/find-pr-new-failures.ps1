<#
.SYNOPSIS
Builds a VSTest filter for failures present in a PR shard but absent from the exact master baseline.

.DESCRIPTION
This is deliberately a retry selector, not a verdict engine. It reads the initial shard TRX and
either the compact baseline ledger or the legacy baseline TRX, writes the candidate inventory, and
publishes rerun_count/filter outputs. The aggregate analyzer consumes both the original and retry
TRX and owns the final policy decision.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $CurrentResultsPath,
    [string] $BaselineResultsPath,
    [string] $BaselineLedgerPath,
    [Parameter(Mandatory = $true)]
    [string] $OutputFile,
    [ValidateRange(1, [int]::MaxValue)]
    [int] $MaxRerunMethods = 40
)

$ErrorActionPreference = 'Stop'

function Get-TrxResults {
    param([string] $Root)

    $results = New-Object System.Collections.Generic.List[object]
    foreach ($trx in @(Get-ChildItem -LiteralPath $Root -Recurse -Filter '*.trx' -File -ErrorAction SilentlyContinue)) {
        [xml] $document = Get-Content -LiteralPath $trx.FullName -Raw
        $definitions = @{}
        foreach ($unitTest in @($document.SelectNodes('//*[local-name()="UnitTest"]'))) {
            $method = $unitTest.SelectSingleNode('./*[local-name()="TestMethod"]')
            if ($method -and $unitTest.id) {
                $className = [string] $method.className
                $methodName = [string] $method.name
                $definitions[[string] $unitTest.id] = if ($className) {
                    "$className.$methodName"
                } else { $methodName }
            }
        }

        foreach ($result in @($document.SelectNodes('//*[local-name()="UnitTestResult"]'))) {
            $display = [string] $result.testName
            $fullyQualified = $definitions[[string] $result.testId]
            if ([string]::IsNullOrWhiteSpace($fullyQualified)) { $fullyQualified = $display }
            $identity = if ($display -and $display -ne $fullyQualified) {
                "$fullyQualified::$display"
            } else { $fullyQualified }
            if (-not [string]::IsNullOrWhiteSpace($identity)) {
                $results.Add([PSCustomObject]@{
                    identity = $identity
                    fullyQualifiedName = $fullyQualified
                    displayName = $display
                    outcome = [string] $result.outcome
                })
            }
        }
    }
    return $results.ToArray()
}

function Set-ActionOutput {
    param([string] $Name, [string] $Value)
    Write-Host "$Name=$Value"
    if ($env:GITHUB_OUTPUT) { Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value "$Name=$Value" }
}

function ConvertTo-VSTestFilterValue {
    param([string] $Value)

    # VSTest reserves these characters inside filter values. Escape them so a parameterized
    # display name or generated method cannot turn one retry candidate into a different filter.
    # Commas are intentionally left alone: they are not operators in the filter grammar.
    return [regex]::Replace($Value, '([\\()!&|=~])', '\$1')
}

if (-not (Test-Path -LiteralPath $CurrentResultsPath)) {
    throw "Current result directory not found: $CurrentResultsPath"
}

$current = @(Get-TrxResults $CurrentResultsPath)
$baseline = if ($BaselineLedgerPath) {
    if (-not (Test-Path -LiteralPath $BaselineLedgerPath)) {
        throw "Baseline ledger not found: $BaselineLedgerPath"
    }
    @((Get-Content -LiteralPath $BaselineLedgerPath -Raw | ConvertFrom-Json).tests)
} elseif ($BaselineResultsPath) {
    if (-not (Test-Path -LiteralPath $BaselineResultsPath)) {
        throw "Baseline result directory not found: $BaselineResultsPath"
    }
    @(Get-TrxResults $BaselineResultsPath)
} else {
    throw 'BaselineLedgerPath or BaselineResultsPath is required.'
}

$baselineFailures = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::Ordinal)
foreach ($failure in @($baseline | Where-Object outcome -eq 'Failed')) {
    [void] $baselineFailures.Add([string] $failure.identity)
}

$candidates = @($current |
    Where-Object { $_.outcome -eq 'Failed' -and -not $baselineFailures.Contains([string] $_.identity) } |
    Sort-Object identity -Unique)
$methods = @($candidates.fullyQualifiedName | Where-Object { $_ } | Sort-Object -Unique)
$retrySkipped = $methods.Count -gt $MaxRerunMethods
$retrySkipReason = if ($retrySkipped) {
    "The $($methods.Count) PR-new failing methods exceed the targeted retry cap of $MaxRerunMethods."
} else { $null }
$filterParts = if ($retrySkipped) {
    @()
} else {
    @($methods | ForEach-Object { "FullyQualifiedName~$(ConvertTo-VSTestFilterValue $_)" })
}
$filter = $filterParts -join '|'

$outputParent = Split-Path -Parent $OutputFile
if ($outputParent) { New-Item -Path $outputParent -ItemType Directory -Force | Out-Null }
[PSCustomObject]@{
    candidateCount = $candidates.Count
    methodCount = $methods.Count
    maxRerunMethods = $MaxRerunMethods
    retrySkipped = $retrySkipped
    retrySkipReason = $retrySkipReason
    candidates = $candidates
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputFile -Encoding utf8

Set-ActionOutput 'rerun_count' ([string] $(if ($retrySkipped) { 0 } else { $methods.Count }))
Set-ActionOutput 'filter' $filter
