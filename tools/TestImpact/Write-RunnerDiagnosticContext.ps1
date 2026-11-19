[CmdletBinding()]
param([Parameter(Mandatory)] [string] $VerifierResult,
    [Parameter(Mandatory)] [string] $TestedSha, [Parameter(Mandatory)] [string] $RunId,
    [Parameter(Mandatory)] [string] $RunAttempt, [Parameter(Mandatory)] [string] $ProofDirectory,
    [Parameter(Mandatory)] [string] $OutputDirectory, [string] $OriginalJobsPath, [string] $JobsOutputPath,
    [string] $PrimaryAttemptImportPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
$proof = $null; $resolved = @(); $jobs = @()
try { $proof = Get-Content (Join-Path $ProofDirectory 'proof.json') -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop }
catch { Write-Warning "Diagnostic proof is unavailable: $($_.Exception.Message)" }
if ($OriginalJobsPath) {
    $jobs = @(Get-Content -LiteralPath $OriginalJobsPath -Raw | ConvertFrom-Json)
    try { $resolved = @(Get-Content (Join-Path $ProofDirectory 'resolved-jobs.json') -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop) }
    catch { Write-Warning "Resolved jobs are unavailable: $($_.Exception.Message)" }
}
$context = Get-FailoverDiagnosticContext -VerifierResult $VerifierResult -Proof $proof -TestedSha $TestedSha `
    -RunId $RunId -RunAttempt $RunAttempt -OriginalJobs $jobs -ResolvedJobs $resolved -RequireResolvedJobs:([bool]$OriginalJobsPath)
if ($PrimaryAttemptImportPath) {
    $intake = Get-Content -LiteralPath $PrimaryAttemptImportPath -Raw | ConvertFrom-Json
    $context | Add-Member primaryAttemptDiagnostics $intake
    if ($intake.imported -gt 0 -or $intake.rejected -gt 0) {
        $context.verified = $false; $context.diagnosticOnly = $true
        $context.reasons += 'Raw primary attempt TRX was substituted or required results remain missing/rejected.'
        $context.regressionArtifactPrefix = 'test-outcome-diagnostics'; $context.aggregateArtifactPrefix = 'ci-test-diagnostics'
        $context.jobs = $jobs
    }
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$context | Select-Object * -ExcludeProperty jobs | ConvertTo-Json -Depth 30 |
    Set-Content (Join-Path $OutputDirectory 'evidence-provenance.json') -Encoding utf8
$status = if ($context.verified) { 'Verified logical shard evidence.' } else {
    "DIAGNOSTIC ONLY: evidence is incomplete or untrusted. Raw primary outcomes are retained; no alternate result or regression verdict is certified.`n`n" + ($context.reasons -join "`n`n")
}
@("## Evidence provenance", '', $status, '', "Verifier errors: $($context.proofErrors.Count)", '') |
    Set-Content (Join-Path $OutputDirectory 'evidence-status.md') -Encoding utf8
if ($JobsOutputPath) { ConvertTo-Json -InputObject @($context.jobs) -Depth 100 | Set-Content -LiteralPath $JobsOutputPath -Encoding utf8 }
if ($env:GITHUB_STEP_SUMMARY) { Get-Content (Join-Path $OutputDirectory 'evidence-status.md') | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY }
if ($env:GITHUB_OUTPUT) {
    "verified=$($context.verified.ToString().ToLowerInvariant())" | Add-Content -LiteralPath $env:GITHUB_OUTPUT
    "regression_artifact_prefix=$($context.regressionArtifactPrefix)" | Add-Content -LiteralPath $env:GITHUB_OUTPUT
    "aggregate_artifact_prefix=$($context.aggregateArtifactPrefix)" | Add-Content -LiteralPath $env:GITHUB_OUTPUT
}
