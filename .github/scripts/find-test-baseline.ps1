<#
.SYNOPSIS
Finds the exact Actions run/artifact that represents a pull request's base SHA.

.DESCRIPTION
Prefers the compact TRX ledger emitted by the aggregate job. A master push may
reuse a certified pull-request run and therefore publish only a promoted analysis
artifact under the landed SHA. In that case the caller follows promotion.json to
the source run's coverage/TRX artifacts. For older ordinary master runs, falls
back to legacy coverage/TRX artifacts from the exact master SHA.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $BaseSha,
    [string] $Repository = $env:GITHUB_REPOSITORY,
    [string] $Workflow = 'sonarcloud.yml',
    [string] $ShardSlug
)

$ErrorActionPreference = 'Stop'
if (-not $Repository) { throw 'Repository is required (owner/name).' }
if (-not $env:GITHUB_STEP_SUMMARY -and -not $env:GITHUB_OUTPUT) {
    Write-Verbose 'Running outside Actions; outputs will be written to the console only.'
}

function Invoke-GhJson {
    param([string] $Endpoint)
    $raw = & gh api $Endpoint
    if ($LASTEXITCODE -ne 0) { throw "gh api failed for '$Endpoint'." }
    return $raw | ConvertFrom-Json
}

function Set-ActionOutput {
    param([string] $Name, [string] $Value)
    Write-Host "$Name=$Value"
    if ($env:GITHUB_OUTPUT) { Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value "$Name=$Value" }
}

$artifactName = "test-outcome-ledger-$BaseSha"
$encodedName = [Uri]::EscapeDataString($artifactName)
$artifactResponse = Invoke-GhJson "repos/$Repository/actions/artifacts?name=$encodedName&per_page=100"
$ledgerArtifact = @($artifactResponse.artifacts |
    Where-Object {
        -not $_.expired -and $_.name -eq $artifactName -and
        $_.workflow_run -and $_.workflow_run.head_sha -eq $BaseSha
    } |
    Sort-Object created_at -Descending | Select-Object -First 1)

if ($ledgerArtifact.Count -gt 0) {
    Set-ActionOutput 'baseline_mode' 'ledger'
    Set-ActionOutput 'baseline_run_id' ([string] $ledgerArtifact[0].workflow_run.id)
    Set-ActionOutput 'baseline_sha' $BaseSha
    if ($ShardSlug) { Set-ActionOutput 'baseline_artifact_name' $artifactName }
    exit 0
}

# Bootstrap path: this base SHA predates compact ledgers. Locate the exact push
# run. Conclusion is intentionally unrestricted: master workflows can be
# cancelled by CodeQL after every test shard has already uploaded a valid TRX.
$encodedWorkflow = [Uri]::EscapeDataString($Workflow)
$runs = Invoke-GhJson "repos/$Repository/actions/workflows/$encodedWorkflow/runs?head_sha=$BaseSha&event=push&per_page=100"
$run = @($runs.workflow_runs |
    Where-Object { $_.head_sha -eq $BaseSha -and $_.status -eq 'completed' } |
    Sort-Object -Property @{ Expression = 'created_at'; Descending = $true },
        @{ Expression = 'run_attempt'; Descending = $true } | Select-Object -First 1)
if ($run.Count -eq 0) {
    throw "No completed '$Workflow' push run exists for exact baseline SHA $BaseSha."
}

$runId = [string] $run[0].id
$runArtifacts = Invoke-GhJson "repos/$Repository/actions/runs/$runId/artifacts?per_page=100"
$coveragePrefix = "coverage-$BaseSha-"
$hasExactCoverage = @($runArtifacts.artifacts | Where-Object {
    -not $_.expired -and ([string] $_.name).StartsWith($coveragePrefix, [StringComparison]::Ordinal)
}).Count -gt 0

if ($hasExactCoverage) {
    Set-ActionOutput 'baseline_mode' 'trx'
    Set-ActionOutput 'baseline_run_id' $runId
    Set-ActionOutput 'baseline_sha' $BaseSha
    if ($ShardSlug) { Set-ActionOutput 'baseline_artifact_name' "coverage-$BaseSha-$ShardSlug" }
    exit 0
}

# A certified-reuse master run intentionally skips the duplicate test matrix.
# Its promoted analysis records the source PR run and tested merge SHA in
# promotion.json, which the workflow uses to fetch the real TRX artifacts.
$analysisName = "ci-test-analysis-$BaseSha"
$encodedAnalysisName = [Uri]::EscapeDataString($analysisName)
$analysisResponse = Invoke-GhJson "repos/$Repository/actions/artifacts?name=$encodedAnalysisName&per_page=100"
$analysisArtifact = @($analysisResponse.artifacts |
    Where-Object {
        -not $_.expired -and $_.name -eq $analysisName -and
        $_.workflow_run -and $_.workflow_run.head_sha -eq $BaseSha
    } |
    Sort-Object created_at -Descending | Select-Object -First 1)

if ($analysisArtifact.Count -gt 0) {
    Set-ActionOutput 'baseline_mode' 'promoted'
    Set-ActionOutput 'baseline_run_id' ([string] $analysisArtifact[0].workflow_run.id)
    Set-ActionOutput 'baseline_sha' $BaseSha
    if ($ShardSlug) { Set-ActionOutput 'baseline_artifact_name' $analysisName }
    exit 0
}

throw "Completed '$Workflow' run $runId for baseline SHA $BaseSha has neither a compact ledger, exact coverage artifacts, nor promoted analysis provenance."
