<#
.SYNOPSIS
Finds the exact Actions run/artifact that represents a pull request's base SHA.

.DESCRIPTION
Prefers the compact TRX ledger emitted by the aggregate job. For the first run
after this feature is introduced, falls back to the legacy coverage/TRX artifacts
from the exact master SHA so the rollout does not require a manual baseline.
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
# run and let download-artifact fetch its coverage-<sha>-* artifacts. Conclusion
# is intentionally unrestricted: master workflows can be cancelled by CodeQL
# after every test shard has already uploaded a valid TRX.
$encodedWorkflow = [Uri]::EscapeDataString($Workflow)
$runs = Invoke-GhJson "repos/$Repository/actions/workflows/$encodedWorkflow/runs?head_sha=$BaseSha&event=push&per_page=100"
$run = @($runs.workflow_runs |
    Where-Object { $_.head_sha -eq $BaseSha -and $_.status -eq 'completed' } |
    Sort-Object run_attempt, created_at -Descending | Select-Object -First 1)
if ($run.Count -eq 0) {
    throw "No completed '$Workflow' push run exists for exact baseline SHA $BaseSha."
}

Set-ActionOutput 'baseline_mode' 'trx'
Set-ActionOutput 'baseline_run_id' ([string] $run[0].id)
Set-ActionOutput 'baseline_sha' $BaseSha
if ($ShardSlug) { Set-ActionOutput 'baseline_artifact_name' "coverage-$BaseSha-$ShardSlug" }
