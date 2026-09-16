[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $WorkloadJson,
    [Parameter(Mandatory)] [ValidateSet('success', 'failure', 'cancelled', 'skipped')] [string] $TestStepOutcome,
    [string] $ResultsDirectory = 'TestResults',
    [string] $DigestDirectory = 'impact-digest'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/CiWorkloadKinds.ps1"
$workload = $WorkloadJson | ConvertFrom-Json
if ((Get-CiWorkloadKind $workload) -eq [CiWorkloadKind]::Tests) {
    throw 'An auxiliary evidence writer cannot relabel an ordinary test shard.'
}
if ([string]::IsNullOrWhiteSpace([string] $workload.name)) { throw 'Missing workload name.' }
$slug = [string] $workload.name -replace '[\\/:*?"<>|\s-]+', '_'
New-Item -ItemType Directory -Path $ResultsDirectory -Force | Out-Null
[pscustomobject]@{
    shard = [string] $workload.name
    slug = $slug
    sha = $env:GITHUB_SHA
    runId = $env:GITHUB_RUN_ID
    runAttempt = $env:GITHUB_RUN_ATTEMPT
    testStepOutcome = $TestStepOutcome
    rerunCandidateCount = '0'
    rerunStepOutcome = 'skipped'
} | ConvertTo-Json | Set-Content (Join-Path $ResultsDirectory 'shard-metadata.json') -Encoding utf8
"slug=$slug" | Out-File -LiteralPath $env:GITHUB_OUTPUT -Append -Encoding utf8

# A reporter can return zero after writing failed findings. Never turn its partial or
# failing test evidence into a digest that would authorize skipping that workload.
if ($TestStepOutcome -ne 'success') { exit 0 }
$trxFiles = @(Get-ChildItem -LiteralPath $ResultsDirectory -Filter '*.trx' -Recurse -File)
if ($trxFiles.Count -eq 0) { Write-Host 'No TRX: workload remains always-run.'; exit 0 }
foreach ($file in $trxFiles) {
    [xml] $trx = Get-Content -LiteralPath $file.FullName -Raw
    $counters = $trx.SelectSingleNode('//*[local-name()="ResultSummary"]/*[local-name()="Counters"]')
    if ($null -eq $counters -or [int] $counters.total -le 0 -or
        [int] $counters.executed -ne [int] $counters.total -or
        [int] $counters.passed -ne [int] $counters.total) {
        Write-Host 'Incomplete or failing TRX: workload remains always-run.'
        exit 0
    }
}
$coverage = @(Get-ChildItem -LiteralPath $ResultsDirectory -Filter 'coverage.opencover.xml' -Recurse -File |
    Sort-Object Length -Descending)
if ($coverage.Count -eq 0) { Write-Host 'No coverage: workload remains always-run.'; exit 0 }
$LASTEXITCODE = 0
& "$PSScriptRoot/New-CoverageDigest.ps1" -CoverageXml $coverage[0].FullName `
    -Shard ([string] $workload.name) -OutFile (Join-Path $DigestDirectory "$slug.digest.json")
if ($LASTEXITCODE -ne 0) { throw 'Auxiliary coverage digest failed.' }
exit 0
