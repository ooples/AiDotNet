[CmdletBinding()]
param(
    [Parameter(Mandatory)][long] $RunId,
    [Parameter(Mandatory)][string] $HeadSha,
    [Parameter(Mandatory)][string] $EvidenceDirectory,
    [string] $Repository = 'ooples/AiDotNet',
    [int] $Attempt = 1
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $root) { throw 'Live import evidence directory must be new.' }
[void](New-Item -ItemType Directory -Path $root)
$cli = Join-Path $PSScriptRoot 'Attribution.Cli/bin/Release/net10.0/Attribution.Cli.dll'
$policy = [ordered]@{ Repository=$Repository; RunId=$RunId; Attempt=$Attempt; HeadSha=$HeadSha;
    WorkflowPath='.github/workflows/test-attribution-canary.yml'; JobName='collector-proof';
    ArtifactName="attribution-proof-$RunId-$Attempt"; CompletionScope='WholeRun'; RequiredScope='FullWorkload' }
$request = [ordered]@{ Policy=$policy; Inventory='planned/inventory.json'; Plan='planned/full-plan.json';
    Reports='planned/full/reports'; Trx='planned/full/results.trx' }
function Invoke-Import([string] $Name, [string] $ExpectedRejection = '') {
    $request | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$root/$Name-request.json"
    dotnet $cli ImportWorkflow "$root/$Name-request.json" "$root/$Name-download" "$root/$Name-result.json" `
        2>&1 | Tee-Object -FilePath "$root/$Name.log" | Out-Host
    if ($ExpectedRejection) {
        if ($LASTEXITCODE -eq 0 -or (Test-Path "$root/$Name-result.json") -or
            -not (Get-Content "$root/$Name.log" -Raw).Contains($ExpectedRejection)) { throw "$Name was not rejected for its intended reason." }
    } elseif ($LASTEXITCODE -ne 0) { throw "$Name import failed." }
}
Invoke-Import baseline
$baseline = Get-Content "$root/baseline-result.json" -Raw | ConvertFrom-Json
if (-not $baseline.AuthenticatedWorkflowOrigin -or -not $baseline.CanReplaceFullBaseline -or
    $baseline.Origin.RunId -ne $RunId -or $baseline.Origin.Attempt -ne $Attempt -or $baseline.Cases.Count -ne 10) {
    throw 'Live baseline import lost its actual execution or workflow binding.'
}
$policy.HeadSha = '0' * 40
Invoke-Import wrong-head 'Workflow origin, attempt or completion does not match policy.'
$policy.HeadSha = $HeadSha
$policy.WorkflowPath = '.github/workflows/not-the-producer.yml'
Invoke-Import wrong-workflow 'Workflow origin, attempt or completion does not match policy.'
$policy.WorkflowPath = '.github/workflows/test-attribution-canary.yml'
$request.Plan = 'planned/partial-plan.json'
$request.Reports = 'planned/selected/reports'
$request.Trx = 'planned/selected/results.trx'
Invoke-Import partial-as-full 'Imported execution has the wrong required scope.'
[ordered]@{ runId=$RunId; attempt=$Attempt; headSha=$HeadSha; testedSource=$baseline.Context.SourceTree;
    authenticatedFullBaseline=$true; cases=$baseline.Cases.Count; wrongHeadRejected=$true;
    wrongWorkflowRejected=$true; partialBaselineRejected=$true; productionSelectionEnabled=$false } |
    ConvertTo-Json | Set-Content -LiteralPath "$root/proof.json"
Write-Host "Live workflow import verified. Evidence: $root"
