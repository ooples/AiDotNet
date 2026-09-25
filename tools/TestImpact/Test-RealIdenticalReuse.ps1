<# Checks the identical-plan path against the existing real CPU configuration tests.
   Inputs must be the previously discovered five-case inventory, checksum-verified
   source bundle and its unchanged compiled binaries. No production workflow is enabled. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Inventory,
    [Parameter(Mandatory)][string] $SourceSnapshot,
    [Parameter(Mandatory)][string] $Binaries,
    [Parameter(Mandatory)][string] $EvidenceDirectory
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $root) { throw 'Evidence directory must be new.' }
[void](New-Item -ItemType Directory -Path $root)
$cli = Join-Path $PSScriptRoot 'Attribution.Cli/bin/Release/net10.0/Attribution.Cli.dll'
$repo = (& git -C $PSScriptRoot rev-parse --show-toplevel | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate repository.' }
$manifest = Get-Content -LiteralPath $Inventory -Raw | ConvertFrom-Json
$prefix = 'AiDotNetTests:AiDotNetTests.UnitTests.DistributedTraining.CpuOffloadShardingConfigTests.ShardingConfiguration_'
if ($manifest.Cases.Count -ne 5 -or @($manifest.Cases | Where-Object { -not $_.MethodId.StartsWith($prefix) }).Count -ne 0) {
    throw 'This proof requires the five real CPU sharding-configuration cases.'
}
$names = @('ATTRIBUTION_MODE','ATTRIBUTION_PLAN','ATTRIBUTION_INVENTORY','ATTRIBUTION_OUTPUT','ATTRIBUTION_RUN',
    'ATTRIBUTION_OWNER','ATTRIBUTION_TOKEN','ATTRIBUTION_SOURCE_TREE','ATTRIBUTION_PROFILE_HASH','ATTRIBUTION_WORKLOAD')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name); [Environment]::SetEnvironmentVariable($name, $null) }
function CheckExit([string] $Message) { if ($LASTEXITCODE -ne 0) { throw $Message } }
try {
    $env:ATTRIBUTION_SOURCE_TREE = $manifest.Context.SourceTree
    $env:ATTRIBUTION_PROFILE_HASH = 'local-net10-cpu-sharding-selection'
    $env:ATTRIBUTION_WORKLOAD = $manifest.Workload
    $env:ATTRIBUTION_MODE = 'ExecutePlan'
    $env:ATTRIBUTION_PLAN = Join-Path $root 'full-plan.json'
    $env:ATTRIBUTION_OUTPUT = Join-Path $root 'baseline/reports'
    $env:ATTRIBUTION_RUN = [guid]::NewGuid().ToString('N')
    dotnet $cli Prepare $Inventory $env:ATTRIBUTION_PLAN FullWorkload
    CheckExit 'Full plan preparation failed.'
    $filter = 'FullyQualifiedName~AiDotNetTests.UnitTests.DistributedTraining.CpuOffloadShardingConfigTests.ShardingConfiguration_'
    dotnet vstest (Join-Path $Binaries 'AiDotNetTests.dll') "/TestCaseFilter:$filter" "/ResultsDirectory:$root/baseline" '/Logger:trx;LogFileName=results.trx'
    CheckExit 'Real baseline execution failed.'
    [xml]$trx = Get-Content "$root/baseline/results.trx" -Raw
    $rows = @($trx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    if ($rows.Count -ne 5 -or @($rows | Where-Object { $_.outcome -ne 'Passed' }).Count -ne 0) { throw 'Real baseline lost a passing case.' }
    $revision = @{ Snapshot=[IO.Path]::GetFullPath($SourceSnapshot); Inventory=[IO.Path]::GetFullPath($Inventory); Bundle=[IO.Path]::GetFullPath($Binaries) }
    @{ Repository=$repo; Before=$revision; After=$revision; Baseline=@{
        Plan=$env:ATTRIBUTION_PLAN; Reports=$env:ATTRIBUTION_OUTPUT; Trx="$root/baseline/results.trx"; CollectionRun=$env:ATTRIBUTION_RUN;
        Origin=@{ Repository='local/aidotnet-real-proof'; RunId=1; Attempt=1 } } } |
        ConvertTo-Json -Depth 8 | Set-Content "$root/request.json"
    dotnet $cli PrepareReuse "$root/request.json" "$root/unexpected-plan.json" "$root/partition.json"
    CheckExit 'Real identical-plan partition failed.'
    $partition = Get-Content "$root/partition.json" -Raw | ConvertFrom-Json
    if ($partition.ExecutionRequired -or $partition.ReusedCases.Count -ne 5 -or (Test-Path "$root/unexpected-plan.json")) {
        throw 'Identical successful workload unexpectedly scheduled execution.'
    }
    dotnet $cli CompleteReuse "$root/request.json" "$root/completed.json"
    CheckExit 'Real identical-plan completion failed.'
    $complete = Get-Content "$root/completed.json" -Raw | ConvertFrom-Json
    if ($complete.CanReplaceFullBaseline -or $null -ne $complete.ExecutedCases -or $complete.ReusedCases.Count -ne 5) {
        throw 'Reused outcomes were represented as newly executed tests.'
    }
    @{ baselinePassed=5; identicalReused=5; currentExecuted=0; source=$manifest.Context.SourceTree;
        newFullBaseline=$false; authenticatedWorkflowOrigin=$false; productionSelectionEnabled=$false } |
        ConvertTo-Json | Set-Content "$root/proof.json"
    Write-Host "Real identical-plan reuse verified. Evidence: $root"
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
