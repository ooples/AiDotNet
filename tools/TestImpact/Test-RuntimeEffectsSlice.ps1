<# Local experimental comparison only. Never creates reuse or production certificates. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BeforeBundle,
    [Parameter(Mandatory)][string]$AfterBundle,
    [Parameter(Mandatory)][string]$BeforeInventory,
    [Parameter(Mandatory)][string]$AfterInventory,
    [Parameter(Mandatory)][string]$BeforeFullTrx,
    [Parameter(Mandatory)][string]$AfterFullTrx,
    [Parameter(Mandatory)][string]$EvidenceDirectory,
    [Parameter(Mandatory)][string]$Profile
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $root) { throw 'Use a new evidence directory.' }
$null = New-Item -ItemType Directory -Path $root
$instrumenter = Join-Path $PSScriptRoot 'fixtures/TestAttribution/Instrumenter/bin/Release/net10.0/AttributionInstrumenter.dll'
$cli = Join-Path $PSScriptRoot 'Attribution.Cli/bin/Release/net10.0/Attribution.Cli.dll'
$candidateFile = Join-Path $root 'candidates.json'
dotnet $instrumenter $BeforeBundle $candidateFile RuntimeEffectsExperiment $AfterBundle $AfterInventory 'AiDotNet.DistributedTraining.ShardingConfiguration`1' CreateForZeROOffload
if ($LASTEXITCODE -ne 0) { throw 'Ownership candidate analysis failed.' }
$candidate = Get-Content -LiteralPath $candidateFile -Raw | ConvertFrom-Json
$before = Get-Content -LiteralPath $BeforeInventory -Raw | ConvertFrom-Json
$after = Get-Content -LiteralPath $AfterInventory -Raw | ConvertFrom-Json
if ($candidate.ProductionSelectionEnabled -or $candidate.CanAuthorizeReuse -or -not $candidate.RequiresFullControl -or
    $candidate.Candidates.Count -ne 2 -or $before.Cases.Count -ne 5 -or $after.Cases.Count -ne 5) {
    throw 'The bounded experiment requires two candidates from five real tests.'
}
function Read-Rows([string]$Path) {
    [xml]$document = Get-Content -LiteralPath $Path -Raw
    @($document.SelectNodes('//*[local-name()="UnitTestResult"]') | ForEach-Object {
        [ordered]@{ CaseId = [string]$_.testName; Outcome = [string]$_.outcome }
    })
}
$names = @('ATTRIBUTION_SOURCE_TREE','ATTRIBUTION_PROFILE_HASH','ATTRIBUTION_WORKLOAD','ATTRIBUTION_MODE','ATTRIBUTION_PLAN','ATTRIBUTION_INVENTORY','ATTRIBUTION_OUTPUT','ATTRIBUTION_RUN')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
try {
    foreach ($side in @('before', 'after')) {
        $manifest = if ($side -ceq 'before') { $before } else { $after }
        $inventory = if ($side -ceq 'before') { $BeforeInventory } else { $AfterInventory }
        $bundle = if ($side -ceq 'before') { $BeforeBundle } else { $AfterBundle }
        $plan = Join-Path $root "$side-plan.json"
        dotnet $cli Prepare $inventory $plan SelectedMethods @($candidate.Candidates)
        if ($LASTEXITCODE -ne 0) { throw 'Could not prepare explicit experimental plan.' }
        $env:ATTRIBUTION_SOURCE_TREE = $manifest.Context.SourceTree
        $env:ATTRIBUTION_PROFILE_HASH = $Profile
        $env:ATTRIBUTION_WORKLOAD = $manifest.Workload
        $env:ATTRIBUTION_MODE = 'ExecutePlan'
        $env:ATTRIBUTION_PLAN = $plan
        $env:ATTRIBUTION_INVENTORY = $inventory
        $env:ATTRIBUTION_OUTPUT = Join-Path $root "$side/reports"
        $env:ATTRIBUTION_RUN = [guid]::NewGuid().ToString('N')
        dotnet vstest (Join-Path $bundle 'AiDotNetTests.dll') '/TestCaseFilter:FullyQualifiedName~AiDotNetTests.UnitTests.DistributedTraining.CpuOffloadShardingConfigTests.ShardingConfiguration_' "/ResultsDirectory:$root/$side" '/Logger:trx;LogFileName=results.trx'
        $expectedExit = if ($side -ceq 'before') { 0 } else { 1 }
        if ($LASTEXITCODE -ne $expectedExit) { throw 'Unexpected selected-run outcome.' }
        $rows = @(Read-Rows (Join-Path $root "$side/results.trx"))
        if ($rows.Count -ne 2) { throw 'Selected execution did not run exactly two cases.' }
        if ($side -ceq 'before') {
            dotnet $cli Verify $inventory $plan $env:ATTRIBUTION_OUTPUT "$root/$side/results.trx" $env:ATTRIBUTION_RUN local/runtime-effects-slice 1 1 "$root/before-verified.json"
            if ($LASTEXITCODE -ne 0) { throw 'Selected baseline receipt failed verification.' }
        } else {
            $failures = @($rows | Where-Object Outcome -cne 'Passed')
            if ($failures.Count -ne 1 -or -not $failures[0].CaseId.EndsWith('.ShardingConfiguration_CreateForZeROOffload_SetsOnlyOptimizerFlag')) {
                throw 'The mutation did not fail exactly its intended existing test.'
            }
            [xml]$failedTrx = Get-Content -LiteralPath "$root/$side/results.trx" -Raw
            $messages = @($failedTrx.SelectNodes('//*[local-name()="Message"]') | ForEach-Object { $_.InnerText })
            if ('CreateForZeROOffload must enable optimizer-state offload.' -cnotin $messages) {
                throw 'The failure is not the intended flag regression.'
            }
            $reports = @(Get-ChildItem -LiteralPath $env:ATTRIBUTION_OUTPUT -Filter '*.json' | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
            $expectedPlan = Get-Content -LiteralPath $plan -Raw | ConvertFrom-Json
            if ($reports.Count -ne 1 -or $reports[0].Kind -cne 'TestHost' -or $reports[0].Run -cne $env:ATTRIBUTION_RUN -or
                $reports[0].Faults.Count -ne 1 -or $reports[0].Faults[0] -cne 'UnsuccessfulCase' -or
                $reports[0].Plan.PlanHash -cne $expectedPlan.PlanHash) {
                throw 'Mutation evidence has an unrelated collection or plan fault.'
            }
            dotnet $cli Verify $inventory $plan $env:ATTRIBUTION_OUTPUT "$root/$side/results.trx" $env:ATTRIBUTION_RUN local/runtime-effects-slice 1 1 "$root/after-verified.json"
            if ($LASTEXITCODE -eq 0 -or (Test-Path "$root/after-verified.json")) { throw 'Failing mutation was incorrectly certified.' }
        }
    }
    $inputFile = Join-Path $root 'control-input.json'
    [ordered]@{
        Inventory = @($after.Cases | ForEach-Object { ([string]$_.MethodId).Split(':', 2)[1] })
        Selected = @($candidate.Candidates | ForEach-Object { ([string]$_).Split(':', 2)[1] })
        BeforeFull = @(Read-Rows $BeforeFullTrx)
        AfterFull = @(Read-Rows $AfterFullTrx)
        AfterSelected = @(Read-Rows "$root/after/results.trx")
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $inputFile -Encoding utf8
    dotnet $instrumenter $inputFile "$root/control-result.json" RuntimeEffectsControl
    if ($LASTEXITCODE -ne 0) { throw 'Independent full-run comparison rejected the candidate.' }
    Write-Output 'REAL_RUNTIME_EFFECTS: baseline=5/5 passed; selected-before=2/2 passed; selected-after=1 passed,1 expected failure; full-after=4 passed,1 expected failure; omitted=3 unchanged; productionSelectionEnabled=false'
}
finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
