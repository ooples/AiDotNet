<# Real xUnit discovery -> bound plan -> narrowed execution -> independent receipt checks.
   Does not enable production CI filters or authenticate a GitHub certificate. #>
[CmdletBinding()]
param([Parameter(Mandatory)][string] $Binaries, [Parameter(Mandatory)][string] $EvidenceDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $root) { throw 'Plan proof output must be new.' }
[void](New-Item -ItemType Directory -Path $root)
$cli = Join-Path $PSScriptRoot 'Attribution.Cli/bin/Release/net10.0/Attribution.Cli.dll'
$testDll = Join-Path $Binaries 'PrototypeTests.dll'
$filter = 'Scenario=Positive&FullyQualifiedName!~PrototypeTests.WorkerTests'
$saved = @{}
enum PlannedRunMode { Discover; ExecutePlan }
enum ReceiptMutation { MissingCase; SkippedRow; DuplicateCase; WrongRun; MissingTrxRow; HostStillRunning }
$names = @('ATTRIBUTION_MODE','ATTRIBUTION_PLAN','ATTRIBUTION_INVENTORY','ATTRIBUTION_OUTPUT','ATTRIBUTION_RUN',
    'ATTRIBUTION_OWNER','ATTRIBUTION_TOKEN','ATTRIBUTION_SOURCE_TREE','ATTRIBUTION_PROFILE_HASH','ATTRIBUTION_WORKLOAD')
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name); [Environment]::SetEnvironmentVariable($name, $null) }
function Check([bool] $Condition, [string] $Message) { if (-not $Condition) { throw $Message } }
function Run([string] $Name, [PlannedRunMode] $Mode, [string] $Plan = '', [bool] $Reject = $false,
    [string] $ExpectedError = 'Plan does not match the current source/build/profile.') {
    $env:ATTRIBUTION_MODE = $Mode.ToString()
    $env:ATTRIBUTION_PLAN = $Plan
    $env:ATTRIBUTION_OUTPUT = Join-Path $root "$Name/reports"
    $env:ATTRIBUTION_RUN = [guid]::NewGuid().ToString('N')
    dotnet vstest $testDll "/TestCaseFilter:$filter" "/ResultsDirectory:$(Join-Path $root $Name)" `
        '/Logger:trx;LogFileName=results.trx' '/Blame:CollectHangDump;TestTimeout=30s' 2>&1 |
        Tee-Object -FilePath (Join-Path $root "$Name.log") | Out-Host
    if ($Reject) {
        Check ($LASTEXITCODE -ne 0) "$Name did not fail the rejected invocation."
        Check ((Get-Content (Join-Path $root "$Name.log") -Raw).Contains($ExpectedError)) "$Name failed for an unexpected reason."
        # A discovery/host failure can yield zero tests and even exit 0. It may
        # never contain a passing planned report, regardless of exit code.
        $reports = @(Get-ChildItem -LiteralPath $env:ATTRIBUTION_OUTPUT -Filter '*.json' -ErrorAction SilentlyContinue)
        foreach ($file in $reports) {
            $report = Get-Content $file.FullName -Raw | ConvertFrom-Json
            Check (@($report.Cases).Count -eq 0) "$Name executed tests despite rejecting the plan."
        }
        $trxPath = Join-Path $root "$Name/results.trx"
        if (Test-Path $trxPath) {
            [xml]$rejectedTrx = Get-Content $trxPath -Raw
            Check (@($rejectedTrx.SelectNodes('//*[local-name()="UnitTestResult"]')).Count -eq 0) "$Name executed tests despite rejecting the plan."
        }
    } else { Check ($LASTEXITCODE -eq 0) "$Name process failed." }
    return [pscustomobject]@{ name=$Name; run=$env:ATTRIBUTION_RUN; directory=(Join-Path $root $Name) }
}
function Verify($Run, [string] $Plan, [string] $Inventory, [string] $Output, [bool] $Reject = $false) {
    dotnet $cli Verify $Inventory $Plan (Join-Path $Run.directory 'reports') (Join-Path $Run.directory 'results.trx') `
        $Run.run 'local/attribution-fixture' 1 1 $Output | Out-Host
    Check (($LASTEXITCODE -eq 0) -ne $Reject) 'Receipt verifier result differed from expectation.'
    if ($Reject) { Check (-not (Test-Path $Output)) 'Rejected evidence published a receipt.' }
}
try {
    $env:ATTRIBUTION_SOURCE_TREE = (& git -C $PSScriptRoot rev-parse HEAD | Out-String).Trim()
    Check ($LASTEXITCODE -eq 0) 'Cannot resolve checkout identity.'
    $profile = "net10.0|$filter|$([Runtime.InteropServices.RuntimeInformation]::FrameworkDescription)|$([Runtime.InteropServices.RuntimeInformation]::OSDescription)|$([Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture)|blame-30s"
    $env:ATTRIBUTION_PROFILE_HASH = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($profile))).ToLowerInvariant()
    $env:ATTRIBUTION_WORKLOAD = 'fixture-positive-single-bundle/net10.0'
    $inventory = Join-Path $root 'inventory.json'
    $env:ATTRIBUTION_INVENTORY = $inventory
    $discovery = Run discovery Discover
    Check (Test-Path $inventory) 'Discovery did not publish an inventory.'
    $manifest = Get-Content $inventory -Raw | ConvertFrom-Json
    Check ($manifest.Cases.Count -eq 11) 'Discovery lost or added cases.'
    if (Test-Path (Join-Path $discovery.directory 'results.trx')) {
        [xml]$trx = Get-Content (Join-Path $discovery.directory 'results.trx') -Raw
        Check (@($trx.SelectNodes('//*[local-name()="UnitTestResult"]')).Count -eq 0) 'Discovery executed tests.'
    }
    $partial = Join-Path $root 'partial-plan.json'
    dotnet $cli Prepare $inventory $partial SelectedMethods 'PrototypeTests:PrototypeTests.MethodTests.First' 'PrototypeTests:PrototypeTests.DeferredTests.AllRows' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Cannot prepare partial plan.'
    $selected = Run selected ExecutePlan $partial
    $partialReceipt = Join-Path $root 'partial-verification.json'
    Verify $selected $partial $inventory $partialReceipt
    $verification = Get-Content $partialReceipt -Raw | ConvertFrom-Json
    Check ($verification.Cases.Count -eq 3 -and -not $verification.CanReplaceFullBaseline) 'Partial execution was promoted or lost a theory discovery case.'
    [xml]$trx = Get-Content (Join-Path $selected.directory 'results.trx') -Raw
    $rows = @($trx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    $expected = @('PrototypeTests.MethodTests.First(value: 1)','PrototypeTests.MethodTests.First(value: 2)',
        'PrototypeTests.DeferredTests.AllRows(value: 1)','PrototypeTests.DeferredTests.AllRows(value: 2)')
    Check ((($rows.testName | Sort-Object) -join '|') -ceq (($expected | Sort-Object) -join '|')) 'Runner did not execute exactly the selected methods and every theory row.'

    foreach ($mutation in [Enum]::GetValues[ReceiptMutation]()) {
        $name = "receipt-$mutation"
        $directory = Join-Path $root $name
        [void](New-Item -ItemType Directory -Path (Join-Path $directory 'reports'))
        $originalReport = @(Get-ChildItem (Join-Path $selected.directory 'reports') -Filter '*.json')
        Check ($originalReport.Count -eq 1) 'Expected exactly one selected host report.'
        $changed = Get-Content $originalReport[0].FullName -Raw | ConvertFrom-Json
        [xml]$changedTrx = Get-Content (Join-Path $selected.directory 'results.trx') -Raw
        switch ($mutation) {
            ([ReceiptMutation]::MissingCase) { $changed.Cases = @($changed.Cases | Select-Object -Skip 1) }
            ([ReceiptMutation]::SkippedRow) { $changed.Cases[0].Results[0].Outcome = 'Skipped' }
            ([ReceiptMutation]::DuplicateCase) { $changed.Cases += $changed.Cases[0] }
            ([ReceiptMutation]::WrongRun) { $changed.Run = [guid]::NewGuid().ToString('N') }
            ([ReceiptMutation]::HostStillRunning) { $changed.ProcessId = $PID }
            ([ReceiptMutation]::MissingTrxRow) {
                $row = $changedTrx.SelectSingleNode('//*[local-name()="UnitTestResult"]')
                [void]$row.ParentNode.RemoveChild($row)
            }
        }
        $changed | ConvertTo-Json -Depth 30 | Set-Content (Join-Path $directory "reports/$($originalReport[0].Name)")
        $changedTrx.Save((Join-Path $directory 'results.trx'))
        $changedRun = [pscustomobject]@{ directory=$directory; run=$selected.run }
        Verify $changedRun $partial $inventory (Join-Path $root "forbidden-$name.json") $true
    }

    $full = Join-Path $root 'full-plan.json'
    dotnet $cli Prepare $inventory $full FullWorkload | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Cannot prepare full plan.'
    Verify $selected $full $inventory (Join-Path $root 'forbidden-partial-as-full.json') $true
    $all = Run full ExecutePlan $full
    $fullReceipt = Join-Path $root 'full-verification.json'
    Verify $all $full $inventory $fullReceipt
    $complete = Get-Content $fullReceipt -Raw | ConvertFrom-Json
    Check ($complete.Cases.Count -eq 11 -and $complete.CanReplaceFullBaseline) 'Full workload did not verify.'
    [xml]$fullTrx = Get-Content (Join-Path $all.directory 'results.trx') -Raw
    Check (@($fullTrx.SelectNodes('//*[local-name()="UnitTestResult"]')).Count -eq 12) 'Full run lost a runtime theory row.'

    $invalid = Get-Content $partial -Raw | ConvertFrom-Json
    $invalid.Context.BuildFingerprint = '0' * 64
    $stale = Join-Path $root 'stale-plan.json'
    $invalid | ConvertTo-Json -Depth 12 | Set-Content $stale
    $staleRun = Run stale ExecutePlan $stale $true
    $invalid = Get-Content $partial -Raw | ConvertFrom-Json
    $invalid.RequiredCases = @($invalid.RequiredCases | Select-Object -SkipLast 1)
    $missing = Join-Path $root 'missing-row-plan.json'
    $invalid | ConvertTo-Json -Depth 12 | Set-Content $missing
    $missingRun = Run missing-row ExecutePlan $missing $true 'Plan was modified or has incomplete method rows.'
    # Change independently supplied invocation identity, not just the plan.
    $originalProfile = $env:ATTRIBUTION_PROFILE_HASH
    $env:ATTRIBUTION_PROFILE_HASH = 'changed-profile'
    $profileRun = Run wrong-profile ExecutePlan $partial $true
    $env:ATTRIBUTION_PROFILE_HASH = $originalProfile
    $originalWorkload = $env:ATTRIBUTION_WORKLOAD
    $env:ATTRIBUTION_WORKLOAD = 'different-workload/net10.0'
    $workloadRun = Run wrong-workload ExecutePlan $partial $true 'Plan is for a different workload.'
    $env:ATTRIBUTION_WORKLOAD = $originalWorkload
    $originalSource = $env:ATTRIBUTION_SOURCE_TREE
    $env:ATTRIBUTION_SOURCE_TREE = '0' * 40
    $sourceRun = Run wrong-source ExecutePlan $partial $true
    $env:ATTRIBUTION_SOURCE_TREE = $originalSource

    # Revoke otherwise valid evidence after process exit. The verifier must not
    # ignore extra/pending files and manufacture a passing receipt.
    $selectedReport = @(Get-ChildItem (Join-Path $selected.directory 'reports') -Filter '*.json')
    Check ($selectedReport.Count -eq 1) 'Expected one host report.'
    $revocation = $selectedReport[0].FullName + '.invalid'
    [IO.File]::WriteAllText($revocation, 'late evidence')
    Verify $selected $partial $inventory (Join-Path $root 'forbidden-revoked.json') $true
    dotnet $cli Prepare $inventory (Join-Path $root 'unknown-plan.json') SelectedMethods Unknown | Out-Host
    Check ($LASTEXITCODE -ne 0 -and -not (Test-Path (Join-Path $root 'unknown-plan.json'))) 'Unknown method was accepted.'
    [ordered]@{ schema=1; discoveredCases=11; fullRows=12; selectedCases=3; selectedRows=4;
        partialCannotReplaceBaseline=$true; rejected=@('partial-as-full','stale-binary','missing-row','unknown-method',
            'wrong-profile','wrong-workload','wrong-source','revoked-report');
        receiptMutations=@([Enum]::GetNames[ReceiptMutation]());
        productionSelectionEnabled=$false; authenticatedWorkflowOrigin=$false } |
        ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'proof.json')
    Write-Host "Planned execution proof passed: 4 selected rows versus 12 full-workload rows. Evidence: $root"
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
