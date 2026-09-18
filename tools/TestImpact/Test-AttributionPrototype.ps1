<# Private-copy prototype; never changes production filters, maps or certificates. #>
[CmdletBinding()]
param([switch] $NoBuild, [string] $EvidenceDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum PrototypeRejection { Results; ReportInventory; WrongRun; Faulted; MissingWorker; UnknownMethod; Binary; MissingOwner; GuardTestFailed }
function Reject([PrototypeRejection] $Reason, [string] $Message) {
    $failure = [IO.InvalidDataException]::new($Message)
    $failure.Data['Reason'] = $Reason
    throw $failure
}
function Check([bool] $Condition, [string] $Message) { if (-not $Condition) { throw $Message } }

function Assert-Evidence {
    param([string] $Directory, [string] $Run, [string[]] $ExpectedCases, [string] $Assembly, [string] $Map)
    try { [xml] $trx = Get-Content -LiteralPath (Join-Path $Directory 'results.trx') -Raw }
    catch { Reject Results 'Missing or malformed TRX.' }
    $cases = @($trx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    $actual = @($cases | ForEach-Object { [string] $_.testName } | Sort-Object)
    if (($actual -join '|') -cne (($ExpectedCases | Sort-Object) -join '|') -or
        @($cases | Where-Object { $_.outcome -cne 'Passed' }).Count -gt 0 -or
        [string] $trx.TestRun.ResultSummary.outcome -cne 'Completed') {
        Reject Results 'Unexpected, missing, skipped, or failing test results.'
    }
    $mapData = Get-Content -LiteralPath $Map -Raw | ConvertFrom-Json
    if ($mapData.Schema -ne 1 -or (Get-FileHash -LiteralPath $Assembly -Algorithm SHA256).Hash -cne $mapData.OutputHash) {
        Reject Binary 'Instrumented binary does not match its map.'
    }
    $known = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($method in $mapData.Methods) {
        if (-not $known.Add([string] $method.Key) -or -not ([string] $method.Key).StartsWith("$($mapData.InputHash):", [StringComparison]::Ordinal)) {
            Reject UnknownMethod 'Invalid method inventory.'
        }
    }
    $expectedOwners = @($ExpectedCases | ForEach-Object { 'PrototypeTests:' + ($_ -replace '\(.*$', '') } | Sort-Object -Unique)
    $reportDirectory = Join-Path $Directory 'attribution'
    if (-not (Test-Path -LiteralPath $reportDirectory -PathType Container)) { Reject ReportInventory 'Missing attribution directory.' }
    if (@(Get-ChildItem -LiteralPath $reportDirectory -Filter '*.pending' -File).Count -gt 0) { Reject ReportInventory 'Incomplete report publication.' }
    if (@(Get-ChildItem -LiteralPath $reportDirectory -Filter '*.invalid' -File).Count -gt 0) { Reject ReportInventory 'Report revoked after publication.' }
    $reports = @(Get-ChildItem -LiteralPath $reportDirectory -Filter '*.json' -File | ForEach-Object {
        try { $value = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json }
        catch { Reject ReportInventory 'Malformed attribution report.' }
        if ($value.Schema -ne 4 -or $_.BaseName -cne $value.Token -or $value.Kind -cnotin @('TestHost', 'Worker')) {
            Reject ReportInventory 'Unsupported report or filename.'
        }
        $value
    })
    $hosts = @($reports | Where-Object { $_.Kind -ceq 'TestHost' })
    if ($hosts.Count -ne 1) { Reject ReportInventory 'Expected exactly one test-host report.' }
    foreach ($report in $reports) {
        if ($report.Run -cne $Run) { Reject WrongRun 'Foreign-run report.' }
        if (@($report.Faults).Count -gt 0) { Reject Faulted "Faulted attribution: $($report.Faults -join ',')." }
        foreach ($owner in $report.CompletedOwners) {
            if ($owner -cnotin $expectedOwners) { Reject MissingOwner 'Unknown completed owner.' }
        }
        foreach ($hit in $report.Hits) {
            if ($hit.Owner -cne '<execution-group>' -and $hit.Owner -cnotin $report.CompletedOwners) {
                Reject MissingOwner 'Hit owner never completed.'
            }
            foreach ($method in $hit.Methods) {
                if (-not $known.Contains([string] $method)) { Reject UnknownMethod 'Hit not present in the instrumented method map.' }
            }
        }
    }
    $hostReport = $hosts[0]
    $caseIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $recordedNames = [Collections.Generic.List[string]]::new()
    foreach ($entry in $hostReport.Cases) {
        if ([string]::IsNullOrWhiteSpace([string] $entry.Case.Id) -or -not $caseIds.Add([string] $entry.Case.Id) -or
            $entry.Case.Owner -cnotin $expectedOwners -or $entry.Finished -isnot [bool] -or -not $entry.Finished -or
            $entry.Case.Kind -cnotin @('Enumerated', 'DeferredOrCustom') -or @($entry.Results).Count -eq 0 -or
            ($entry.Case.Kind -ceq 'Enumerated' -and @($entry.Results).Count -ne 1)) {
            Reject ReportInventory 'Invalid or incomplete independently discovered case ledger.'
        }
        foreach ($result in $entry.Results) {
            if ($result.Outcome -cne 'Passed' -or [string]::IsNullOrWhiteSpace([string] $result.DisplayName) -or
                ($entry.Case.Kind -ceq 'Enumerated' -and $result.DisplayName -cne $entry.Case.DisplayName)) {
                Reject ReportInventory 'Case ledger contains an invalid result.'
            }
            $recordedNames.Add([string] $result.DisplayName)
        }
    }
    if ((($recordedNames | Sort-Object) -join '|') -cne ($actual -join '|')) {
        Reject ReportInventory 'Case ledger and independent TRX results differ.'
    }
    if (($expectedOwners -join '|') -cne (($hostReport.CompletedOwners | Sort-Object -Unique) -join '|')) {
        Reject MissingOwner 'Execution results and completed ownership boundaries differ.'
    }
    $workers = @($reports | Where-Object { $_.Kind -ceq 'Worker' })
    $tickets = @($hostReport.Workers)
    if ($workers.Count -ne $tickets.Count -or @($tickets | ForEach-Object { $_.Token } | Sort-Object -Unique).Count -ne $tickets.Count) {
        Reject MissingWorker 'Worker report count does not match registrations.'
    }
    foreach ($ticket in $tickets) {
        $matching = @($workers | Where-Object { $_.Token -ceq $ticket.Token })
        if ($ticket.Started -isnot [bool] -or -not $ticket.Started -or $ticket.Completed -isnot [bool] -or -not $ticket.Completed -or
            $ticket.Run -cne $Run -or $ticket.Owner -cnotin $expectedOwners -or $matching.Count -ne 1 -or
            $matching[0].WorkerOwner -cne $ticket.Owner -or @($matching[0].CompletedOwners).Count -ne 1 -or
            $matching[0].CompletedOwners[0] -cne $ticket.Owner -or @($matching[0].Workers).Count -ne 0 -or @($matching[0].Cases).Count -ne 0) {
            Reject MissingWorker 'Unmatched/incomplete worker ownership or unsupported nested worker.'
        }
    }
    return $reports
}

$fixture = Join-Path $PSScriptRoot 'fixtures/TestAttribution'
$project = Join-Path $fixture 'PrototypeTests/PrototypeTests.csproj'
$root = if ($EvidenceDirectory) { [IO.Path]::GetFullPath($EvidenceDirectory) }
    else { Join-Path ([IO.Path]::GetTempPath()) "attribution-prototype-$([guid]::NewGuid().ToString('N'))" }
Check (-not (Test-Path -LiteralPath $root)) 'Evidence directory must be new; existing evidence cannot be overwritten.'
[void] (New-Item -ItemType Directory -Path $root)
$prior = @{}
foreach ($name in @('ATTRIBUTION_OUTPUT', 'ATTRIBUTION_RUN', 'ATTRIBUTION_OWNER', 'ATTRIBUTION_TOKEN', 'ATTRIBUTION_WORKER_DLL', 'ATTRIBUTION_HIT_MODE',
    'ATTRIBUTION_MODE', 'ATTRIBUTION_PLAN', 'ATTRIBUTION_INVENTORY', 'ATTRIBUTION_SOURCE_TREE', 'ATTRIBUTION_PROFILE_HASH', 'ATTRIBUTION_WORKLOAD',
    'ATTRIBUTION_DISCOVERY_PROBE')) {
    $prior[$name] = [Environment]::GetEnvironmentVariable($name)
    [Environment]::SetEnvironmentVariable($name, $null)
}
$positive = @('PrototypeTests.MethodTests.First(value: 1)', 'PrototypeTests.MethodTests.First(value: 2)',
    'PrototypeTests.MethodTests.Second', 'PrototypeTests.MethodTests.SuppressedContext',
    'PrototypeTests.MethodTests.PotentialCaller',
    'PrototypeTests.InheritedLeftTests.Inherited', 'PrototypeTests.InheritedRightTests.Inherited',
    'PrototypeTests.CleanupTests.CleanupJoinsBackground',
    'PrototypeTests.DeferredTests.AllRows(value: 1)', 'PrototypeTests.DeferredTests.AllRows(value: 2)',
    'PrototypeTests.ParallelLeftTests.Left', 'PrototypeTests.ParallelRightTests.Right', 'PrototypeTests.WorkerTests.Complete')
$runs = [Collections.Generic.List[object]]::new()
$rejections = [Collections.Generic.List[string]]::new()

function Invoke-PrototypeRun([string] $Name, [string] $Scenario, [string] $Binaries, [bool] $Collect, [int] $ExpectedExit = 0) {
    $directory = Join-Path $root $Name
    $run = [guid]::NewGuid().ToString('N')
    $env:ATTRIBUTION_RUN = $run
    $env:ATTRIBUTION_OUTPUT = if ($Collect) { Join-Path $directory 'attribution' } else { $null }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    dotnet vstest (Join-Path $Binaries 'PrototypeTests.dll') "/TestCaseFilter:Scenario=$Scenario" `
        "/ResultsDirectory:$directory" '/Logger:trx;LogFileName=results.trx' '/Blame:CollectHangDump;TestTimeout=30s' | Out-Host
    Check ($LASTEXITCODE -eq $ExpectedExit) "$Name returned $LASTEXITCODE instead of $ExpectedExit."
    $timer.Stop()
    $item = [pscustomobject]@{ name = $Name; seconds = [math]::Round($timer.Elapsed.TotalSeconds, 3); directory = $directory; run = $run }
    $runs.Add($item)
    return $item
}

function Expect-Rejected([string] $Name, [PrototypeRejection] $Reason, [scriptblock] $Action) {
    $observed = $false
    try { & $Action | Out-Null }
    catch {
        if ($_.Exception.Data['Reason'] -ne $Reason) { throw }
        $observed = $true
    }
    if (-not $observed) { Reject GuardTestFailed "$Name unexpectedly accepted invalid evidence." }
    $rejections.Add($Name)
}

try {
    $sdkVersion = (& dotnet --version | Out-String).Trim()
    Check ($LASTEXITCODE -eq 0 -and $sdkVersion.Length -gt 0) 'Could not resolve the effective SDK.'
    if (-not $NoBuild) {
        dotnet build $project -c Release --nologo -m:2 -p:UseSharedCompilation=false
        Check ($LASTEXITCODE -eq 0) 'Prototype build failed.'
    }
    & (Join-Path $PSScriptRoot 'Test-AttributionBuildBinding.ps1') -EvidenceDirectory (Join-Path $root 'build-binding')
    Check (Test-Path (Join-Path $root 'build-binding/proof.json')) 'Stable attribution build binding was not proved.'
    $original = Join-Path $fixture 'PrototypeTests/bin/Release/net10.0'
    $workerOriginal = Join-Path $fixture 'Worker/bin/Release/net10.0'
    $env:ATTRIBUTION_WORKER_DLL = Join-Path $workerOriginal 'AttributionWorker.dll'
    # Attempt to re-enable eager discovery through adapter configuration. The
    # sealed opt-in framework must still prevent an unrelated data provider
    # from running in the selected test's process, with and without collection.
    $discoveryBundle = Join-Path $root 'discovery-probe-bundle'
    Copy-Item -LiteralPath $original -Destination $discoveryBundle -Recurse
    [IO.File]::WriteAllText((Join-Path $discoveryBundle 'xunit.runner.json'), '{"preEnumerateTheories":true}')
    foreach ($collect in @($false, $true)) {
        $probeName = if ($collect) { 'discovery-collected' } else { 'discovery-uncollected' }
        $env:ATTRIBUTION_DISCOVERY_PROBE = Join-Path $root "$probeName.jsonl"
        $null = Invoke-PrototypeRun $probeName 'DiscoveryProbe' $discoveryBundle $collect
        $observations = @(Get-Content -LiteralPath $env:ATTRIBUTION_DISCOVERY_PROBE | ForEach-Object { $_ | ConvertFrom-Json })
        Check ($observations.Count -eq 1 -and $observations[0].Stage -ceq 'Execution' -and $observations[0].Discoveries -eq 0) `
            'Unselected theory provider executed despite enforced deferred discovery.'
    }
    $env:ATTRIBUTION_DISCOVERY_PROBE = $null
    $env:ATTRIBUTION_OUTPUT = $null
    dotnet vstest (Join-Path $original 'PrototypeTests.dll') '/TestCaseFilter:Scenario=Protocol' `
        "/ResultsDirectory:$(Join-Path $root 'protocol')" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Execution evidence protocol tests failed.'
    [xml] $protocolTrx = Get-Content -LiteralPath (Join-Path $root 'protocol/results.trx') -Raw
    $protocolCases = @($protocolTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($protocolCases.Count -eq 15 -and @($protocolCases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) `
        'Protocol checks did not execute the complete expected case set.'
    dotnet vstest (Join-Path $original 'PrototypeTests.dll') '/TestCaseFilter:Scenario=DependencySelection' `
        "/ResultsDirectory:$(Join-Path $root 'dependency-selection')" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Dependency selection tests failed.'
    [xml] $selectionTrx = Get-Content -LiteralPath (Join-Path $root 'dependency-selection/results.trx') -Raw
    $selectionCases = @($selectionTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($selectionCases.Count -eq 17 -and @($selectionCases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) `
        'Dependency selection checks did not execute the complete expected case set.'
    dotnet vstest (Join-Path $original 'PrototypeTests.dll') '/TestCaseFilter:Scenario=RunnerProtocol' `
        "/ResultsDirectory:$(Join-Path $root 'runner-protocol')" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Runner binding protocol tests failed.'
    [xml] $runnerTrx = Get-Content -LiteralPath (Join-Path $root 'runner-protocol/results.trx') -Raw
    $runnerCases = @($runnerTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($runnerCases.Count -eq 124 -and @($runnerCases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) `
        'Runner binding checks did not execute the expected complete set.'
    dotnet vstest (Join-Path $original 'PrototypeTests.dll') '/TestCaseFilter:Scenario=ReuseProtocol' `
        "/ResultsDirectory:$(Join-Path $root 'reuse-protocol')" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Changed-base partition protocol tests failed.'
    [xml] $reuseTrx = Get-Content -LiteralPath (Join-Path $root 'reuse-protocol/results.trx') -Raw
    $reuseCases = @($reuseTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($reuseCases.Count -eq 14 -and @($reuseCases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) `
        'Changed-base partition checks did not execute the expected complete set.'
    dotnet vstest (Join-Path $original 'PrototypeTests.dll') '/TestCaseFilter:Scenario=SourceImpact' `
        "/ResultsDirectory:$(Join-Path $root 'source-impact')" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Source-impact protocol tests failed.'
    [xml] $sourceTrx = Get-Content -LiteralPath (Join-Path $root 'source-impact/results.trx') -Raw
    $sourceCases = @($sourceTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($sourceCases.Count -eq 67 -and @($sourceCases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) `
        'Source-impact checks did not execute the expected complete set.'
    dotnet vstest (Join-Path $original 'PrototypeTests.dll') '/TestCaseFilter:Scenario=RuntimeEffects' `
        "/ResultsDirectory:$(Join-Path $root 'runtime-effects')" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Runtime-effects experimental controls failed.'
    [xml] $effectsTrx = Get-Content -LiteralPath (Join-Path $root 'runtime-effects/results.trx') -Raw
    $effectsCases = @($effectsTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($effectsCases.Count -eq 566 -and @($effectsCases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) `
        'Runtime-effects controls did not execute the complete expected set.'
    dotnet vstest (Join-Path $original 'PrototypeTests.dll') '/TestCaseFilter:Scenario=WorkflowProtocol' `
        "/ResultsDirectory:$(Join-Path $root 'workflow-protocol')" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Workflow import protocol tests failed.'
    [xml] $workflowTrx = Get-Content -LiteralPath (Join-Path $root 'workflow-protocol/results.trx') -Raw
    $workflowCases = @($workflowTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($workflowCases.Count -eq 21 -and @($workflowCases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) `
        'Workflow import checks did not execute the expected complete set.'
    $sourceAssembly = Join-Path $original 'AttributionSubject.dll'
    $originalHash = (Get-FileHash -LiteralPath $sourceAssembly).Hash
    $env:ATTRIBUTION_WORKER_DLL = Join-Path $workerOriginal 'AttributionWorker.dll'
    $plain = Invoke-PrototypeRun plain Positive $original $false
    $hostCopy = Join-Path $root 'host'
    $workerCopy = Join-Path $root 'worker'
    Copy-Item -LiteralPath $original -Destination $hostCopy -Recurse
    Copy-Item -LiteralPath $workerOriginal -Destination $workerCopy -Recurse
    $instrumented = Join-Path $root 'AttributionSubject.dll'
    $instrumenter = Join-Path $fixture 'Instrumenter/bin/Release/net10.0/AttributionInstrumenter.dll'
    dotnet $instrumenter $sourceAssembly $instrumented
    Check ($LASTEXITCODE -eq 0) 'Instrumentation failed.'
    $map = $instrumented + '.map.json'
    $taskAssembly = Join-Path $root 'PrototypeTests.dll'
    dotnet $instrumenter (Join-Path $original 'PrototypeTests.dll') $taskAssembly TaskBoundaries
    Check ($LASTEXITCODE -eq 0) 'Task boundary instrumentation failed.'
    Copy-Item -LiteralPath $taskAssembly -Destination (Join-Path $hostCopy 'PrototypeTests.dll')
    Copy-Item -LiteralPath ([IO.Path]::ChangeExtension($taskAssembly, '.pdb')) -Destination (Join-Path $hostCopy 'PrototypeTests.pdb')
    foreach ($destination in @($hostCopy, $workerCopy)) {
        Copy-Item -LiteralPath $instrumented -Destination (Join-Path $destination 'AttributionSubject.dll')
        Copy-Item -LiteralPath ([IO.Path]::ChangeExtension($instrumented, '.pdb')) -Destination (Join-Path $destination 'AttributionSubject.pdb')
        Check ((Get-FileHash (Join-Path $destination 'AttributionSubject.dll')).Hash -ceq (Get-FileHash $instrumented).Hash) 'Binary copy mismatch.'
    }
    $env:ATTRIBUTION_WORKER_DLL = Join-Path $workerCopy 'AttributionWorker.dll'
    $collected = Invoke-PrototypeRun collected Positive $hostCopy $true
    $reports = @(Assert-Evidence $collected.directory $collected.run $positive $instrumented $map)
    [xml] $plainTrx = Get-Content (Join-Path $plain.directory 'results.trx') -Raw
    $plainNames = @($plainTrx.SelectNodes('//*[local-name()="UnitTestResult"]') | ForEach-Object { [string] $_.testName } | Sort-Object)
    Check (($plainNames -join '|') -ceq (($positive | Sort-Object) -join '|')) 'Plain control inventory differs.'
    $hostReport = @($reports | Where-Object { $_.Kind -ceq 'TestHost' })[0]
    $deferred = @($hostReport.Cases | Where-Object { $_.Case.Owner -ceq 'PrototypeTests:PrototypeTests.DeferredTests.AllRows' })
    Check ($deferred.Count -eq 1 -and $deferred[0].Case.Kind -ceq 'DeferredOrCustom' -and $deferred[0].Results.Count -eq 2) `
        'Deferred theory was not tracked as one discovered case with both actual rows.'
    Check ($hostReport.PeakScopes -ge 2) 'No actual test-scope overlap observed.'
    $mapData = Get-Content $map -Raw | ConvertFrom-Json
    Check ($mapData.OutputStringHeapBytes -le 0x00ffffff -and
        ($mapData.OutputStringHeapBytes - $mapData.InputStringHeapBytes) -le 256) `
        'Instrumentation added per-method user strings instead of one shared module identity.'
    $potential = @($mapData.DependencyGraph.Methods | Where-Object { $_.Name -match '::UntakenBranch\(' })
    $leftNode = @($mapData.DependencyGraph.Methods | Where-Object { $_.Name -match 'Operations::Left\(' })
    Check ($potential.Count -eq 1 -and $leftNode.Count -eq 1 -and $leftNode[0].Key -cin $potential[0].LocalCalls) `
        'Static graph lost the dependency in an unexecuted branch.'
    $initializer = @($mapData.DependencyGraph.Methods | Where-Object { $_.Name -match 'StaticDependencyProbe::\.cctor\(' })
    $readStatic = @($mapData.DependencyGraph.Methods | Where-Object { $_.Name -match 'StaticDependencyProbe::Read\(' })
    Check ($initializer.Count -eq 1 -and $readStatic.Count -eq 1 -and $initializer[0].Key -cin $readStatic[0].LocalCalls `
        -and $readStatic[0].StaticFields.Count -eq 1) 'Implicit initializer or shared static field dependency disappeared.'
    $externalField = @($mapData.DependencyGraph.Methods | Where-Object { $_.Name -match 'StaticDependencyProbe::ExternalField\(' })
    Check ($externalField.Count -eq 1 -and @($externalField[0].OpenDependencies | Where-Object { $_.Target -match 'System.DateTime::MinValue$' }).Count -eq 1) `
        'Unresolved external field was incorrectly classified as complete.'
    $potentialOwner = @($reports.Hits | Where-Object { $_.Owner -ceq 'PrototypeTests:PrototypeTests.MethodTests.PotentialCaller' })
    Check ($potentialOwner.Count -eq 1 -and $leftNode[0].Key -cnotin $potentialOwner[0].Methods) `
        'Untaken-branch control unexpectedly executed Left.'
    function Methods-For([string] $Owner) {
        $keys = @($reports.Hits | Where-Object { $_.Owner -ceq $Owner } | ForEach-Object { $_.Methods })
        return @($mapData.Methods | Where-Object { $_.Key -cin $keys } | ForEach-Object { [string] $_.Name })
    }
    foreach ($case in @(@('MethodTests.First', 'Left', 'Right'), @('MethodTests.Second', 'Right', 'Left'),
        @('InheritedLeftTests.Inherited', 'Left', 'Right'), @('InheritedRightTests.Inherited', 'Right', 'Left'),
        @('ParallelLeftTests.Left', 'Left', 'Right'), @('ParallelRightTests.Right', 'Right', 'Left'))) {
        $methods = @(Methods-For "PrototypeTests:PrototypeTests.$($case[0])")
        Check (@($methods | Where-Object { $_ -match "Operations::$($case[1])\(" }).Count -eq 1) "Missing expected dependency for $($case[0])."
        Check (@($methods | Where-Object { $_ -match "Operations::$($case[2])\(" }).Count -eq 0) "Cross-test attribution leak for $($case[0])."
    }
    $shared = @(Methods-For '<execution-group>')
    foreach ($name in @('Setup', 'Cleanup', 'Unowned')) {
        Check (@($shared | Where-Object { $_ -match "::$name\(" }).Count -eq 1) "Lost conservative group dependency: $name."
    }
    $workerMethods = @(Methods-For 'PrototypeTests:PrototypeTests.WorkerTests.Complete')
    Check (@($workerMethods | Where-Object { $_ -match '::WorkerOnly\(' }).Count -eq 1) 'Child-only method was not attributed.'
    $completedDerived = Invoke-PrototypeRun 'derived-complete' 'DerivedTaskComplete' $hostCopy $true 0
    $null = Assert-Evidence $completedDerived.directory $completedDerived.run @('PrototypeTests.DerivedTaskTests.CompletedTasksKeepTheirOriginalReturnTypes') $instrumented $map
    foreach ($case in @(@('late', 'Late', 'PrototypeTests.LateTests.LateBackground', [PrototypeRejection]::Faulted, 0),
        @('missing-worker', 'MissingWorker', 'PrototypeTests.WorkerTests.Missing', [PrototypeRejection]::MissingWorker, 0),
        @('unclosed-worker', 'UnclosedWorker', 'PrototypeTests.WorkerTests.Unclosed', [PrototypeRejection]::Faulted, 0),
        @('unjoined-worker', 'UnjoinedWorker', 'PrototypeTests.WorkerTests.Unjoined', [PrototypeRejection]::Faulted, 0),
        @('detached-task', 'DetachedTask', 'PrototypeTests.DetachedTaskTests.NeverHitsCoveredCode', [PrototypeRejection]::Faulted, 0),
        @('untracked-timer', 'UntrackedTimer', 'PrototypeTests.UntrackedBoundaryTests.TimerNeverFires', [PrototypeRejection]::Faulted, 0),
        @('untracked-process', 'UntrackedProcess', 'PrototypeTests.UntrackedBoundaryTests.ProcessNeverProducesCoverage', [PrototypeRejection]::Faulted, 0),
        @('killed-worker', 'KilledWorker', 'PrototypeTests.WorkerTests.KilledWorker', [PrototypeRejection]::Faulted, 0),
        @('worker-replay', 'WorkerReplay', 'PrototypeTests.WorkerTests.ReplayStart', [PrototypeRejection]::Faulted, 0),
        @('worker-never-started', 'WorkerNeverStarted', 'PrototypeTests.WorkerTests.CompleteWithoutStarting', [PrototypeRejection]::Faulted, 0),
        @('value-task', 'ValueTask', 'PrototypeTests.ValueTaskTests.Unfinished', [PrototypeRejection]::Faulted, 0),
        @('derived-task', 'DerivedTask', 'PrototypeTests.DerivedTaskTests.UnfinishedDerivedReturn', [PrototypeRejection]::Faulted, 0),
        @('derived-generic-task', 'DerivedGenericTask', 'PrototypeTests.DerivedTaskTests.UnfinishedDerivedGenericReturn', [PrototypeRejection]::Faulted, 0),
        @('generic-value-task', 'GenericValueTask', 'PrototypeTests.ValueTaskTests.UnfinishedGeneric', [PrototypeRejection]::Faulted, 0),
        @('value-task-constructor', 'ValueTaskConstructor', 'PrototypeTests.ValueTaskTests.DirectConstructor', [PrototypeRejection]::Faulted, 0),
        @('custom-skip', 'CustomSkip', 'PrototypeTests.CustomCaseTests.CustomRunnerStillSkips', [PrototypeRejection]::Results, 0),
        @('after-publication', 'AfterPublication', 'PrototypeTests.UntrackedBoundaryTests.ExitCallbackHitsAfterReport', [PrototypeRejection]::ReportInventory, 0),
        @('test-failure', 'Failure', 'PrototypeTests.FailingTests.FailsAfterCoverage', [PrototypeRejection]::Results, 1))) {
        $negative = Invoke-PrototypeRun $case[0] $case[1] $hostCopy $true $case[4]
        if ($case[1] -ceq 'CustomSkip') {
            [xml] $skipTrx = Get-Content -LiteralPath (Join-Path $negative.directory 'results.trx') -Raw
            $skipResults = @($skipTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
            Check ($skipResults.Count -eq 1 -and $skipResults[0].outcome -ceq 'NotExecuted') `
                'Custom test runner was bypassed or its skip became a failure.'
        }
        Expect-Rejected $case[0] $case[3] { Assert-Evidence $negative.directory $negative.run @($case[2]) $instrumented $map }
    }
    # Remove the safeguard in a private compiled copy and require the original
    # detached-task regression assertion to fail, rather than merely checking a green control.
    $mutantHost = Join-Path $root 'mutant-host'
    Copy-Item -LiteralPath $hostCopy -Destination $mutantHost -Recurse
    $mutantRuntime = Join-Path $root 'AttributionRuntime.dll'
    dotnet $instrumenter (Join-Path $hostCopy 'AttributionRuntime.dll') $mutantRuntime RemoveTaskObserverForMutationTest
    Check ($LASTEXITCODE -eq 0) 'Guard mutation setup failed.'
    Copy-Item -LiteralPath $mutantRuntime -Destination (Join-Path $mutantHost 'AttributionRuntime.dll')
    Copy-Item -LiteralPath ([IO.Path]::ChangeExtension($mutantRuntime, '.pdb')) -Destination (Join-Path $mutantHost 'AttributionRuntime.pdb')
    $guardRun = Invoke-PrototypeRun guard-removal DetachedTask $mutantHost $true
    Expect-Rejected guard-removal-regression GuardTestFailed {
        Expect-Rejected detached-task-required Faulted {
            Assert-Evidence $guardRun.directory $guardRun.run @('PrototypeTests.DetachedTaskTests.NeverHitsCoveredCode') $instrumented $map
        }
    }
    Expect-Rejected wrong-run WrongRun { Assert-Evidence $collected.directory ([guid]::NewGuid().ToString('N')) $positive $instrumented $map }
    Expect-Rejected stale-binary Binary { Assert-Evidence $collected.directory $collected.run $positive $sourceAssembly $map }
    Expect-Rejected missing-test Results { Assert-Evidence $collected.directory $collected.run ($positive + 'PrototypeTests.Missing.Test') $instrumented $map }
    # Mutate private copies of real evidence, not hand-written success fixtures.
    foreach ($mutation in @('missing-artifact', 'pending-artifact', 'malformed-artifact', 'unknown-method', 'skipped-case', 'wrong-worker-owner',
        'missing-case-ledger', 'duplicate-case-ledger', 'unfinished-case-ledger', 'skipped-case-ledger', 'zero-test-results')) {
        $mutated = Join-Path $root "mutant-$mutation"
        Copy-Item -LiteralPath $collected.directory -Destination $mutated -Recurse
        $hostFile = Join-Path $mutated "attribution/$($hostReport.Token).json"
        $workerReport = @($reports | Where-Object { $_.Kind -ceq 'Worker' })[0]
        $workerFile = Join-Path $mutated "attribution/$($workerReport.Token).json"
        $expectedReason = [PrototypeRejection]::ReportInventory
        switch ($mutation) {
            'missing-artifact' {
                Move-Item -LiteralPath $workerFile -Destination ($workerFile + '.removed')
                $expectedReason = [PrototypeRejection]::MissingWorker
            }
            'pending-artifact' { Move-Item -LiteralPath $workerFile -Destination ($workerFile + '.pending') }
            'malformed-artifact' { [IO.File]::WriteAllText($workerFile, '{truncated') }
            { $_ -in @('missing-case-ledger', 'duplicate-case-ledger', 'unfinished-case-ledger', 'skipped-case-ledger') } {
                $changed = Get-Content -LiteralPath $hostFile -Raw | ConvertFrom-Json
                switch ($mutation) {
                    'missing-case-ledger' { $changed.Cases = @($changed.Cases | Select-Object -Skip 1) }
                    'duplicate-case-ledger' { $changed.Cases = @($changed.Cases) + @($changed.Cases[0]) }
                    'unfinished-case-ledger' { $changed.Cases[0].Finished = $false }
                    'skipped-case-ledger' { $changed.Cases[0].Results[0].Outcome = 'Skipped' }
                }
                $changed | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $hostFile
            }
            'unknown-method' {
                $changed = Get-Content -LiteralPath $hostFile -Raw | ConvertFrom-Json
                $changed.Hits[0].Methods = @('unknown:method')
                $changed | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $hostFile
                $expectedReason = [PrototypeRejection]::UnknownMethod
            }
            'skipped-case' {
                $trxPath = Join-Path $mutated 'results.trx'
                [xml] $changedTrx = Get-Content -LiteralPath $trxPath -Raw
                $changedTrx.SelectNodes('//*[local-name()="UnitTestResult"]')[0].outcome = 'NotExecuted'
                $changedTrx.Save($trxPath)
                $expectedReason = [PrototypeRejection]::Results
            }
            'zero-test-results' {
                $trxPath = Join-Path $mutated 'results.trx'
                [xml] $changedTrx = Get-Content -LiteralPath $trxPath -Raw
                foreach ($result in @($changedTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))) {
                    [void] $result.ParentNode.RemoveChild($result)
                }
                $changedTrx.Save($trxPath)
                $expectedReason = [PrototypeRejection]::Results
            }
            'wrong-worker-owner' {
                $changed = Get-Content -LiteralPath $workerFile -Raw | ConvertFrom-Json
                $changed.WorkerOwner = 'PrototypeTests:PrototypeTests.MethodTests.Second'
                $changed | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $workerFile
                $expectedReason = [PrototypeRejection]::MissingWorker
            }
        }
        Expect-Rejected $mutation $expectedReason { Assert-Evidence $mutated $collected.run $positive $instrumented $map }
    }
    Check ((Get-FileHash -LiteralPath $sourceAssembly).Hash -ceq $originalHash) 'Original build output was modified.'
    $benchmarks = @()
    foreach ($mode in @('Plain', 'Serialized', 'Cached')) {
        $env:ATTRIBUTION_OUTPUT = if ($mode -ceq 'Plain') { $null } else { Join-Path $root "benchmark-$mode" }
        $env:ATTRIBUTION_OWNER = 'benchmark'
        $env:ATTRIBUTION_TOKEN = [guid]::NewGuid().ToString('N')
        $env:ATTRIBUTION_RUN = [guid]::NewGuid().ToString('N')
        $env:ATTRIBUTION_HIT_MODE = if ($mode -ceq 'Plain') { $null } else { $mode }
        $executable = Join-Path $(if ($mode -ceq 'Plain') { $workerOriginal } else { $workerCopy }) 'AttributionWorker.dll'
        $measurement = & dotnet $executable Benchmark | ConvertFrom-Json
        Check ($LASTEXITCODE -eq 0) "Benchmark $mode failed."
        Check ($measurement.Count -eq 131072 -and $measurement.Checksum -eq 8589869056) "Benchmark $mode checksum mismatch."
        $samples = @($measurement.NanosecondsPerCall | Sort-Object)
        $benchmarks += [pscustomobject]@{ mode = $mode; callsPerSample = $measurement.Count; checksum = $measurement.Checksum;
            nanosecondsPerCall = $measurement.NanosecondsPerCall; medianNanoseconds = $samples[3] }
    }
    & (Join-Path $PSScriptRoot 'Test-PlannedAttribution.ps1') -Binaries $hostCopy -EvidenceDirectory (Join-Path $root 'planned')
    Check (Test-Path (Join-Path $root 'planned/proof.json')) 'Planned-execution proof was not completed.'
    & (Join-Path $PSScriptRoot 'Test-SourceSelection.ps1') -EvidenceDirectory (Join-Path $root 'source-selection')
    Check (Test-Path (Join-Path $root 'source-selection/proof.json')) 'Actual source-selection proof was not completed.'
    & (Join-Path $PSScriptRoot 'Test-SourceSelection.ps1') -CrossAssembly -EvidenceDirectory (Join-Path $root 'cross-assembly')
    Check (Test-Path (Join-Path $root 'cross-assembly/proof.json')) 'Cross-assembly selection proof was not completed.'
    [ordered]@{ schemaVersion = 1; sdkVersion = $sdkVersion; productionSelectionEnabled = $false; runs = $runs.ToArray(); rejectedCases = $rejections.ToArray();
        peakScopes = $hostReport.PeakScopes; protocolCases = $protocolCases.Count; selectionCases = $selectionCases.Count; runnerCases = $runnerCases.Count; reuseCases = $reuseCases.Count; sourceCases = $sourceCases.Count; runtimeEffectsCases = $effectsCases.Count; workflowCases = $workflowCases.Count; benchmarks = $benchmarks; limitations = @('Prototype method-level attribution, not branch coverage.',
            'Unknown context applies to the entire execution group.', 'No production selector, trust certificate, or live workflow proof.') } |
        ConvertTo-Json -Depth 6 | Set-Content (Join-Path $root 'proof.json') -Encoding utf8
    Write-Host "Prototype checks passed. Production selection remains disabled. Evidence: $root"
}
finally {
    foreach ($name in $prior.Keys) { [Environment]::SetEnvironmentVariable($name, $prior[$name]) }
}
