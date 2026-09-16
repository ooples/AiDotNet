<# Private-copy prototype; never changes production filters, maps or certificates. #>
[CmdletBinding()]
param([switch] $NoBuild, [string] $EvidenceDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum PrototypeRejection { Results; ReportInventory; WrongRun; Faulted; MissingWorker; UnknownMethod; Binary; MissingOwner }
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
    $reports = @(Get-ChildItem -LiteralPath $reportDirectory -Filter '*.json' -File | ForEach-Object {
        try { $value = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json }
        catch { Reject ReportInventory 'Malformed attribution report.' }
        if ($value.Schema -ne 1 -or $_.BaseName -cne $value.Token -or $value.Kind -cnotin @('TestHost', 'Worker')) {
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
        if ($ticket.Completed -isnot [bool] -or -not $ticket.Completed -or
            $ticket.Run -cne $Run -or $ticket.Owner -cnotin $expectedOwners -or $matching.Count -ne 1 -or
            $matching[0].WorkerOwner -cne $ticket.Owner -or @($matching[0].CompletedOwners).Count -ne 1 -or
            $matching[0].CompletedOwners[0] -cne $ticket.Owner -or @($matching[0].Workers).Count -ne 0) {
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
foreach ($name in @('ATTRIBUTION_OUTPUT', 'ATTRIBUTION_RUN', 'ATTRIBUTION_OWNER', 'ATTRIBUTION_TOKEN', 'ATTRIBUTION_WORKER_DLL', 'ATTRIBUTION_HIT_MODE')) {
    $prior[$name] = [Environment]::GetEnvironmentVariable($name)
    [Environment]::SetEnvironmentVariable($name, $null)
}
$positive = @('PrototypeTests.MethodTests.First(value: 1)', 'PrototypeTests.MethodTests.First(value: 2)',
    'PrototypeTests.MethodTests.Second', 'PrototypeTests.MethodTests.SuppressedContext',
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
    Check $observed "$Name unexpectedly accepted invalid evidence."
    $rejections.Add($Name)
}

try {
    $sdkVersion = (& dotnet --version | Out-String).Trim()
    Check ($LASTEXITCODE -eq 0 -and $sdkVersion.Length -gt 0) 'Could not resolve the effective SDK.'
    if (-not $NoBuild) {
        dotnet build $project -c Release --nologo -m:2 -p:UseSharedCompilation=false
        Check ($LASTEXITCODE -eq 0) 'Prototype build failed.'
    }
    $original = Join-Path $fixture 'PrototypeTests/bin/Release/net10.0'
    $workerOriginal = Join-Path $fixture 'Worker/bin/Release/net10.0'
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
    Check ($hostReport.PeakScopes -ge 2) 'No actual test-scope overlap observed.'
    $mapData = Get-Content $map -Raw | ConvertFrom-Json
    function Methods-For([string] $Owner) {
        $keys = @($reports.Hits | Where-Object { $_.Owner -ceq $Owner } | ForEach-Object { $_.Methods })
        return @($mapData.Methods | Where-Object { $_.Key -cin $keys } | ForEach-Object { [string] $_.Name })
    }
    foreach ($case in @(@('MethodTests.First', 'Left', 'Right'), @('MethodTests.Second', 'Right', 'Left'),
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
    foreach ($case in @(@('late', 'Late', 'PrototypeTests.LateTests.LateBackground', [PrototypeRejection]::Faulted, 0),
        @('missing-worker', 'MissingWorker', 'PrototypeTests.WorkerTests.Missing', [PrototypeRejection]::MissingWorker, 0),
        @('unclosed-worker', 'UnclosedWorker', 'PrototypeTests.WorkerTests.Unclosed', [PrototypeRejection]::Faulted, 0),
        @('unjoined-worker', 'UnjoinedWorker', 'PrototypeTests.WorkerTests.Unjoined', [PrototypeRejection]::Faulted, 0),
        @('detached-task', 'DetachedTask', 'PrototypeTests.DetachedTaskTests.NeverHitsCoveredCode', [PrototypeRejection]::Faulted, 0),
        @('untracked-timer', 'UntrackedTimer', 'PrototypeTests.UntrackedBoundaryTests.TimerNeverFires', [PrototypeRejection]::Faulted, 0),
        @('untracked-process', 'UntrackedProcess', 'PrototypeTests.UntrackedBoundaryTests.ProcessNeverProducesCoverage', [PrototypeRejection]::Faulted, 0),
        @('test-failure', 'Failure', 'PrototypeTests.FailingTests.FailsAfterCoverage', [PrototypeRejection]::Results, 1))) {
        $negative = Invoke-PrototypeRun $case[0] $case[1] $hostCopy $true $case[4]
        Expect-Rejected $case[0] $case[3] { Assert-Evidence $negative.directory $negative.run @($case[2]) $instrumented $map }
    }
    Expect-Rejected wrong-run WrongRun { Assert-Evidence $collected.directory ([guid]::NewGuid().ToString('N')) $positive $instrumented $map }
    Expect-Rejected stale-binary Binary { Assert-Evidence $collected.directory $collected.run $positive $sourceAssembly $map }
    Expect-Rejected missing-test Results { Assert-Evidence $collected.directory $collected.run ($positive + 'PrototypeTests.Missing.Test') $instrumented $map }
    # Mutate private copies of real evidence, not hand-written success fixtures.
    foreach ($mutation in @('missing-artifact', 'pending-artifact', 'malformed-artifact', 'unknown-method', 'skipped-case', 'wrong-worker-owner')) {
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
    [ordered]@{ schemaVersion = 1; sdkVersion = $sdkVersion; productionSelectionEnabled = $false; runs = $runs.ToArray(); rejectedCases = $rejections.ToArray();
        peakScopes = $hostReport.PeakScopes; benchmarks = $benchmarks; limitations = @('Prototype method-level attribution, not branch coverage.',
            'Unknown context applies to the entire execution group.', 'No production selector, trust certificate, or live workflow proof.') } |
        ConvertTo-Json -Depth 6 | Set-Content (Join-Path $root 'proof.json') -Encoding utf8
    Write-Host "Prototype checks passed. Production selection remains disabled. Evidence: $root"
}
finally {
    foreach ($name in $prior.Keys) { [Environment]::SetEnvironmentVariable($name, $prior[$name]) }
}
