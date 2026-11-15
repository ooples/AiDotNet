<#
.SYNOPSIS
    Proves Layer 2 of test-impact selection: repacking shard slices into time-balanced jobs (ShardPacking.ps1).

.DESCRIPTION
    -RulesOnly runs the checks that need no child process and no repository fixture: the flag, the scope, the
    prediction, every packing rule, set equality over a spread of generated matrices and the real manifest, the
    timing table, the reading of a packed run's jobs, the report, and the workflow's side of the contract.
    Select-AffectedTests.ps1 runs it before it trusts a packed matrix, and the pipeline runs it with the other
    impact-tooling checks.

    Without -RulesOnly it also runs the real scripts end to end:
      * Select-AffectedTests.ps1 with the flag unset, off, report and on: unset and off publish what they
        published before this existed, report publishes the same and records a plan, on packs, and a packed
        matrix over the job-output cap, a run out of scope and a missing table all fall back to one job per shard;
      * Invoke-Shard.ps1 over a packed job, with `dotnet` replaced by a stub: a failing slice does not stop the
        ones after it and fails the job, each slice gets its own filter, environment, hang timeout, test host
        settings and results directory, a job killed part-way keeps what finished and records the rest as
        cancelled, and a job without segments runs as it always did;
      * the per-slice artifacts read by the regression analysis as separate shards, the targeted retry of a
        packed slice, and New-ShardTimingTable.ps1 from the ledger of a packed run.
    It never starts a real `dotnet`.
#>
[CmdletBinding()]
param([switch] $RulesOnly)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/ShardPacking.ps1"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$script:Checks = 0

function Assert-True {
    param($Condition, [string] $Because)
    $script:Checks++
    if (-not $Condition) { throw "shard packing self-test: $Because" }
}

function Assert-Equal {
    param($Expected, $Actual, [string] $Because)
    $script:Checks++
    if ($Expected -cne $Actual) { throw "shard packing self-test: $Because (expected '$Expected', got '$Actual')" }
}

function Assert-Throws {
    param([scriptblock] $Block, [string] $Because)
    $script:Checks++
    $threw = $false
    try { & $Block | Out-Null } catch { $threw = $true }
    if (-not $threw) { throw "shard packing self-test: nothing refused $Because" }
}

function New-Slice {
    param([string] $Name, [hashtable] $With = @{})
    $slice = [ordered]@{
        name = $Name; framework = 'net10.0'; project = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
        filter = "FullyQualifiedName~Ns.$($Name -replace '[^A-Za-z0-9]', '')"
    }
    foreach ($key in $With.Keys) { $slice[$key] = $With[$key] }
    return [pscustomobject] $slice
}

function New-Table {
    <# A table in which each named shard takes exactly the given minutes (no invocation cost, factor 1). #>
    param([hashtable] $MinutesByShard)
    $shards = [ordered]@{}
    foreach ($name in @($MinutesByShard.Keys | Sort-Object)) {
        $seconds = [double] $MinutesByShard[$name] * 60
        $shards[(ConvertTo-ShardSlug $name)] = [ordered]@{ seconds = $seconds; wallFactor = 1.0; classes = [ordered]@{ 'Ns.Whole' = $seconds } }
    }
    $json = ConvertTo-Json -Depth 6 -InputObject ([ordered]@{
            schemaVersion = 1; generatedUtc = '2026-11-14T00:00:00Z'; invocationSeconds = 0.0; defaultWallFactor = 1.0
            sources = @(); shards = $shards })
    return ($json | ConvertFrom-Json -AsHashtable)
}

function ConvertTo-SliceJson($Slice) { ConvertTo-Json -InputObject $Slice -Depth 8 -Compress }

function Assert-PackInvariants {
    <# Independent of Assert-ShardPackComplete: every rule, re-derived from the packed matrix alone. #>
    param([object[]] $Slices, [object[]] $Matrix, [object] $Plan, [string[]] $HeavyNames = @(), [string[]] $AloneNames = @(),
        [double] $Target = 20.0, [string] $Label)
    $expected = @{}
    foreach ($slice in $Slices) { $expected[[string] $slice.name] = ConvertTo-SliceJson $slice }
    $source = @{}
    $predicted = @{}
    foreach ($job in @($Plan.jobs)) {
        foreach ($slice in @($job.slices)) { $source[[string] $slice.name] = [string] $slice.source; $predicted[[string] $slice.name] = [double] $slice.predictedMinutes }
    }
    $seen = [Collections.Generic.List[string]]::new()
    foreach ($entry in $Matrix) {
        $segments = if ($entry.PSObject.Properties['segments']) { @($entry.segments) } else { $null }
        if ($null -eq $segments) {
            $seen.Add([string] $entry.name)
            Assert-Equal $expected[[string] $entry.name] (ConvertTo-SliceJson $entry) "$Label`: an unpacked job is not the slice it was"
            continue
        }
        Assert-True ($segments.Count -ge 2 -and $segments.Count -le $script:ShardPackMaxSegments) "$Label`: a packed job holds $($segments.Count) slices"
        Assert-True ([string] $entry.name -cmatch $script:ShardPackJobNamePattern) "$Label`: packed job '$($entry.name)' is not named as one"
        Assert-True (-not $entry.PSObject.Properties['filter']) "$Label`: a packed job carries a filter of its own"
        $sum = 0.0
        $weights = @{}
        for ($slot = 0; $slot -lt $segments.Count; $slot++) {
            $segment = $segments[$slot]
            $name = [string] $segment.name
            $seen.Add($name)
            $bare = [ordered]@{}
            foreach ($property in $segment.PSObject.Properties) { if ($property.Name -cnotin 'slug', 'predictedMinutes') { $bare[$property.Name] = $property.Value } }
            Assert-Equal $expected[$name] (ConvertTo-SliceJson ([pscustomobject] $bare)) "$Label`: packed slice '$name' is not the slice it was"
            Assert-Equal (ConvertTo-ShardSlug $name) ([string] $segment.slug) "$Label`: packed slice '$name' has the wrong slug"
            Assert-Equal $name ([string] $entry."slot$slot".name) "$Label`: slot$slot does not name its slice"
            Assert-Equal ([string] $segment.slug) ([string] $entry."slot$slot".slug) "$Label`: slot$slot has the wrong slug"
            Assert-True (-not $segment.PSObject.Properties['timeout']) "$Label`: '$name' has its own timeout and was packed"
            Assert-True (-not $segment.PSObject.Properties['workload']) "$Label`: '$name' is not an ordinary workload and was packed"
            Assert-True ($name -cnotin $AloneNames) "$Label`: timing shard '$name' was packed"
            Assert-True ($source[$name] -cne 'default') "$Label`: '$name' has no timing and was packed"
            Assert-Equal ([string] $entry.project) ([string] $segment.project) "$Label`: '$name' is packed with another project"
            Assert-Equal ([string] $entry.framework) ([string] $segment.framework) "$Label`: '$name' is packed with another framework"
            $runner = if ($segment.PSObject.Properties['runner']) { [string] $segment.runner } else { '' }
            $packRunner = if ($entry.PSObject.Properties['runner']) { [string] $entry.runner } else { '' }
            Assert-Equal $packRunner $runner "$Label`: '$name' is packed with another runner class"
            $heavy = ($name -cin $HeavyNames) -or ($segment.PSObject.Properties['heavy'] -and $segment.heavy -eq $true)
            $weights[$heavy] = $true
            $sum += $predicted[$name]
        }
        Assert-True (-not $entry.PSObject.Properties["slot$($segments.Count)"]) "$Label`: a packed job names a slot it has no slice for"
        Assert-True ($weights.Count -eq 1) "$Label`: packed job '$($entry.name)' mixes heavy and ordinary slices"
        $packHeavy = [bool] ($entry.PSObject.Properties['heavy'] -and $entry.heavy -eq $true)
        Assert-Equal ([bool] @($weights.Keys)[0]) $packHeavy "$Label`: packed job '$($entry.name)' has the wrong weight"
        Assert-True ($sum -le $Target + 0.011) "$Label`: packed job '$($entry.name)' is predicted at $sum minutes, over $Target"
        Assert-True ([int] $entry.timeout -ge $script:ShardPackDefaultTimeoutMinutes -and [int] $entry.timeout -le $script:ShardPackMaxTimeoutMinutes) `
            "$Label`: packed job '$($entry.name)' has timeout $($entry.timeout)"
    }
    Assert-Equal $Slices.Count $seen.Count "$Label`: the packed matrix does not hold every slice exactly once"
    Assert-Equal $Slices.Count @($seen | Sort-Object -Unique -CaseSensitive).Count "$Label`: a slice is in the packed matrix twice"
    foreach ($name in $seen) { Assert-True $expected.ContainsKey($name) "$Label`: '$name' was not selected" }
}

# ------------------------------------------------------------------ the flag and the scope

foreach ($case in @(
        @{ Value = $null; Mode = 'off' }, @{ Value = ''; Mode = 'off' }, @{ Value = 'off'; Mode = 'off' }, @{ Value = 'OFF'; Mode = 'off' },
        @{ Value = 'true'; Mode = 'off' }, @{ Value = '1'; Mode = 'off' }, @{ Value = 'enabled'; Mode = 'off' }, @{ Value = 'on,report'; Mode = 'off' },
        @{ Value = 'report'; Mode = 'report' }, @{ Value = ' Report '; Mode = 'report' }, @{ Value = 'on'; Mode = 'on' }, @{ Value = 'ON'; Mode = 'on' })) {
    Assert-Equal $case.Mode (Get-ShardRepackMode $case.Value) "TEST_IMPACT_REPACK='$($case.Value)' was misread"
}
Assert-True (Get-ShardRepackScope -EventName 'pull_request' -CoverageEverywhere '').InScope 'a pull request run is out of scope'
Assert-True (Get-ShardRepackScope -EventName 'merge_group' -CoverageEverywhere 'false').InScope 'a merge queue run is out of scope'
foreach ($event in 'push', 'schedule', 'workflow_dispatch', '', 'Pull_Request') {
    Assert-True (-not (Get-ShardRepackScope -EventName $event -CoverageEverywhere '').InScope) "a '$event' run is in scope"
}
Assert-True (-not (Get-ShardRepackScope -EventName 'pull_request' -CoverageEverywhere 'true').InScope) 'a coverage-everywhere run is in scope'
Assert-Equal 'Unit_03b4_Diffusion_A_M' (ConvertTo-ShardSlug 'Unit - 03b4 Diffusion/A-M') 'the slug is not the artifact slug'
Assert-Equal 'Ns.Class' (Get-TestClassName 'Ns.Class.Method') 'a method was not reduced to its class'
Assert-Equal 'Ns.Class' (Get-TestClassName 'Ns.Class.Method(x: 1.5, name: "a.b")') 'a theory row was not reduced to its class'

# ------------------------------------------------------------------ the timing table and the prediction

$scratch = Join-Path ([IO.Path]::GetTempPath()) ("shard-packing-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch -Force | Out-Null
try {
    function Write-Ledger([string] $Path, [object[]] $Shards, [object[]] $Tests) {
        ConvertTo-Json -Depth 5 -InputObject ([ordered]@{ schemaVersion = 1; sha = 'abc123'; generatedUtc = '2026-11-13T00:00:00Z'; shards = $Shards; tests = $Tests }) |
            Set-Content -LiteralPath $Path -Encoding utf8
    }
    function New-LedgerTest([string] $Shard, [string] $Name, [string] $Duration, [string] $Outcome = 'Passed', [string] $Identity = '') {
        [ordered]@{ identity = $(if ($Identity) { $Identity } else { $Name }); fullyQualifiedName = $Name; outcome = $Outcome; shard = $Shard; duration = $Duration }
    }
    $newest = Join-Path $scratch 'ledger-newest.json'
    Write-Ledger $newest @(
        [ordered]@{ key = 'A_1'; name = 'A - 1'; status = 'Passed' },
        [ordered]@{ key = 'B_2'; name = 'B - 2'; status = 'Incomplete' },
        [ordered]@{ key = 'C_3'; name = 'C - 3'; status = 'Failed'; testWallSeconds = 200 }
    ) @(
        (New-LedgerTest 'A_1' 'Ns.X.T1' '00:00:10'), (New-LedgerTest 'A_1' 'Ns.X.T1' '00:00:15'), (New-LedgerTest 'A_1' 'Ns.X.T2' '00:00:20'),
        (New-LedgerTest 'A_1' 'Ns.Y.T1(x: 1.5)' '00:00:30'), (New-LedgerTest 'A_1' 'Ns.Y.Skipped' '00:00:00' 'NotExecuted'),
        (New-LedgerTest 'B_2' 'Ns.Half.T' '00:10:00'), (New-LedgerTest 'C_3' 'Ns.Z.T' '00:01:40' 'Failed')
    )
    $first = Read-ShardTimingObservation -LedgerPath $newest -WallSeconds @{ A_1 = 114.0; C_3 = 999.0 } -Source 'newest'
    Assert-Equal 2 $first.shards.Count 'a shard that did not finish was timed, or a finished one was not'
    Assert-Equal 35.0 $first.shards['A_1'].classes['Ns.X'] 'a retried test was counted twice, or not at its longer run'
    Assert-Equal 30.0 $first.shards['A_1'].classes['Ns.Y'] 'a theory row was not counted in its class, or a skipped test was'
    Assert-Equal 114.0 $first.shards['A_1'].wallSeconds 'the test step duration was not used as the wall time'
    Assert-Equal 200.0 $first.shards['C_3'].wallSeconds "a packed slice's own wall time did not take precedence"

    $older = Join-Path $scratch 'ledger-older.json'
    Write-Ledger $older @([ordered]@{ key = 'A_1'; name = 'A - 1'; status = 'Passed' }) @((New-LedgerTest 'A_1' 'Ns.X.T1' '00:00:45'))
    $nightly = Join-Path $scratch 'ledger-nightly.json'
    Write-Ledger $nightly @([ordered]@{ key = 'A_1'; name = 'A - 1'; status = 'Passed' }) @(
        (New-LedgerTest 'A_1' 'Ns.X.T1' '00:16:40'), (New-LedgerTest 'A_1' 'Ns.W.T1' '00:00:50'))
    $built = New-ShardTimingTable -Observations @(
        $first,
        (Read-ShardTimingObservation -LedgerPath $older -Source 'older'),
        (Read-ShardTimingObservation -LedgerPath $nightly -Instrumented $true -Source 'nightly'))
    Assert-Equal 40.0 ([double] $built.shards['A_1'].classes['Ns.X']) 'a class is not the median of its uninstrumented timings'
    Assert-Equal 50.0 ([double] $built.shards['A_1'].classes['Ns.W']) 'a class only an instrumented run timed was left out'
    Assert-Equal 120.0 ([double] $built.shards['A_1'].seconds) "a shard's seconds are not the sum of its classes"
    Assert-Equal ([Math]::Round((114.0 - 24.0) / 65.0, 3)) ([double] $built.shards['A_1'].wallFactor) 'the wall factor is not wall less invocation over TRX time'
    Assert-Equal 1.76 ([double] $built.shards['C_3'].wallFactor) "a packed slice's wall factor was not taken from its ledger row"
    Assert-True (-not $built.shards.Contains('B_2')) 'an unfinished shard reached the table'

    $tableFile = Join-Path $scratch 'table.json'
    ConvertTo-Json -InputObject $built -Depth 6 -Compress | Set-Content -LiteralPath $tableFile -Encoding utf8
    $table = Read-ShardTimingTable -Path $tableFile
    Assert-True ($null -ne $table) 'a table this code wrote could not be read back'
    Assert-True ($null -eq (Read-ShardTimingTable -Path (Join-Path $scratch 'absent.json') 6>$null)) 'a missing table was read'
    Set-Content -LiteralPath (Join-Path $scratch 'broken.json') -Value '{ not json'
    Assert-True ($null -eq (Read-ShardTimingTable -Path (Join-Path $scratch 'broken.json') 6>$null)) 'a broken table was read'
    Set-Content -LiteralPath (Join-Path $scratch 'other.json') -Value '{"schemaVersion":2,"shards":{}}'
    Assert-True ($null -eq (Read-ShardTimingTable -Path (Join-Path $scratch 'other.json') 6>$null)) 'a table of another schema version was read'

    $factor = [double] $table['shards']['A_1']['wallFactor']
    $whole = Get-ShardSlicePrediction -Slice (New-Slice 'A - 1') -Table $table -Classes $null
    Assert-Equal 'timed' $whole.Source 'a whole shard the table knows is not timed'
    Assert-Equal ([Math]::Round((24.0 + $factor * 120.0) / 60, 2)) $whole.Minutes 'a whole shard is not invocation plus factor times its seconds'
    $narrow = Get-ShardSlicePrediction -Slice (New-Slice 'A - 1') -Table $table -Classes @('Ns.Y')
    Assert-Equal ([Math]::Round((24.0 + $factor * 30.0) / 60, 2)) $narrow.Minutes 'a narrowed slice is not predicted from its own classes'
    $partial = Get-ShardSlicePrediction -Slice (New-Slice 'A - 1') -Table $table -Classes @('Ns.Y', 'Ns.Never.Seen')
    Assert-Equal 'partial' $partial.Source 'a class the table has never timed went unnoticed'
    Assert-Equal ([Math]::Round((24.0 + $factor * (30.0 + 60.0)) / 60, 2)) $partial.Minutes 'an untimed class did not count as the conservative default'
    foreach ($unknown in @(
            (Get-ShardSlicePrediction -Slice (New-Slice 'Never - Seen') -Table $table -Classes $null),
            (Get-ShardSlicePrediction -Slice (New-Slice 'A - 1') -Table $null -Classes @('Ns.Y')))) {
        Assert-Equal 'default' $unknown.Source 'a slice with no timing was not marked as such'
        Assert-Equal $script:ShardPackTargetMinutes $unknown.Minutes 'a slice with no timing is not predicted to fill a job'
    }

    # ------------------------------------------------------------------ the packing rules, one at a time

    function Get-Packing {
        param([object[]] $Slices, [hashtable] $MinutesByShard, [string[]] $HeavyNames = @(), [string[]] $AloneNames = @(), [string] $Label)
        $result = New-ShardPackPlan -Slices $Slices -Table (New-Table $MinutesByShard) -HeavyNames $HeavyNames -AloneNames $AloneNames -Mode 'on' -EventName 'pull_request'
        Assert-PackInvariants -Slices $Slices -Matrix @($result.Matrix) -Plan $result.Plan -HeavyNames $HeavyNames -AloneNames $AloneNames -Label $Label
        return $result
    }
    function Get-Shape($Result) {
        # '3' for a packed job of three slices, the shard name for a job that stayed alone.
        (@($Result.Matrix | ForEach-Object { if ($_.PSObject.Properties['segments']) { [string] @($_.segments).Count } else { [string] $_.name } }) -join ' | ')
    }
    function Get-Minutes([string[]] $Names, [double] $Each) { $map = @{}; foreach ($name in $Names) { $map[$name] = $Each }; return $map }

    $five = @('S1', 'S2', 'S3', 'S4', 'S5')
    $r = Get-Packing -Slices @($five | ForEach-Object { New-Slice $_ }) -MinutesByShard (Get-Minutes $five 5) -Label 'fill'
    Assert-Equal '4 | S5' (Get-Shape $r) 'five 5-minute slices are not one 20-minute job and one alone'
    Assert-Equal 75 ([int] $r.Matrix[0].timeout) 'a packed job predicted at 20 minutes does not get fixed time plus three times that'
    Assert-Equal 2 @($r.Plan.jobs).Count 'the plan does not list every job'
    Assert-Equal 1 ([int] $r.Plan.packedJobs) 'the plan miscounts packed jobs'

    $r = Get-Packing -Slices @((New-Slice 'Heavy By Name'), (New-Slice 'Heavy By Flag' @{ heavy = $true }), (New-Slice 'Light 1'), (New-Slice 'Light 2')) `
        -MinutesByShard (Get-Minutes @('Heavy By Name', 'Heavy By Flag', 'Light 1', 'Light 2') 1) -HeavyNames @('Heavy By Name') -Label 'heavy'
    Assert-Equal '2 | 2' (Get-Shape $r) 'heavy slices were not kept apart from ordinary ones'
    Assert-True ($r.Matrix[0].heavy -eq $true -and -not $r.Matrix[1].PSObject.Properties['heavy']) 'a packed job does not carry the weight of its slices'

    $r = Get-Packing -Slices @((New-Slice 'Own 1' @{ runner = 'aidotnet-ci' }), (New-Slice 'Hosted 1'), (New-Slice 'Own 2' @{ runner = 'aidotnet-ci' }), (New-Slice 'Hosted 2')) `
        -MinutesByShard (Get-Minutes @('Own 1', 'Own 2', 'Hosted 1', 'Hosted 2') 1) -Label 'runner'
    Assert-Equal '2 | 2' (Get-Shape $r) 'self-hosted and hosted slices share a job'
    Assert-Equal 'aidotnet-ci' ([string] $r.Matrix[0].runner) 'a packed job of self-hosted slices is not routed to the self-hosted runner'
    Assert-True (-not $r.Matrix[1].PSObject.Properties['runner']) 'a packed job of hosted slices is routed to a self-hosted runner'

    $r = Get-Packing -Slices @((New-Slice 'Long' @{ timeout = 60 }), (New-Slice 'P1'), (New-Slice 'P2')) -MinutesByShard (Get-Minutes @('Long', 'P1', 'P2') 1) -Label 'timeout'
    Assert-Equal 'Long | 2' (Get-Shape $r) 'a slice with its own job timeout was packed'
    $r = Get-Packing -Slices @((New-Slice 'Timing'), (New-Slice 'P1'), (New-Slice 'P2')) -MinutesByShard (Get-Minutes @('Timing', 'P1', 'P2') 1) -AloneNames @('Timing') -Label 'timing'
    Assert-Equal 'Timing | 2' (Get-Shape $r) 'a shard that measures timing was packed'
    $r = Get-Packing -Slices @((New-Slice 'Lane' @{ workload = 'paper-fidelity' }), (New-Slice 'P1'), (New-Slice 'P2')) -MinutesByShard (Get-Minutes @('Lane', 'P1', 'P2') 1) -Label 'workload'
    Assert-Equal 'Lane | 2' (Get-Shape $r) 'a workload that is not an ordinary test shard was packed'
    $r = Get-Packing -Slices @((New-Slice 'New Shard'), (New-Slice 'P1'), (New-Slice 'P2')) -MinutesByShard (Get-Minutes @('P1', 'P2') 1) -Label 'unknown'
    Assert-Equal 'New Shard | 2' (Get-Shape $r) 'a slice with no timing was packed'
    Assert-Equal 1 ([int] $r.Plan.defaultSlices) 'the plan does not count the slice with no timing'
    $r = Get-Packing -Slices @((New-Slice 'Big'), (New-Slice 'P1'), (New-Slice 'P2')) -MinutesByShard @{ Big = 31; P1 = 1; P2 = 1 } -Label 'big'
    Assert-Equal 'Big | 2' (Get-Shape $r) 'a slice predicted to fill a job was packed'

    $ten = @(1..10 | ForEach-Object { "T$_" })
    $r = Get-Packing -Slices @($ten | ForEach-Object { New-Slice $_ }) -MinutesByShard (Get-Minutes $ten 0.5) -Label 'slots'
    Assert-Equal "$($script:ShardPackMaxSegments) | 2" (Get-Shape $r) 'a packed job holds more slices than the workflow has slots'

    $r = Get-Packing -Slices @((New-Slice 'Main 1'), (New-Slice 'Serving 1' @{ project = 'tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj' }),
        (New-Slice 'Main 2'), (New-Slice 'Old 1' @{ framework = 'net8.0' })) -MinutesByShard (Get-Minutes @('Main 1', 'Main 2', 'Serving 1', 'Old 1') 1) -Label 'project'
    Assert-Equal '2 | Serving 1 | Old 1' (Get-Shape $r) 'slices of different projects or frameworks share a job'

    # The Conformance and Sweep shards: one test class split by an environment window. Windows may share a job,
    # each still with its own environment and hang timeout - Assert-PackInvariants compares every definition.
    $windows = @(0..2 | ForEach-Object { New-Slice "Sweep - Counts $_/8" @{ env = [pscustomobject]@{ AIDOTNET_PARAMETER_COUNT_SHARD = "$_" }; hangTimeout = '35min' } })
    $r = Get-Packing -Slices $windows -MinutesByShard (Get-Minutes @($windows.name) 2) -Label 'windows'
    Assert-Equal '3' (Get-Shape $r) 'environment-window slices that fit one job were not packed'
    Assert-Equal '0,1,2' (@($r.Matrix[0].segments | ForEach-Object { $_.env.AIDOTNET_PARAMETER_COUNT_SHARD }) -join ',') 'a packed window lost its environment'
    Assert-Equal '35min' ([string] $r.Matrix[0].segments[2].hangTimeout) 'a packed window lost its hang timeout'

    $long = @((New-Slice 'Wide 1' @{ filter = 'x' * 400 }), (New-Slice 'Wide 2' @{ filter = 'y' * 400 }), (New-Slice 'Thin 1'), (New-Slice 'Thin 2'))
    $groups = @(Group-ShardSlices -Slices $long -Minutes (Get-Minutes @($long.name) 1) -MaxSegmentCharacters 700)
    Assert-Equal '2,2' (@($groups | ForEach-Object { @($_.Slices).Count } | Sort-Object) -join ',') 'slices were not split by the text one packed job can carry'
    foreach ($group in $groups) {
        Assert-True ((@($group.Slices | ForEach-Object { (ConvertTo-SliceJson $_).Length }) | Measure-Object -Sum).Sum -le 700) 'a packed job carries more slice text than one environment string holds'
    }
    $groups = @(Group-ShardSlices -Slices $long -Minutes (Get-Minutes @($long.name) 1) -MaxSegmentCharacters 300)
    Assert-True (@($groups | Where-Object { @($_.Slices).Count -eq 1 -and $_.Alone -match 'too large' }).Count -eq 2) 'a slice too large to pass to a packed job was packed'

    $r = Get-Packing -Slices @() -MinutesByShard @{} -Label 'empty'
    Assert-Equal 0 @($r.Matrix).Count 'an empty matrix did not stay empty'
    $one = New-Slice 'Only'
    $r = Get-Packing -Slices @($one) -MinutesByShard @{ Only = 1 } -Label 'one'
    Assert-Equal (ConvertTo-SliceJson $one) (ConvertTo-SliceJson $r.Matrix[0]) 'a single slice is not the job it always was'

    Assert-Equal 60 (Get-ShardPackTimeout -Minutes 5) 'a short packed job has less than the default job timeout'
    Assert-Equal 75 (Get-ShardPackTimeout -Minutes 20) 'a full packed job does not have fixed time plus three times its prediction'
    Assert-Equal 120 (Get-ShardPackTimeout -Minutes 60) 'a packed job timeout is not capped'

    # With no table nothing is packed: the matrix is, text for text, the one it was given.
    $mixed = @((New-Slice 'M1'), (New-Slice 'M2' @{ heavy = $true }), (New-Slice 'M3' @{ env = [pscustomobject]@{ AIDOTNET_X = '1' }; hangTimeout = '20min' }),
        (New-Slice 'M4' @{ narrowed = $true; narrowedClasses = 3 }))
    $untimed = New-ShardPackPlan -Slices $mixed -Table $null -Mode 'on' -EventName 'pull_request'
    Assert-Equal (ConvertTo-Json -InputObject @($mixed) -Depth 6 -Compress) (ConvertTo-Json -InputObject @($untimed.Matrix) -Depth 8 -Compress) `
        'without a timing table the matrix is not the unpacked one'
    Assert-Equal 0 ([int] $untimed.Plan.packedJobs) 'something was packed without a timing table'

    # Narrowed slices are predicted from the plan's classes.
    $narrowTable = ('{"schemaVersion":1,"generatedUtc":"x","invocationSeconds":0,"defaultWallFactor":1,"sources":[],"shards":{"N1":{"seconds":3000,"wallFactor":1,' +
        '"classes":{"Ns.Big":2940,"Ns.Small":60}},"N2":{"seconds":3000,"wallFactor":1,"classes":{"Ns.Big":2940,"Ns.Small":60}}}}') | ConvertFrom-Json -AsHashtable
    $narrowSlices = @((New-Slice 'N1' @{ narrowed = $true; narrowedClasses = 1 }), (New-Slice 'N2' @{ narrowed = $true; narrowedClasses = 1 }))
    $planShards = @{ N1 = [pscustomobject]@{ testClasses = @('Ns.Small') }; N2 = [pscustomobject]@{ testClasses = @('Ns.Small') } }
    $r = New-ShardPackPlan -Slices $narrowSlices -Table $narrowTable -PlanShards $planShards -Mode 'on' -EventName 'pull_request'
    Assert-Equal 1 ([int] $r.Plan.packedJobs) 'two narrowed slices of a minute each were predicted as their whole 50-minute shards'
    $r = New-ShardPackPlan -Slices @((New-Slice 'N1'), (New-Slice 'N2')) -Table $narrowTable -PlanShards $planShards -Mode 'on' -EventName 'pull_request'
    Assert-Equal 0 ([int] $r.Plan.packedJobs) 'whole shards were predicted from the classes of a narrowing that was not applied'

    # ------------------------------------------------------------------ what the completeness check refuses

    $a = New-Slice 'A'; $b = New-Slice 'B'; $c = New-Slice 'C'
    $good = New-ShardPackPlan -Slices @($a, $b, $c) -Table (New-Table @{ A = 1; B = 1; C = 1 }) -Mode 'on' -EventName 'pull_request'
    Assert-ShardPackComplete -Slices @($a, $b, $c) -Matrix @($good.Matrix)
    Assert-Throws { Assert-ShardPackComplete -Slices @($a, $b, $c, (New-Slice 'D')) -Matrix @($good.Matrix) } 'a packed matrix that drops a slice'
    Assert-Throws { Assert-ShardPackComplete -Slices @($a, $b) -Matrix @($good.Matrix) } 'a packed matrix with a slice that was not selected'
    Assert-Throws { Assert-ShardPackComplete -Slices @($a, $b, $c) -Matrix (@($good.Matrix) + @($a)) } 'a packed matrix with a slice twice'
    $altered = $good.Matrix[0].PSObject.Copy()
    $altered.segments = @($good.Matrix[0].segments | ForEach-Object { $copy = $_.PSObject.Copy(); $copy.filter = 'FullyQualifiedName~Else'; $copy })
    Assert-Throws { Assert-ShardPackComplete -Slices @($a, $b, $c) -Matrix @($altered) } 'a packed slice whose filter changed'
    $named = New-Slice 'Pack 01 of 01 (2 slices)'
    Assert-Throws { Assert-ShardPackComplete -Slices @($named) -Matrix @($named) } 'a shard named like a packed job'
    Assert-Throws { Group-ShardSlices -Slices @($a, $a) -Minutes @{ A = 1 } } 'a slice listed twice'
    Assert-Throws { Group-ShardSlices -Slices @($a, $b) -Minutes @{ A = 1 } } 'a slice with no predicted time'

    # ------------------------------------------------------------------ set equality over a spread of matrices

    $random = [Random]::new(20261114)
    for ($round = 0; $round -lt 60; $round++) {
        $count = $random.Next(0, 70)
        $slices = [Collections.Generic.List[object]]::new()
        $minutes = @{}
        $heavyNames = [Collections.Generic.List[string]]::new()
        $aloneNames = [Collections.Generic.List[string]]::new()
        for ($index = 0; $index -lt $count; $index++) {
            $name = "R$round - shard $index/$count"
            $with = @{}
            if ($random.Next(10) -eq 0) { $with['project'] = 'tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj' }
            if ($random.Next(10) -lt 3) { $with['runner'] = 'aidotnet-ci' }
            if ($random.Next(10) -eq 0) { $with['heavy'] = $true }
            if ($random.Next(20) -eq 0) { $with['timeout'] = 60 }
            if ($random.Next(50) -eq 0) { $with['workload'] = 'other' }
            if ($random.Next(7) -eq 0) { $with['env'] = [pscustomobject]@{ ADNSHAPE_CONF_OFFSET = "$index" }; $with['hangTimeout'] = '20min' }
            if ($random.Next(2) -eq 0) { $with['narrowed'] = $true; $with['narrowedClasses'] = $random.Next(1, 40) }
            if ($random.Next(10) -eq 0) { $heavyNames.Add($name) }
            if ($random.Next(30) -eq 0) { $aloneNames.Add($name) }
            # One slice in ten has no timing; the rest run from a few seconds to longer than a job.
            if ($random.Next(10) -ne 0) { $minutes[$name] = [Math]::Round([Math]::Pow($random.NextDouble(), 3) * 26 + 0.05, 2) }
            $slices.Add((New-Slice $name $with))
        }
        $result = New-ShardPackPlan -Slices $slices.ToArray() -Table (New-Table $minutes) -HeavyNames $heavyNames.ToArray() -AloneNames $aloneNames.ToArray() `
            -Mode 'on' -EventName 'pull_request'
        Assert-PackInvariants -Slices $slices.ToArray() -Matrix @($result.Matrix) -Plan $result.Plan -HeavyNames $heavyNames.ToArray() `
            -AloneNames $aloneNames.ToArray() -Label "generated matrix $round"
        Assert-True (@($result.Matrix).Count -le $count) "generated matrix $round`: packing made more jobs than slices"
        $back = @(Expand-PackedShardMatrix -Matrix @($result.Matrix))
        Assert-Equal (@($slices | ForEach-Object { ConvertTo-SliceJson $_ } | Sort-Object -CaseSensitive) -join "`n") (@($back | ForEach-Object { ConvertTo-SliceJson $_ } | Sort-Object -CaseSensitive) -join "`n") `
            "generated matrix $round`: expanding the packed matrix does not give back the slices"
    }

    # ------------------------------------------------------------------ reading a packed run

    function New-Job([string] $Name, [string] $Conclusion, [double] $JobMinutes, [double] $StepMinutes, [object[]] $MoreSteps = @(), [string[]] $Labels = @('ubuntu-latest')) {
        $start = [DateTimeOffset]::Parse('2026-11-13T10:00:00Z')
        $steps = @([pscustomobject]@{ name = 'Run tests (sharded) with coverage'; conclusion = $Conclusion
                started_at = $start.AddMinutes(2).ToString('o'); completed_at = $start.AddMinutes(2 + $StepMinutes).ToString('o') }) + @($MoreSteps)
        [pscustomobject]@{ id = 1; name = $Name; status = 'completed'; conclusion = $Conclusion; labels = $Labels; html_url = "https://example.invalid/$Name"
            started_at = $start.ToString('o'); completed_at = $start.AddMinutes($JobMinutes).ToString('o'); steps = $steps }
    }
    $packedJob = New-Job 'Tests (net10.0) - Pack 01 of 02 (2 slices)' 'failure' 12 6 @(
        [pscustomobject]@{ name = 'Packed slice 1: Unit - 01 Core'; conclusion = 'success'; started_at = '2026-11-13T10:08:00Z'; completed_at = '2026-11-13T10:08:01Z' },
        [pscustomobject]@{ name = 'Packed slice 2: Unit - 02 A/B'; conclusion = 'failure'; started_at = '2026-11-13T10:08:01Z'; completed_at = '2026-11-13T10:08:02Z' },
        [pscustomobject]@{ name = 'Upload coverage + test results (packed slice 1)'; conclusion = 'success'; started_at = $null; completed_at = $null })
    $queued = [pscustomobject]@{ id = 2; name = 'Tests (net10.0) - Pack 02 of 02 (3 slices)'; status = 'queued'; conclusion = $null; labels = @(); steps = @() }
    $plain = New-Job 'Tests (net10.0) - Unit - 03 Other' 'success' 9 4
    $build = [pscustomobject]@{ id = 3; name = 'Build'; status = 'completed'; conclusion = 'success'; labels = @('ubuntu-latest'); steps = @() }
    $expanded = @(Expand-PackedShardJobs -Jobs @($packedJob, $queued, $plain, $build))
    Assert-Equal 'Tests (net10.0) - Unit - 01 Core|Tests (net10.0) - Unit - 02 A/B|Tests (net10.0) - Pack 02 of 02 (3 slices)|Tests (net10.0) - Unit - 03 Other|Build' `
        (@($expanded.name) -join '|') 'a packed job was not read as one job per slice'
    Assert-Equal 'success|failure' (@($expanded[0].conclusion, $expanded[1].conclusion) -join '|') "a slice did not get its own step's conclusion"
    Assert-Equal 'Tests (net10.0) - Pack 01 of 02 (2 slices)' ([string] $expanded[0].packedIn) 'a slice does not say which job it ran in'
    Assert-True ($expanded[3] -eq $plain -and $expanded[4] -eq $build) 'a job that is not packed was changed'
    Assert-Equal 'failure' ([string] $packedJob.conclusion) "expanding changed the packed job's own record"

    # ------------------------------------------------------------------ predicted against actual

    $reportSlices = @((New-Slice 'Rep 1'), (New-Slice 'Rep 2'), (New-Slice 'Rep Big'))
    $reported = New-ShardPackPlan -Slices $reportSlices -Table (New-Table @{ 'Rep 1' = 4; 'Rep 2' = 6; 'Rep Big' = 30 }) -Mode 'report' -EventName 'pull_request'
    $reported.Plan['applied'] = $false
    $planRead = ConvertTo-Json -InputObject $reported.Plan -Depth 8 | ConvertFrom-Json
    $runJobs = @((New-Job 'Tests (net10.0) - Rep 1' 'success' 8 3), (New-Job 'Tests (net10.0) - Rep 2' 'success' 12 7), (New-Job 'Tests (net10.0) - Rep Big' 'failure' 40 35), $build)
    $report = Get-ShardPackReport -Plan $planRead -Jobs $runJobs
    Assert-Equal 'estimated' ([string] $report.runnerMinutesPackedIs) 'a report-mode estimate is not labelled as one'
    Assert-Equal 60.0 ([double] $report.runnerMinutesMeasured) 'the measured runner-minutes are not the sum of the jobs'
    Assert-Equal 5.0 ([double] $report.fixedMinutesHosted) 'the fixed cost is not the median of job minus test step'
    # Packed: the would-be job's two test steps (3 + 7) plus one fixed cost (5), and the job that stays alone (40).
    Assert-Equal 55.0 ([double] $report.runnerMinutesPacked) 'the packed estimate is not the test steps plus one fixed cost per job'
    $packedRow = @($report.jobs | Where-Object { $_.packed })[0]
    Assert-Equal 10.0 ([double] $packedRow.predictedMinutes) 'a would-be job is not predicted as the sum of its slices'
    Assert-Equal 10.0 ([double] $packedRow.actualMinutes) "a would-be job's actual time is not the sum of its slices' test steps"
    Assert-Equal 1.0 ([double] $report.sliceMedianAbsoluteError) 'the prediction error per slice is wrong'
    Assert-True ((Get-ShardPackReportSummary -Report $report) -match 'estimated') 'the summary does not say the packed minutes are an estimate'

    $reported.Plan['applied'] = $true
    $reported.Plan['mode'] = 'on'
    $planRead = ConvertTo-Json -InputObject $reported.Plan -Depth 8 | ConvertFrom-Json
    $packName = [string] @($planRead.jobs | Where-Object { $_.packed })[0].name
    $report = Get-ShardPackReport -Plan $planRead -Jobs @((New-Job "Tests (net10.0) - $packName" 'success' 16 11), (New-Job 'Tests (net10.0) - Rep Big' 'failure' 40 35))
    Assert-Equal 'measured' ([string] $report.runnerMinutesPackedIs) 'a packed run is not reported as measured'
    Assert-Equal 11.0 ([double] @($report.jobs | Where-Object { $_.packed })[0].actualMinutes) "a packed job's actual time is not its test step"
    Assert-Equal 56.0 ([double] $report.runnerMinutesPacked) 'a packed run does not report the minutes its jobs took'

    # ------------------------------------------------------------------ the workflow's side of the contract

    $workflowPath = Join-Path $repositoryRoot '.github/workflows/sonarcloud.yml'
    $workflow = Get-Content -LiteralPath $workflowPath -Raw
    $shardScript = Get-Content -LiteralPath (Join-Path $repositoryRoot '.github/scripts/Invoke-Shard.ps1') -Raw
    $slugRule = "-replace '[\\/:*?`"<>|\s-]+', '_'"
    Assert-True ($shardScript.Contains($slugRule) -and $workflow.Contains($slugRule)) 'the artifact slug rule this file copies is no longer the one the workflow and Invoke-Shard.ps1 use'
    Assert-True ($workflow.Contains("- name: $($script:ShardPackTestStepName)`n") -or $workflow.Contains("- name: $($script:ShardPackTestStepName)`r`n")) `
        'the test step is no longer named what the timing code reads'
    Assert-True $workflow.Contains('SHARD_SEGMENTS: ${{ toJSON(matrix.shard.segments) }}') 'the test step does not pass a packed job its slices'
    Assert-True $workflow.Contains('TEST_IMPACT_REPACK: ${{ vars.TEST_IMPACT_REPACK }}') 'the Build job does not pass the flag to the selection'
    $max = $script:ShardPackMaxSegments
    foreach ($slot in 0..($max - 1)) {
        $number = $slot + 1
        foreach ($needle in @(
                "- name: `"Packed slice $number`: `${{ matrix.shard.slot$slot.name }}`"",
                "run: ./.github/scripts/Get-PackedSliceOutcome.ps1 -Slot $slot",
                "- name: Upload coverage + test results (packed slice $number)",
                "name: coverage-`${{ github.sha }}-`${{ matrix.shard.slot$slot.slug }}",
                "- name: Upload compact test diagnostics (packed slice $number)",
                "name: test-results-`${{ github.sha }}-`${{ matrix.shard.slot$slot.slug }}",
                "TestResults/slot-$slot/shard-metadata.json")) {
            Assert-True $workflow.Contains($needle) "the workflow has no '$needle' for slice $number of a packed job"
        }
    }
    Assert-True (-not $workflow.Contains("matrix.shard.slot$max")) "the workflow has more per-slice steps than ShardPackMaxSegments ($max)"
    Assert-Equal $max ([regex]::Matches($workflow, '(?m)^\s+- name: "Packed slice \d+: ').Count) 'the per-slice marker steps and ShardPackMaxSegments differ'

    $yq = Get-Command yq -ErrorAction SilentlyContinue
    if (-not $yq) {
        if ($env:GITHUB_ACTIONS -eq 'true') { throw 'shard packing self-test: yq is not available, so the workflow and manifest checks cannot run' }
        Write-Host 'yq not found: the structured workflow checks and the real manifest were NOT checked'
    }
    else {
        $jobs = yq -o=json -I=0 '.jobs' $workflowPath | ConvertFrom-Json -AsHashtable
        $testSteps = @($jobs['test-net10-sharded']['steps'])
        $byName = @{}
        foreach ($step in $testSteps) { if ($step.Contains('name')) { $byName[[string] $step['name']] = $step } }
        # Every step that reads or publishes the job's results as ONE shard is skipped in a packed job...
        foreach ($name in 'Download legacy retry baseline shard TRX', 'Select PR-new failures for targeted retry', 'Record shard outcome',
            'Build test-impact digest', 'Upload coverage + test results', 'Upload compact test diagnostics') {
            Assert-True $byName.ContainsKey($name) "the test job has no '$name' step"
            Assert-True ([string] $byName[$name]['if']).Contains('!matrix.shard.segments') "'$name' would run in a packed job, over several shards at once"
        }
        # ...and every step added for a packed job is skipped in every other one, so an unpacked job is unchanged.
        foreach ($step in $testSteps) {
            $name = if ($step.Contains('name')) { [string] $step['name'] } else { '' }
            if ($name -notmatch '(?i)packed slice') { continue }
            $condition = if ($step.Contains('if')) { [string] $step['if'] } else { '' }
            Assert-True ($condition -cmatch 'matrix\.shard\.(slot\d+|segments)\b' -and $condition -cnotmatch '!matrix\.shard\.(slot\d+|segments)') `
                "'$name' is not limited to packed jobs"
            if ($name -cmatch '^Upload .*\(packed slice (\d+)\)$') {
                $slot = [int] $Matches[1] - 1
                Assert-True $condition.Contains("matrix.shard.slot$slot") "'$name' is not conditional on its own slice"
                foreach ($path in @(([string] $step['with']['path']) -split "`n" | Where-Object { $_.Trim() })) {
                    Assert-True $path.Trim().StartsWith("TestResults/slot-$slot/") "'$name' uploads '$($path.Trim())', outside its own slice"
                }
            }
        }
        $runStep = $byName[$script:ShardPackTestStepName]
        Assert-Equal '${{ toJSON(matrix.shard.segments) }}' ([string] $runStep['env']['SHARD_SEGMENTS']) 'the test step does not receive the slices of a packed job'
        $gate = "(vars.TEST_IMPACT_REPACK == 'report' || vars.TEST_IMPACT_REPACK == 'on')"
        foreach ($case in @(
                @{ Job = 'build'; Step = 'Restore the shard timing table'; Needs = $gate },
                @{ Job = 'build'; Step = 'Upload the shard pack plan'; Needs = $gate },
                @{ Job = 'ci-test-analysis'; Step = 'Verify shard repacking'; Needs = "vars.TEST_IMPACT_REPACK == 'report' || vars.TEST_IMPACT_REPACK == 'on'" },
                @{ Job = 'ci-test-analysis'; Step = 'Report shard repacking'; Needs = $gate },
                @{ Job = 'ci-test-analysis'; Step = 'Upload the shard repacking report'; Needs = $gate },
                @{ Job = 'ci-test-analysis'; Step = 'Read packed jobs by shard'; Needs = "vars.TEST_IMPACT_REPACK == 'on'" })) {
            $step = @($jobs[$case.Job]['steps'] | Where-Object { $_.Contains('name') -and [string] $_['name'] -ceq $case.Step })
            Assert-Equal 1 $step.Count "the $($case.Job) job has no '$($case.Step)' step"
            Assert-True ([string] $step[0]['if']).Contains($case.Needs) "'$($case.Step)' runs when TEST_IMPACT_REPACK is unset or off"
            Assert-True ($step[0].Contains('continue-on-error') -and $step[0]['continue-on-error'] -eq $true) "'$($case.Step)' can fail its job"
        }
        $restore = @($jobs['build']['steps'] | Where-Object { $_.Contains('name') -and [string] $_['name'] -ceq 'Restore the shard timing table' })[0]
        $tableWorkflow = yq -o=json -I=0 '.jobs' (Join-Path $repositoryRoot '.github/workflows/shard-timing-table.yml') | ConvertFrom-Json -AsHashtable
        $save = @($tableWorkflow['build-table']['steps'] | Where-Object { $_.Contains('uses') -and [string] $_['uses'] -clike 'actions/cache/save@*' })
        Assert-Equal 1 $save.Count 'the timing table workflow does not save the table'
        Assert-True ([string] $restore['uses'] -clike 'actions/cache/restore@*') 'the Build job could write the timing table cache'
        Assert-Equal ([string] $save[0]['with']['path']) ([string] $restore['with']['path']) 'the table is saved and restored under different paths, so it would never be found'
        Assert-True ([string] $save[0]['with']['key']).StartsWith(([string] $restore['with']['restore-keys']).Trim()) 'the table is saved under a key the Build job does not restore'

        # The real manifest: every shard, every rule, and nothing dropped.
        $manifest = @(yq -o=json -I=0 '.shard' (Join-Path $repositoryRoot '.github/test-shards.yml') | ConvertFrom-Json)
        Assert-True ($manifest.Count -gt 100) 'the shard manifest was not read'
        $weights = Get-ShardPackWeights -ShardScriptPath (Join-Path $repositoryRoot '.github/scripts/Invoke-Shard.ps1')
        Assert-True (@($weights.Heavy).Count -gt 10) 'the heavy shard list of Invoke-Shard.ps1 was not read'
        Assert-True (@($weights.Timing).Count -gt 0) 'the timing shard list of Invoke-Shard.ps1 was not read'
        foreach ($shard in $manifest) {
            Assert-True ([string] $shard.name -cnotmatch $script:ShardPackJobNamePattern) "shard '$($shard.name)' is named like a packed job"
        }
        Assert-Equal $manifest.Count @($manifest | ForEach-Object { ConvertTo-ShardSlug $_.name } | Sort-Object -Unique -CaseSensitive).Count 'two shards share an artifact slug'
        $realMinutes = @{}
        for ($index = 0; $index -lt $manifest.Count; $index++) { $realMinutes[[string] $manifest[$index].name] = 0.5 + ($index % 9) }
        $real = New-ShardPackPlan -Slices $manifest -Table (New-Table $realMinutes) -HeavyNames $weights.Heavy -AloneNames $weights.Timing -Mode 'on' -EventName 'pull_request'
        Assert-PackInvariants -Slices $manifest -Matrix @($real.Matrix) -Plan $real.Plan -HeavyNames $weights.Heavy -AloneNames $weights.Timing -Label 'the real manifest'
        Assert-True ([int] $real.Plan.packedJobs -gt 5 -and @($real.Matrix).Count -lt $manifest.Count) 'the real manifest was not packed at all'
        foreach ($entry in @($real.Matrix | Where-Object { $_.PSObject.Properties['segments'] })) {
            Assert-True ((ConvertTo-Json -InputObject @($entry.segments) -Depth 8).Length -lt 120000) "packed job '$($entry.name)' is too large for one environment string"
        }
        $withTimeout = @($manifest | Where-Object { $_.PSObject.Properties['timeout'] })
        Assert-True ($withTimeout.Count -gt 0) 'the manifest has no shard with its own timeout to prove the rule on'
        foreach ($shard in $withTimeout) {
            Assert-True (@($real.Matrix | Where-Object { -not $_.PSObject.Properties['segments'] -and $_.name -ceq $shard.name }).Count -eq 1) "'$($shard.name)' has its own timeout and did not stay alone"
        }
    }

    if ($RulesOnly) {
        Write-Host "Shard packing rules self-test passed ($script:Checks checks)."
        exit 0
    }

    # ================================================================== the real scripts, end to end

    $saved = @{}
    $names = 'GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY', 'GITHUB_EVENT_NAME', 'GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'REQUIRES_SHARDS', 'DELTA_MODE',
        'COLLECT_COVERAGE_EVERYWHERE', 'SELECTED_MATRIX', 'MAP_LINEAGE_REF', 'TEST_IMPACT_REPACK', 'SHARD_TIMING_TABLE', 'SHARD_SEGMENTS', 'SHARD_PACK_TIMEOUT',
        'SHARD_RESULTS_ROOT', 'SHARD_NAME', 'SHARD_PROJECT', 'SHARD_FRAMEWORK', 'SHARD_FILTER', 'SHARD_HEAVY', 'SHARD_ENV', 'SHARD_HANG_TIMEOUT',
        'SHARD_COVERAGE_INCLUDE', 'SHARD_FORCE_COVERAGE', 'COVERAGE_CARRIED', 'COVERAGE_RUN_WITHOUT_INSTRUMENTATION', 'COVERAGE_ROLLING', 'STUB_LOG',
        'STUB_PACK_PID_FILE', 'PATH', 'AIDOTNET_STUB_WINDOW', 'DOTNET_gcServer', 'DOTNET_GCHeapHardLimit', 'RETRY_FILTER'
    foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
    try {
        foreach ($name in $names) { if ($name -cne 'PATH') { [Environment]::SetEnvironmentVariable($name, $null) } }
        $pwsh = [Environment]::ProcessPath

        # ---------------------------------------------------------------- Select-AffectedTests.ps1 and the flag

        $selector = (Resolve-Path (Join-Path $PSScriptRoot 'Select-AffectedTests.ps1')).Path
        $fixture = Join-Path $scratch 'selection'
        New-Item -ItemType Directory -Path (Join-Path $fixture '.github/scripts') -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $repositoryRoot '.github/scripts/Invoke-Shard.ps1') -Destination (Join-Path $fixture '.github/scripts/Invoke-Shard.ps1')
        $chosen = @((New-Slice 'Sel 1'), (New-Slice 'Sel 2'), (New-Slice 'Sel 3' @{ env = [pscustomobject]@{ AIDOTNET_X = '1' }; hangTimeout = '20min' }),
            (New-Slice 'Sel Heavy' @{ heavy = $true }), (New-Slice 'Sel Long' @{ timeout = 60 }), (New-Slice 'Sel New'))
        $chosenJson = ConvertTo-Json -InputObject $chosen -Depth 6 -Compress
        $chosenTable = Join-Path $fixture 'shard-timing-table.json'
        $writeTable = { ConvertTo-Json -InputObject (New-Table @{ 'Sel 1' = 2; 'Sel 2' = 3; 'Sel 3' = 4; 'Sel Heavy' = 1; 'Sel Long' = 1 }) -Depth 6 | Set-Content -LiteralPath $chosenTable -Encoding utf8 }
        & $writeTable
        $planFile = Join-Path $fixture 'shard-pack-plan.json'

        function Invoke-Selection {
            param([string] $Flag, [string] $EventName = 'merge_group', [string] $Extra = '')
            $output = Join-Path $fixture 'github-output.txt'
            Remove-Item -LiteralPath $output, $planFile -Force -ErrorAction SilentlyContinue
            $env:GITHUB_OUTPUT = $output
            $env:GITHUB_STEP_SUMMARY = Join-Path $fixture 'summary.md'
            $env:GITHUB_EVENT_NAME = $EventName
            $env:REQUIRES_SHARDS = 'true'
            $env:SELECTED_MATRIX = $chosenJson
            # A pull request reaches the same pass-through, before any git or TypeImpact work, as a stacked one.
            $env:MAP_LINEAGE_REF = if ($EventName -cin 'pull_request', 'push') { 'a-base-branch' } else { $null }
            [Environment]::SetEnvironmentVariable('TEST_IMPACT_REPACK', $Flag)
            $command = "Set-Location '$fixture'; & '$selector' -Repository '$fixture' $Extra; if (Test-Path -LiteralPath variable:\LASTEXITCODE) { exit `$LASTEXITCODE }"
            $log = & $pwsh -NoProfile -Command $command 2>&1 | Out-String
            Assert-Equal 0 $LASTEXITCODE "the selection step failed with TEST_IMPACT_REPACK='$Flag' on $EventName`: $log"
            $values = @{}
            foreach ($line in @(Get-Content -LiteralPath $output)) { $split = $line.IndexOf('='); if ($split -gt 0) { $values[$line.Substring(0, $split)] = $line.Substring($split + 1) } }
            return [pscustomobject]@{ Outputs = $values; Log = $log; Plan = $(if (Test-Path -LiteralPath $planFile) { Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json } else { $null }) }
        }
        function Format-Outputs($Run) { (@($Run.Outputs.Keys | Sort-Object | ForEach-Object { "$_=$($Run.Outputs[$_])" }) -join "`n") }

        $baseline = Invoke-Selection -Flag $null
        Assert-Equal '' $baseline.Outputs['matrix'] 'the pass-through published a matrix before repacking existed'
        Assert-Equal 'passthrough' $baseline.Outputs['impact_mode'] 'the fixture did not reach the pass-through'
        foreach ($flag in '', 'off', 'OFF', 'true', 'banana') {
            $run = Invoke-Selection -Flag $flag
            Assert-Equal (Format-Outputs $baseline) (Format-Outputs $run) "TEST_IMPACT_REPACK='$flag' changed what the selection publishes"
            Assert-True ($null -eq $run.Plan) "TEST_IMPACT_REPACK='$flag' computed a packing"
            Assert-True ($run.Log -notmatch 'repack') "TEST_IMPACT_REPACK='$flag' ran the repacking code"
        }

        foreach ($event in 'merge_group', 'pull_request') {
            $run = Invoke-Selection -Flag 'report' -EventName $event
            Assert-Equal ((Format-Outputs $baseline) -replace "event 'merge_group'", "event '$event'") (Format-Outputs $run) "report mode changed what the selection publishes on $event"
            Assert-True ($null -ne $run.Plan -and $run.Plan.applied -eq $false -and $run.Plan.mode -ceq 'report') "report mode recorded no plan on $event"
            Assert-Equal 1 ([int] $run.Plan.packedJobs) "report mode did not compute the packing on $event"
            Assert-Equal 6 ([int] $run.Plan.slices) "the plan does not cover every slice on $event"
        }

        foreach ($event in 'merge_group', 'pull_request') {
            $run = Invoke-Selection -Flag 'on' -EventName $event
            Assert-True ($run.Outputs['matrix'].Length -gt 0) "on mode did not pack on $event`: $($run.Log)"
            $packedMatrix = @($run.Outputs['matrix'] | ConvertFrom-Json)
            Assert-Equal '3|Sel Heavy|Sel Long|Sel New' (@($packedMatrix | ForEach-Object { if ($_.PSObject.Properties['segments']) { [string] @($_.segments).Count } else { [string] $_.name } }) -join '|') `
                "on mode did not pack the three ordinary timed slices and leave the rest alone on $event"
            Assert-Equal (@($chosen | ForEach-Object { ConvertTo-SliceJson $_ } | Sort-Object -CaseSensitive) -join "`n") `
                (@(Expand-PackedShardMatrix -Matrix $packedMatrix | ForEach-Object { ConvertTo-SliceJson $_ } | Sort-Object -CaseSensitive) -join "`n") `
                "the published packed matrix does not run exactly the chosen slices on $event"
            foreach ($output in 'ledger_matrix', 'skipped', 'impact_mode') {
                Assert-Equal $baseline.Outputs[$output] $run.Outputs[$output] "on mode changed the $output output on $event"
            }
            Assert-True ($run.Plan.applied -eq $true) "the plan does not record that it was applied on $event"
            Assert-True ($run.Log -match 'Shard packing rules self-test passed') 'a packed matrix was published without the rules passing their self-test first'
        }

        # Every way out of packing is one job per shard: exactly what the flag being off publishes.
        $capped = Invoke-Selection -Flag 'on' -Extra '-MaxMatrixCharacters 200'
        Assert-Equal '' $capped.Outputs['matrix'] 'a packed matrix over the job-output cap was published'
        Assert-True ($capped.Plan.applied -eq $false) 'a packed matrix over the cap is recorded as applied'
        foreach ($event in 'push', 'schedule', 'workflow_dispatch') {
            $run = Invoke-Selection -Flag 'on' -EventName $event
            Assert-Equal '' $run.Outputs['matrix'] "a $event run was packed"
            Assert-True ($null -eq $run.Plan) "a $event run computed a packing"
        }
        $env:COLLECT_COVERAGE_EVERYWHERE = 'true'
        Assert-Equal '' (Invoke-Selection -Flag 'on' -EventName 'pull_request').Outputs['matrix'] 'a coverage-everywhere run was packed'
        $env:COLLECT_COVERAGE_EVERYWHERE = $null
        Remove-Item -LiteralPath $chosenTable -Force
        $run = Invoke-Selection -Flag 'on'
        Assert-Equal '' $run.Outputs['matrix'] 'slices were packed without a timing table'
        Assert-Equal 0 ([int] $run.Plan.packedJobs) 'the plan packs without a timing table'
        Set-Content -LiteralPath $chosenTable -Value '{ "schemaVersion": 1, "shards": [ broken'
        Assert-Equal '' (Invoke-Selection -Flag 'on').Outputs['matrix'] 'slices were packed from an unreadable timing table'
        & $writeTable
        $brokenScript = Join-Path $fixture '.github/scripts/Invoke-Shard.ps1'
        Remove-Item -LiteralPath $brokenScript -Force
        $run = Invoke-Selection -Flag 'on'
        Assert-Equal '' $run.Outputs['matrix'] 'an error while packing did not leave the matrix one job per shard'
        Assert-True ($run.Log -match '::warning::shard repacking unavailable') 'an error while packing was not reported'
        Copy-Item -LiteralPath (Join-Path $repositoryRoot '.github/scripts/Invoke-Shard.ps1') -Destination $brokenScript

        # The narrowed path calls the same function with the slices it kept and the TypeImpact plan's classes.
        $tokens = $null; $errors = $null
        $ast = [Management.Automation.Language.Parser]::ParseFile($selector, [ref] $tokens, [ref] $errors)
        $function = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Get-RepackedMatrix' }, $true)
        Assert-True ($null -ne $function) 'Select-AffectedTests.ps1 no longer has Get-RepackedMatrix'
        $script:PackingRoot = $PSScriptRoot
        . ([scriptblock]::Create($function.Extent.Text.Replace('$PSScriptRoot', '$script:PackingRoot')))
        $Repository = $fixture; $PackPlanFile = 'shard-pack-plan.json'; $MaxMatrixCharacters = 400000
        $env:GITHUB_EVENT_NAME = 'pull_request'
        $env:GITHUB_STEP_SUMMARY = Join-Path $fixture 'summary.md'
        ConvertTo-Json -InputObject $narrowTable -Depth 6 | Set-Content -LiteralPath $chosenTable -Encoding utf8
        foreach ($flag in $null, 'off') {
            [Environment]::SetEnvironmentVariable('TEST_IMPACT_REPACK', $flag)
            Remove-Item -LiteralPath $planFile -Force -ErrorAction SilentlyContinue
            Assert-True ($null -eq (Get-RepackedMatrix -Slices $narrowSlices -PlanShards $planShards -OtherOutputCharacters 10)) "a narrowed matrix was repacked with TEST_IMPACT_REPACK='$flag'"
            Assert-True (-not (Test-Path -LiteralPath $planFile)) "a narrowed matrix got a pack plan with TEST_IMPACT_REPACK='$flag'"
        }
        $env:TEST_IMPACT_REPACK = 'report'
        Assert-True ($null -eq (Get-RepackedMatrix -Slices $narrowSlices -PlanShards $planShards -OtherOutputCharacters 10)) 'report mode repacked a narrowed matrix'
        Assert-Equal 1 ([int] (Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json).packedJobs) 'report mode did not record the packing of a narrowed matrix'
        $env:TEST_IMPACT_REPACK = 'on'
        $narrowPacked = Get-RepackedMatrix -Slices $narrowSlices -PlanShards $planShards -OtherOutputCharacters 10
        Assert-True ($narrowPacked) 'on mode did not repack a narrowed matrix'
        Assert-Equal (@($narrowSlices | ForEach-Object { ConvertTo-SliceJson $_ }) -join "`n") (@(Expand-PackedShardMatrix -Matrix @($narrowPacked | ConvertFrom-Json) | ForEach-Object { ConvertTo-SliceJson $_ }) -join "`n") `
            'a repacked narrowed matrix does not run exactly the narrowed slices'
        Assert-True ($null -eq (Get-RepackedMatrix -Slices $narrowSlices -PlanShards $planShards -OtherOutputCharacters 400000 6>$null)) 'a packed matrix that does not fit beside ledger_matrix was published'
        Assert-True ($null -eq (Get-RepackedMatrix -Slices @($narrowSlices[0], $narrowSlices[0]) -PlanShards $planShards 6>$null)) 'a matrix the packer refuses was published'
        $env:TEST_IMPACT_REPACK = $null

        # ---------------------------------------------------------------- Write-ShardPackReport.ps1

        $reporter = Join-Path $PSScriptRoot 'Write-ShardPackReport.ps1'
        $jobsFile = Join-Path $scratch 'current-jobs.json'
        $rawFile = Join-Path $scratch 'current-jobs-raw.json'
        # The aggregate analysis job writes the job list as one array; an unpacked run's must come back untouched.
        ConvertTo-Json -InputObject @($plain, $build) -Depth 8 | Set-Content -LiteralPath $jobsFile -Encoding utf8
        $before = [IO.File]::ReadAllBytes($jobsFile)
        & $pwsh -NoProfile -File $reporter -JobsFile $jobsFile -ExpandJobs -RawJobsFile $rawFile | Out-Null
        Assert-Equal 0 $LASTEXITCODE 'expanding the job list of an unpacked run failed'
        Assert-True ([Linq.Enumerable]::SequenceEqual([byte[]] $before, [IO.File]::ReadAllBytes($jobsFile))) 'the job list of an unpacked run was rewritten'
        Assert-True (-not (Test-Path -LiteralPath $rawFile)) 'an unpacked run got a second copy of its job list'
        ConvertTo-Json -InputObject @($packedJob, $queued, $plain, $build) -Depth 8 | Set-Content -LiteralPath $jobsFile -Encoding utf8
        & $pwsh -NoProfile -File $reporter -JobsFile $jobsFile -ExpandJobs -RawJobsFile $rawFile | Out-Null
        Assert-Equal 0 $LASTEXITCODE 'expanding the job list of a packed run failed'
        $rewritten = @(Get-Content -LiteralPath $jobsFile -Raw | ConvertFrom-Json)
        Assert-Equal 'Tests (net10.0) - Unit - 01 Core:success|Tests (net10.0) - Unit - 02 A/B:failure' (@($rewritten[0..1] | ForEach-Object { "$($_.name):$($_.conclusion)" }) -join '|') `
            'the job list of a packed run was not rewritten slice by slice'
        Assert-Equal 5 $rewritten.Count 'the rewritten job list lost or gained a job'
        Assert-Equal 4 @(Get-Content -LiteralPath $rawFile -Raw | ConvertFrom-Json).Count 'the original job list of a packed run was not kept'

        $reported.Plan['applied'] = $false
        $reported.Plan['mode'] = 'report'
        $reportPlan = Join-Path $scratch 'shard-pack-plan.json'
        ConvertTo-Json -InputObject $reported.Plan -Depth 8 | Set-Content -LiteralPath $reportPlan -Encoding utf8
        ConvertTo-Json -InputObject $runJobs -Depth 8 | Set-Content -LiteralPath $jobsFile -Encoding utf8
        $reportFile = Join-Path $scratch 'shard-pack-report.json'
        $reportSummary = Join-Path $scratch 'report-summary.md'
        & $pwsh -NoProfile -File $reporter -JobsFile $jobsFile -PlanFile $reportPlan -OutFile $reportFile -SummaryFile $reportSummary | Out-Null
        Assert-Equal 0 $LASTEXITCODE 'the repacking report failed'
        $written = Get-Content -LiteralPath $reportFile -Raw | ConvertFrom-Json
        Assert-Equal '60|55|estimated' (@($written.runnerMinutesMeasured, $written.runnerMinutesPacked, $written.runnerMinutesPackedIs) -join '|') 'the published report is not the one computed'
        Assert-True ((Get-Content -LiteralPath $reportSummary -Raw) -match 'predicted against actual') 'the report did not reach the step summary'
        # Report-only: a missing plan, or a job list it cannot read, is a note and exit 0, never a failed step.
        & $pwsh -NoProfile -File $reporter -JobsFile $jobsFile -PlanFile (Join-Path $scratch 'no-plan.json') -OutFile $reportFile -SummaryFile $reportSummary | Out-Null
        Assert-Equal 0 $LASTEXITCODE 'a run with no pack plan failed the report step'
        Set-Content -LiteralPath $jobsFile -Value '{ broken'
        & $pwsh -NoProfile -File $reporter -JobsFile $jobsFile -PlanFile $reportPlan -OutFile $reportFile -SummaryFile $reportSummary | Out-Null
        Assert-Equal 0 $LASTEXITCODE 'an unreadable job list failed the report step'

        # ---------------------------------------------------------------- Invoke-Shard.ps1 over a packed job

        $workspace = Join-Path $scratch 'job'
        $bin = Join-Path $scratch 'bin'
        New-Item -ItemType Directory -Path (Join-Path $workspace '.github/scripts'), (Join-Path $workspace 'tests/Fake/bin/Release/net10.0'), $bin -Force | Out-Null
        Copy-Item -Path (Join-Path $repositoryRoot '.github/scripts/*.ps1') -Destination (Join-Path $workspace '.github/scripts')
        $runnerConfig = Join-Path $workspace 'tests/Fake/bin/Release/net10.0/xunit.runner.json'
        $runnerConfigText = '{"parallelizeTestCollections":true,"maxParallelThreads":4,"diagnosticMessages":true,"internalDiagnosticMessages":true}'
        [IO.File]::WriteAllText($runnerConfig, $runnerConfigText)
        $stub = @'
$filter = ''; $directory = ''; $hang = ''; $trxName = 'test-results.trx'
for ($i = 0; $i -lt $args.Count; $i++) {
    $argument = [string] $args[$i]
    if ($argument -ceq '--filter') { $filter = [string] $args[$i + 1] }
    elseif ($argument -ceq '--results-directory') { $directory = [string] $args[$i + 1] }
    elseif ($argument -ceq '--blame-hang-timeout') { $hang = [string] $args[$i + 1] }
    elseif ($argument -ceq '--logger' -and [string] $args[$i + 1] -cmatch '^trx;LogFileName=(.+)$') { $trxName = $Matches[1] }
}
$class = if ($filter -cmatch 'FullyQualifiedName~([A-Za-z0-9_.]+)') { $Matches[1] } else { 'Ns.Unknown' }
$retry = $trxName -ceq 'rerun-new-failures.trx'
$record = [ordered]@{
    shard = "$($env:SHARD_NAME)"; filter = $filter; results = ($directory -replace '\\', '/'); trx = $trxName; hang = $hang
    window = "$($env:AIDOTNET_STUB_WINDOW)"; gcServer = "$($env:DOTNET_gcServer)"; segments = "$($env:SHARD_SEGMENTS)"
    runnerConfig = [IO.File]::ReadAllText((Join-Path (Get-Location).Path 'tests/Fake/bin/Release/net10.0/xunit.runner.json'))
}
Add-Content -LiteralPath $env:STUB_LOG -Value (ConvertTo-Json -InputObject $record -Compress)
if ($class -cmatch 'KILLS') {
    # The job dies here: the process running the packed job is killed while this slice is in its test host.
    $until = [DateTime]::UtcNow.AddSeconds(30)
    while (-not (Test-Path -LiteralPath $env:STUB_PACK_PID_FILE) -and [DateTime]::UtcNow -lt $until) { Start-Sleep -Milliseconds 100 }
    Stop-Process -Id ([int] (Get-Content -LiteralPath $env:STUB_PACK_PID_FILE -Raw)) -Force
    Start-Sleep -Milliseconds 500
    Add-Content -LiteralPath $env:STUB_LOG -Value '{"killed":true}'
    exit 137
}
$failed = ($class -cmatch 'FAILS') -and -not $retry
$outcome = if ($failed) { 'Failed' } else { 'Passed' }
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$trx = @"
<?xml version="1.0" encoding="utf-8"?>
<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">
  <TestDefinitions><UnitTest name="Works" id="test-1"><TestMethod className="$class" name="Works" /></UnitTest></TestDefinitions>
  <Results><UnitTestResult executionId="run-1" testId="test-1" testName="Works" outcome="$outcome" duration="00:00:01.5000000" /></Results>
  <ResultSummary outcome="$outcome"><Counters total="1" executed="1" passed="$(1 - [int] $failed)" failed="$([int] $failed)" notExecuted="0" /></ResultSummary>
</TestRun>
"@
Set-Content -LiteralPath (Join-Path $directory $trxName) -Value $trx -Encoding utf8
exit $(if ($failed) { 1 } else { 0 })
'@
        if ($IsWindows) {
            Set-Content -LiteralPath (Join-Path $bin 'dotnet.ps1') -Value $stub -Encoding utf8
            # Invoke-Shard.ps1 prints the runner's memory and disk with these; a Windows machine has none of them.
            foreach ($tool in 'free', 'df', 'nproc') { Set-Content -LiteralPath (Join-Path $bin "$tool.ps1") -Value "Write-Host '$tool (stub)'" -Encoding utf8 }
        }
        else {
            [IO.File]::WriteAllText((Join-Path $bin 'dotnet'), "#!/usr/bin/env pwsh`n" + ($stub -replace "`r`n", "`n"))
            & chmod +x (Join-Path $bin 'dotnet')
            if ($LASTEXITCODE -ne 0) { throw 'shard packing self-test: the dotnet stub could not be made executable' }
        }
        $env:PATH = $bin + [IO.Path]::PathSeparator + $env:PATH
        Assert-True ((Get-Command dotnet).Source.StartsWith($bin)) 'the dotnet stub is not the dotnet this test would run'
        $stubLog = Join-Path $scratch 'stub.log'
        $env:STUB_LOG = $stubLog
        $env:GITHUB_EVENT_NAME = 'pull_request'
        $env:GITHUB_SHA = 'abc123'
        $env:GITHUB_RUN_ID = '42'
        $env:GITHUB_RUN_ATTEMPT = '1'
        $env:GITHUB_STEP_SUMMARY = Join-Path $scratch 'job-summary.md'
        $env:GITHUB_OUTPUT = Join-Path $scratch 'job-output.txt'
        $invoke = Join-Path $workspace '.github/scripts/Invoke-Shard.ps1'
        $project = 'tests/Fake/Fake.csproj'

        function Reset-Job {
            Remove-Item -LiteralPath (Join-Path $workspace 'TestResults'), $stubLog -Recurse -Force -ErrorAction SilentlyContinue
            [IO.File]::WriteAllText($runnerConfig, $runnerConfigText)
        }
        function Invoke-Job {
            Push-Location $workspace
            try {
                $log = & $pwsh -NoProfile -File $invoke 2>&1 | Out-String
                return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Log = $log }
            }
            finally { Pop-Location }
        }
        function Get-StubCalls { if (Test-Path -LiteralPath $stubLog) { @(Get-Content -LiteralPath $stubLog | ForEach-Object { $_ | ConvertFrom-Json }) } else { @() } }
        function Get-SlotMetadata([int] $Slot) { Get-Content -LiteralPath (Join-Path $workspace "TestResults/slot-$Slot/shard-metadata.json") -Raw | ConvertFrom-Json }
        function Get-SliceOutcome([int] $Slot) {
            Push-Location $workspace
            try { & $pwsh -NoProfile -File (Join-Path $workspace '.github/scripts/Get-PackedSliceOutcome.ps1') -Slot $Slot | Out-Null; return $LASTEXITCODE }
            finally { Pop-Location }
        }
        function New-Segment([string] $Name, [string] $Class, [hashtable] $With = @{}) {
            $segment = [ordered]@{ name = $Name; framework = 'net10.0'; project = $project; filter = "FullyQualifiedName~$Class" }
            foreach ($key in $With.Keys) { $segment[$key] = $With[$key] }
            $segment['slug'] = ConvertTo-ShardSlug $Name
            $segment['predictedMinutes'] = 1.5
            return [pscustomobject] $segment
        }

        # An unpacked job: SHARD_SEGMENTS is the text `null` (toJSON of a property the matrix entry does not have),
        # or absent. It runs as it always did, into TestResults/<slug>.
        foreach ($absent in 'null', '', $null) {
            Reset-Job
            [Environment]::SetEnvironmentVariable('SHARD_SEGMENTS', $absent)
            $env:SHARD_NAME = 'Unit - 01 Plain'; $env:SHARD_PROJECT = $project; $env:SHARD_FRAMEWORK = 'net10.0'
            $env:SHARD_FILTER = 'FullyQualifiedName~Ns.Plain'; $env:SHARD_ENV = 'null'; $env:SHARD_HEAVY = ''; $env:SHARD_HANG_TIMEOUT = ''
            $run = Invoke-Job
            Assert-Equal 0 $run.ExitCode "an unpacked job failed with SHARD_SEGMENTS='$absent'`: $($run.Log)"
            $calls = @(Get-StubCalls)
            Assert-Equal 1 $calls.Count "an unpacked job did not run dotnet test exactly once with SHARD_SEGMENTS='$absent'"
            Assert-Equal 'TestResults/Unit_01_Plain' ([string] $calls[0].results) 'an unpacked job no longer writes to TestResults/<slug>'
            Assert-True (-not (Test-Path -LiteralPath (Join-Path $workspace 'TestResults/slot-0'))) 'an unpacked job made a packed slice directory'
            Assert-True (-not (Test-Path -LiteralPath (Join-Path $workspace 'TestResults/shard-metadata.json'))) 'an unpacked job wrote the outcome record the workflow writes'
        }
        $env:SHARD_FILTER = 'FullyQualifiedName~Ns.FAILS'
        Reset-Job
        Assert-Equal 1 (Invoke-Job).ExitCode "an unpacked job did not return its tests' exit code"

        # A packed job of three slices: a heavy one, one that fails and has an environment window, one after it.
        $segments = @(
            (New-Segment 'Pack - A heavy' 'Ns.Alpha' @{ heavy = $true }),
            (New-Segment 'Pack - B fails/window' 'Ns.FAILS' @{ env = [pscustomobject]@{ AIDOTNET_STUB_WINDOW = '3' }; hangTimeout = '20min' }),
            (New-Segment 'Pack - C after' 'Ns.Gamma'))
        Reset-Job
        $env:SHARD_SEGMENTS = ConvertTo-Json -InputObject $segments -Depth 6
        $env:SHARD_NAME = 'Pack 01 of 01 (3 slices)'; $env:SHARD_PROJECT = $project; $env:SHARD_FRAMEWORK = 'net10.0'
        $env:SHARD_FILTER = ''; $env:SHARD_ENV = 'null'; $env:SHARD_HEAVY = ''; $env:SHARD_HANG_TIMEOUT = ''; $env:SHARD_PACK_TIMEOUT = '60'
        $run = Invoke-Job
        Assert-Equal 1 $run.ExitCode "a packed job with a failing slice did not fail: $($run.Log)"
        $calls = @(Get-StubCalls)
        Assert-Equal 'Pack - A heavy|Pack - B fails/window|Pack - C after' (@($calls.shard) -join '|') 'a failing slice stopped the slices after it, or the order changed'
        Assert-Equal 'TestResults/slot-0/Pack_A_heavy|TestResults/slot-1/Pack_B_fails_window|TestResults/slot-2/Pack_C_after' (@($calls.results) -join '|') `
            'a slice did not get its own results directory'
        foreach ($call in $calls) { Assert-Equal '' ([string] $call.segments) "slice '$($call.shard)' was told it is a packed job" }
        Assert-True ([string] $calls[0].filter).StartsWith('(FullyQualifiedName~Ns.Alpha)&') 'a slice did not run with its own filter'
        Assert-True ([string] $calls[1].filter).StartsWith('(FullyQualifiedName~Ns.FAILS)&') 'a slice did not run with its own filter'
        Assert-Equal '|3|' (@($calls.window) -join '|') "a slice's environment window reached another slice, or did not reach its own"
        Assert-Equal '5min|20min|5min' (@($calls.hang) -join '|') "a slice's hang timeout reached another slice, or did not reach its own"
        Assert-Equal '0||' (@($calls.gcServer) -join '|') "a heavy slice's test host settings reached the slices after it"
        Assert-True (([string] $calls[0].runnerConfig | ConvertFrom-Json).parallelizeTestCollections -eq $false) 'a heavy slice did not serialize its collections'
        Assert-Equal $runnerConfigText ([string] $calls[1].runnerConfig) "a heavy slice's xunit.runner.json was left for the slice after it"
        Assert-Equal 'success|failure|success' (@(0..2 | ForEach-Object { (Get-SlotMetadata $_).testStepOutcome }) -join '|') 'a slice did not record its own outcome'
        $second = Get-SlotMetadata 1
        Assert-Equal 'Pack - B fails/window' ([string] $second.shard) "a slice's record does not name its shard"
        Assert-Equal 'Pack_B_fails_window' ([string] $second.slug) "a slice's record does not carry its artifact slug"
        Assert-Equal 'Pack 01 of 01 (3 slices)' ([string] $second.packedIn) "a slice's record does not name its packed job"
        Assert-True ($second.PSObject.Properties['testWallSeconds'] -and [double] $second.testWallSeconds -gt 0) "a slice's record does not carry its wall time"
        Assert-Equal '0|1|0|1' (@((Get-SliceOutcome 0), (Get-SliceOutcome 1), (Get-SliceOutcome 2), (Get-SliceOutcome 5)) -join '|') `
            "the per-slice steps do not conclude as their slices did (and fail for a slice that has no record)"
        Assert-True ((Get-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Raw) -match 'Pack - B fails/window \| failure') 'the job summary does not list each slice with its outcome'

        # What the per-slice upload steps publish, read back by the regression analysis: three shards, not one.
        $artifacts = Join-Path $scratch 'artifacts'
        function Copy-SlotArtifacts([string] $To) {
            Remove-Item -LiteralPath $To -Recurse -Force -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Path $To -Force | Out-Null
            foreach ($slot in @(Get-ChildItem -LiteralPath (Join-Path $workspace 'TestResults') -Directory -Filter 'slot-*')) {
                $slug = (Get-Content -LiteralPath (Join-Path $slot.FullName 'shard-metadata.json') -Raw | ConvertFrom-Json).slug
                # upload-artifact strips the common root of the paths it is given: TestResults/slot-N.
                Copy-Item -LiteralPath $slot.FullName -Destination (Join-Path $To "coverage-abc123-$slug") -Recurse
            }
        }
        function Read-Analysis([string] $From, [string[]] $Expected) {
            $out = Join-Path $scratch ("analysis-" + [guid]::NewGuid().ToString('N'))
            $env:PACK_TEST_EXPECTED = ConvertTo-Json -InputObject @($Expected) -Compress
            try {
                $analyzer = Join-Path $repositoryRoot '.github/scripts/test-regression-analysis.ps1'
                $command = "& '$analyzer' -CurrentResultsPath '$From' -OutputDirectory '$out' -CurrentSha abc123 -RepositoryPath '$repositoryRoot' -ExpectedShardNames @(`$env:PACK_TEST_EXPECTED | ConvertFrom-Json)"
                & $pwsh -NoProfile -Command $command 2>&1 | Out-Null
            }
            finally { $env:PACK_TEST_EXPECTED = $null }
            $ledgerFile = Join-Path $out 'ledger.json'
            Assert-True (Test-Path -LiteralPath $ledgerFile) 'the regression analysis wrote no ledger for a packed run'
            return [pscustomobject]@{ Ledger = (Get-Content -LiteralPath $ledgerFile -Raw | ConvertFrom-Json); Path = $ledgerFile }
        }
        Copy-SlotArtifacts $artifacts
        $analysis = Read-Analysis $artifacts @($segments.name)
        $rows = @($analysis.Ledger.shards | Sort-Object key)
        Assert-Equal 'Pack_A_heavy|Pack_B_fails_window|Pack_C_after' (@($rows.key) -join '|') 'the regression analysis does not read each packed slice as its own shard'
        Assert-Equal 'Passed|Failed|Passed' (@($rows.status) -join '|') 'a packed slice did not get its own status in the ledger'
        Assert-Equal 'Pack - B fails/window' ([string] $rows[1].name) 'a packed slice lost its shard name in the ledger'
        Assert-Equal 'Pack_A_heavy|Pack_B_fails_window|Pack_C_after' (@($analysis.Ledger.tests | Sort-Object shard | ForEach-Object { $_.shard }) -join '|') `
            "a packed slice's tests were attributed to another shard"
        Assert-True ($rows[0].PSObject.Properties['testWallSeconds'] -and [double] $rows[0].testWallSeconds -gt 0) "the ledger does not keep a packed slice's wall time"

        # ...and the timing table reads that ledger: a packed run keeps the table fresh by itself.
        $sources = Join-Path $scratch 'sources.json'
        ConvertTo-Json -InputObject @([ordered]@{ ledger = $analysis.Path; source = 'packed run' }) | Set-Content -LiteralPath $sources -Encoding utf8
        $fromPacked = Join-Path $scratch 'table-from-packed.json'
        & $pwsh -NoProfile -File (Join-Path $PSScriptRoot 'New-ShardTimingTable.ps1') -SourcesFile $sources -OutFile $fromPacked -SummaryFile (Join-Path $scratch 'table-summary.md') | Out-Null
        Assert-Equal 0 $LASTEXITCODE 'New-ShardTimingTable.ps1 failed on the ledger of a packed run'
        $tableFromPacked = Read-ShardTimingTable -Path $fromPacked
        Assert-Equal 'Pack_A_heavy,Pack_B_fails_window,Pack_C_after' (@($tableFromPacked['shards'].Keys | Sort-Object) -join ',') 'the timing table does not hold the slices of a packed run'
        Assert-Equal 1.5 ([double] $tableFromPacked['shards']['Pack_C_after']['classes']['Ns.Gamma']) 'the timing table did not read a packed slice from its TRX durations'

        # The targeted retry of the failed slice: only that slice, with its own environment, into its own directory.
        $baselineLedger = Join-Path $workspace 'retry-baseline-ledger/artifact'
        New-Item -ItemType Directory -Path $baselineLedger -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $baselineLedger 'ledger.json') -Value '{"schemaVersion":1,"shards":[],"tests":[]}' -Encoding utf8
        Remove-Item -LiteralPath $stubLog -Force
        Push-Location $workspace
        try { $retryLog = & $pwsh -NoProfile -File (Join-Path $workspace '.github/scripts/Invoke-PackedShardRetry.ps1') 2>&1 | Out-String; $retryExit = $LASTEXITCODE }
        finally { Pop-Location }
        Assert-Equal 0 $retryExit "the packed retry failed although the rerun passed: $retryLog"
        $calls = @(Get-StubCalls)
        Assert-Equal 1 $calls.Count 'the packed retry did not rerun exactly the one failed slice'
        Assert-Equal 'TestResults/slot-1/Pack_B_fails_window/retry' ([string] $calls[0].results) "the retry did not write where an unpacked retry writes, under the slice's own directory"
        Assert-Equal 'rerun-new-failures.trx' ([string] $calls[0].trx) 'the retry TRX is not named as the unpacked retry names it'
        Assert-Equal 'FullyQualifiedName~Ns.FAILS.Works' ([string] $calls[0].filter) "the retry did not run the slice's PR-new failure"
        Assert-Equal '3' ([string] $calls[0].window) "the retry ran without the slice's environment window"
        $second = Get-SlotMetadata 1
        Assert-Equal 'failure|1|success' (@($second.testStepOutcome, $second.rerunCandidateCount, $second.rerunStepOutcome) -join '|') "the retry was not recorded in the slice's own record"
        Assert-True (Test-Path -LiteralPath (Join-Path $workspace 'TestResults/slot-1/rerun-candidates.json')) "the retry candidates were not written beside the slice's results"
        Assert-Equal '|' (@((Get-SlotMetadata 0).rerunStepOutcome, (Get-SlotMetadata 2).rerunStepOutcome) -join '|') 'a slice that passed was retried'
        Copy-SlotArtifacts $artifacts
        $analysis = Read-Analysis $artifacts @($segments.name)
        Assert-Equal 3 @($analysis.Ledger.shards).Count 'the regression analysis lost a packed slice after its retry'

        # Every slice passes: the job passes.
        Reset-Job
        $env:SHARD_SEGMENTS = ConvertTo-Json -InputObject @($segments[0], $segments[2]) -Depth 6
        $run = Invoke-Job
        Assert-Equal 0 $run.ExitCode "a packed job whose slices all passed failed: $($run.Log)"
        Assert-Equal 2 @(Get-StubCalls).Count 'a packed job did not run each of its slices once'

        # Out of time: no new slice starts inside the last 12 minutes of the job, and what did not start is
        # recorded as cancelled - which fails its step and is read as an incomplete shard.
        Reset-Job
        $env:SHARD_SEGMENTS = ConvertTo-Json -InputObject @($segments[0], $segments[2]) -Depth 6
        $env:SHARD_PACK_TIMEOUT = '12'
        $run = Invoke-Job
        Assert-Equal 1 $run.ExitCode 'a packed job that ran out of time before its last slice passed'
        Assert-Equal 1 @(Get-StubCalls).Count 'a slice was started inside the time reserved for the upload steps'
        Assert-Equal 'success|cancelled' (@((Get-SlotMetadata 0).testStepOutcome, (Get-SlotMetadata 1).testStepOutcome) -join '|') 'a slice that never started is not recorded as cancelled'
        Assert-Equal 1 (Get-SliceOutcome 1) 'the step of a slice that never started passed'
        $env:SHARD_PACK_TIMEOUT = '60'

        # The job is killed while its second slice is in the test host (an out-of-memory runner, a cancellation).
        # The slice that finished keeps its results and its record; the others are recorded as cancelled.
        Reset-Job
        $killSegments = @((New-Segment 'Kill - A done' 'Ns.Alpha'), (New-Segment 'Kill - B dies' 'Ns.KILLS'), (New-Segment 'Kill - C never' 'Ns.Gamma'))
        $env:SHARD_SEGMENTS = ConvertTo-Json -InputObject $killSegments -Depth 6
        $pidFile = Join-Path $scratch 'pack.pid'
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
        $env:STUB_PACK_PID_FILE = $pidFile
        $process = Start-Process -FilePath $pwsh -ArgumentList @('-NoProfile', '-File', $invoke) -WorkingDirectory $workspace -PassThru -NoNewWindow `
            -RedirectStandardOutput (Join-Path $scratch 'killed.out') -RedirectStandardError (Join-Path $scratch 'killed.err')
        Set-Content -LiteralPath $pidFile -Value $process.Id
        Assert-True ($process.WaitForExit(120000)) 'the packed job was not killed by the stub'
        $until = [DateTime]::UtcNow.AddSeconds(30)
        while ([DateTime]::UtcNow -lt $until -and -not (@(Get-StubCalls) | Where-Object { $_.PSObject.Properties['killed'] })) { Start-Sleep -Milliseconds 200 }
        Start-Sleep -Seconds 2   # the orphaned slice process finishes on its own
        Assert-Equal 'success|cancelled|cancelled' (@(0..2 | ForEach-Object { (Get-SlotMetadata $_).testStepOutcome }) -join '|') `
            'a killed packed job did not keep the finished slice and record the rest as cancelled'
        Assert-True (Test-Path -LiteralPath (Join-Path $workspace 'TestResults/slot-0/Kill_A_done/test-results.trx')) 'a killed packed job lost the results of the slice that had finished'
        Assert-True (-not (@(Get-StubCalls) | Where-Object { $_.PSObject.Properties['shard'] -and $_.shard -ceq 'Kill - C never' })) 'a slice ran after its packed job was killed'
        Copy-SlotArtifacts $artifacts
        $analysis = Read-Analysis $artifacts @($killSegments.name)
        $rows = @($analysis.Ledger.shards | Sort-Object key)
        Assert-Equal 'Kill_A_done:Passed|Kill_B_dies:Incomplete|Kill_C_never:Incomplete' (@($rows | ForEach-Object { "$($_.key):$($_.status)" }) -join '|') `
            'the regression analysis does not read the slices of a killed packed job as incomplete'
    }
    finally {
        foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
    }

    Write-Host "Shard packing self-test passed ($script:Checks checks)."
    exit 0
}
finally {
    Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
