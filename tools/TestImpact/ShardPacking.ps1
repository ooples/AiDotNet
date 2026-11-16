# Layer 2 of test-impact selection: repack narrowed shard slices into time-balanced jobs.
# Dot-source; Test-ShardPacking.ps1 proves every rule here.
#
# THE PROBLEM. The pipeline runs one job per selected shard. After Select-AffectedTests.ps1 narrows each
#   selected shard to the test classes a change reaches, a hub-file change still selects most shards but only
#   a few classes in each. Every job pays the same fixed cost before its first test (checkout, NuGet cache,
#   the 1.4 GB build artifact, restore): a median of 4.1 minutes on a hosted runner and 5.4 on a self-hosted
#   one (run 43612258685, 166 jobs), against a median test step of 0.9 minutes. In that run 753 of 1,651
#   runner-minutes were fixed cost. The job count should follow the selected test time instead.
#
# WHAT IS PACKED. The unit is the shard slice exactly as the matrix would have run it: one matrix entry, with
#   its own filter, environment, heavy flag and hang timeout. A packed job runs each of its slices in turn as
#   its own `dotnet test` (Invoke-Shard.ps1), in its own process and its own TestResults/slot-N directory.
#   Filters are never merged, so nothing here can change which tests a slice selects:
#     * a test host that runs out of memory takes one slice with it, as it takes one job today;
#     * the Conformance and Sweep shards split one test class by an environment window - several windows can
#       share a job, each in its own process with its own environment, and no window is ever split;
#     * method-level splits (Unit - 03b4) and vstest's filter length limit are untouched;
#     * every artifact keeps its per-shard name and layout (coverage-<sha>-<slug>, test-results-<sha>-<slug>,
#       shard-metadata.json), so the regression ledger, the aggregate analysis, the delta import and Sonar read
#       a packed run exactly as they read an unpacked one.
#
# THE RULES (Group-ShardSlices). First-fit decreasing by predicted minutes into jobs of ShardPackTargetMinutes.
#   A job holds slices of one project and framework, one runner class (self-hosted routing is decided before
#   packing and kept) and one weight (heavy slices only with heavy ones). A slice stays alone when it has its
#   own `timeout`, measures timing, is predicted to fill a job, has no usable timing (see below) or is too
#   large to pass to a packed job. A job holds at most ShardPackMaxSegments slices, the number of per-slice
#   upload steps the workflow has.
#
# THE PREDICTION (Get-ShardSlicePrediction). tests/AiDotNet.Tests/xunit.runner.json runs collections one at a
#   time, so a slice's wall time is close to the sum of its tests' TRX durations. The timing table
#   (New-ShardTimingTable, built by .github/workflows/shard-timing-table.yml from recent test-outcome ledgers)
#   holds seconds per test class per shard and a wall factor per shard for the time TRX does not count.
#   A slice the table does not know is predicted to fill a job, so it runs alone exactly as it does today.
#
# THE FLAG. Repository variable TEST_IMPACT_REPACK: unset or 'off' computes nothing and leaves the matrix as it
#   is; 'report' computes the packing and publishes it, with predicted against actual time, without changing
#   the matrix; 'on' packs. Only pull_request and merge_group runs are in scope: coverage-everywhere, scheduled
#   and push runs feed the map, the nightly net and the ledgers other tools read by job name. Any failure while
#   packing leaves the matrix as it was.

$script:ShardPackSchemaVersion = 1
# Predicted test minutes one packed job holds.
$script:ShardPackTargetMinutes = 20.0
# Slices per packed job. sonarcloud.yml has one "Packed slice N" marker and two upload steps per slot;
# Test-ShardPacking.ps1 fails when this and the workflow disagree.
$script:ShardPackMaxSegments = 8
# Characters of slice JSON one packed job carries. The job passes its slices to Invoke-Shard.ps1 in one
# environment string, and Linux limits a single one to 128 KiB; GitHub's toJSON also indents what it writes.
$script:ShardPackMaxSegmentCharacters = 60000
# Seconds `dotnet test` spends outside the tests it reports (host start, discovery of the whole assembly).
# Measured: test step minus TRX sum, median 0.26 min for the 100 slices under a minute in run 43612258685.
$script:ShardPackInvocationSeconds = 24.0
# Wall time over TRX time for a shard with no wall observation. Least squares over the 165 shards of run
# 43612258685: step = 0.46 min + 1.34 x TRX.
$script:ShardPackDefaultWallFactor = 1.35
$script:ShardPackWallFactorRange = @(1.0, 4.0)
# A test class the table has never timed, in a shard it has.
$script:ShardPackUnknownClassSeconds = 60.0
# Ledgers a table entry is the median of.
$script:ShardPackMaxObservations = 5
# Job timeout of a packed job: room for the fixed steps plus three times the predicted test time, never less
# than the 60 minutes each of its slices would have had alone, never more than 120.
$script:ShardPackDefaultTimeoutMinutes = 60
$script:ShardPackFixedMinutes = 15
$script:ShardPackTimeoutFactor = 3.0
$script:ShardPackMaxTimeoutMinutes = 120
$script:ShardPackEvents = @('pull_request', 'merge_group')
# Keys ConvertTo-PackedShardMatrix adds to a slice; Expand-PackedShardMatrix removes them again.
$script:ShardPackAddedSliceKeys = @('slug', 'predictedMinutes')
$script:ShardPackJobNamePattern = '^Pack \d+ of \d+ \(\d+ slices\)$'
$script:ShardPackSliceStepPattern = '^Packed slice \d+: (?<shard>.+)$'

function Get-ShardRepackMode {
    <# off | report | on. Anything else, including unset, is off: an unreadable flag must change nothing. #>
    param([AllowNull()] [AllowEmptyString()] [string] $Value)
    $text = ([string] $Value).Trim().ToLowerInvariant()
    if ($text -ceq 'report') { return 'report' }
    if ($text -ceq 'on') { return 'on' }
    return 'off'
}

function Get-ShardRepackScope {
    <# Whether a run of this event may be repacked, and why not when it may not. #>
    param([AllowNull()] [AllowEmptyString()] [string] $EventName, [AllowNull()] [AllowEmptyString()] [string] $CoverageEverywhere)
    if ([string] $CoverageEverywhere -ceq 'true') {
        return [pscustomobject]@{ InScope = $false; Why = 'a coverage-everywhere run keeps one job per shard for the map' }
    }
    if ([string] $EventName -cnotin $script:ShardPackEvents) {
        return [pscustomobject]@{ InScope = $false; Why = "a '$EventName' run keeps one job per shard (only $($script:ShardPackEvents -join ' and ') runs are repacked)" }
    }
    return [pscustomobject]@{ InScope = $true; Why = '' }
}

function ConvertTo-ShardSlug {
    <# Identical to the test job's artifact naming and the regression analyzer's shard key. #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Name)
    return $Name -replace '[\\/:*?"<>|\s-]+', '_'
}

function Get-ShardSliceProperty {
    param([Parameter(Mandatory)] [object] $Slice, [Parameter(Mandatory)] [string] $Name)
    if ($Slice -is [Collections.IDictionary]) { if ($Slice.Contains($Name)) { return $Slice[$Name] } else { return $null } }
    $property = $Slice.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Get-TestClassName {
    <# 'Ns.Class.Method' or 'Ns.Class.Method(x: 1.5)' -> 'Ns.Class'. #>
    param([AllowNull()] [AllowEmptyString()] [string] $FullyQualifiedName)
    $text = [string] $FullyQualifiedName
    $paren = $text.IndexOf('(')
    if ($paren -ge 0) { $text = $text.Substring(0, $paren) }
    $dot = $text.LastIndexOf('.')
    if ($dot -le 0) { return $text }
    return $text.Substring(0, $dot)
}

# ------------------------------------------------------------------ the timing table

function Read-ShardTimingObservation {
    <#
    .SYNOPSIS
        One run's timings, from its test-outcome ledger (ledger.json of test-regression-analysis.ps1).
    .DESCRIPTION
        Seconds per test class per shard, from the TRX durations the ledger keeps for every test. A shard that
        did not finish (Incomplete) is left out: its last class stopped part-way. A test the targeted retry ran
        twice counts once, at its longer duration.

        -WallSeconds gives the measured duration of each shard's test step, by slug, where the caller knows it
        (the jobs API for a one-job-per-shard run). A ledger row's own testWallSeconds, which a packed job
        records per slice, takes precedence.

        -Instrumented marks a run whose light shards collected coverage (every non-pull-request run). Coverlet
        slows them, so the table uses such an observation only for what no pull request run has timed.
    #>
    param(
        [Parameter(Mandatory)] [string] $LedgerPath,
        [hashtable] $WallSeconds = @{},
        [bool] $Instrumented = $false,
        [string] $Source = ''
    )
    $shards = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
    $stream = [IO.File]::OpenRead($LedgerPath)
    try {
        $document = [Text.Json.JsonDocument]::Parse($stream)
        try {
            $root = $document.RootElement
            $probe = [Text.Json.JsonElement]::new()
            if (-not $root.TryGetProperty('schemaVersion', [ref] $probe) -or $probe.GetInt32() -ne 1) {
                throw "ledger '$LedgerPath' is not schema version 1"
            }
            $sha = if ($root.TryGetProperty('sha', [ref] $probe)) { [string] $probe.GetString() } else { '' }
            $generated = if ($root.TryGetProperty('generatedUtc', [ref] $probe)) { [string] $probe.GetString() } else { '' }
            foreach ($row in $root.GetProperty('shards').EnumerateArray()) {
                $key = [string] $row.GetProperty('key').GetString()
                $status = if ($row.TryGetProperty('status', [ref] $probe)) { [string] $probe.GetString() } else { '' }
                if ($status -cnotin @('Passed', 'Failed')) { continue }
                $wall = $null
                if ($row.TryGetProperty('testWallSeconds', [ref] $probe) -and $probe.ValueKind -eq [Text.Json.JsonValueKind]::Number) {
                    $wall = $probe.GetDouble()
                }
                elseif ($WallSeconds.ContainsKey($key)) { $wall = [double] $WallSeconds[$key] }
                $shards[$key] = [pscustomobject]@{
                    wallSeconds = $wall
                    classes     = [Collections.Generic.Dictionary[string, double]]::new([StringComparer]::Ordinal)
                    tests       = [Collections.Generic.Dictionary[string, double]]::new([StringComparer]::Ordinal)
                }
            }
            foreach ($test in $root.GetProperty('tests').EnumerateArray()) {
                $key = [string] $test.GetProperty('shard').GetString()
                if (-not $shards.ContainsKey($key)) { continue }
                $outcome = [string] $test.GetProperty('outcome').GetString()
                if ($outcome -cnotin @('Passed', 'Failed')) { continue }
                $duration = [TimeSpan]::Zero
                if (-not [TimeSpan]::TryParse([string] $test.GetProperty('duration').GetString(),
                        [Globalization.CultureInfo]::InvariantCulture, [ref] $duration)) { continue }
                $seconds = $duration.TotalSeconds
                $shard = $shards[$key]
                $identity = [string] $test.GetProperty('identity').GetString()
                $previous = 0.0
                if ($shard.tests.TryGetValue($identity, [ref] $previous)) {
                    if ($seconds -le $previous) { continue }
                    $shard.tests[$identity] = $seconds
                    $seconds -= $previous
                }
                else { $shard.tests[$identity] = $seconds }
                $class = Get-TestClassName ([string] $test.GetProperty('fullyQualifiedName').GetString())
                $sum = 0.0
                [void] $shard.classes.TryGetValue($class, [ref] $sum)
                $shard.classes[$class] = $sum + $seconds
            }
        }
        finally { $document.Dispose() }
    }
    finally { $stream.Dispose() }

    $result = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
    foreach ($key in $shards.Keys) {
        $shard = $shards[$key]
        if ($shard.classes.Count -eq 0) { continue }
        $trx = 0.0
        foreach ($value in $shard.classes.Values) { $trx += $value }
        $result[$key] = [pscustomobject]@{ trxSeconds = $trx; wallSeconds = $shard.wallSeconds; classes = $shard.classes }
    }
    return [pscustomobject]@{ source = $Source; sha = $sha; generatedUtc = $generated; instrumented = $Instrumented; shards = $result }
}

function Get-ShardTimingMedian {
    param([Parameter(Mandatory)] [double[]] $Values)
    $sorted = @($Values | Sort-Object)
    $middle = [int] [Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) { return [double] $sorted[$middle] }
    return ([double] $sorted[$middle - 1] + [double] $sorted[$middle]) / 2
}

function New-ShardTimingTable {
    <#
    .SYNOPSIS
        Pure. The timing table from observations, newest first.
    .DESCRIPTION
        For each test class of each shard: the median of its newest ShardPackMaxObservations timings from
        uninstrumented runs, or from instrumented ones when no uninstrumented run timed it. A shard's seconds
        are the sum over every class any used observation saw in it, so a whole shard is predicted from
        narrowed runs too. A shard's wall factor is the median, over the same observations, of the test step's
        wall time less the invocation cost, divided by its TRX time.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Observations,
        [int] $MaxObservations = $script:ShardPackMaxObservations
    )
    $classValues = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
    $factorValues = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
    foreach ($observation in $Observations) {
        $rank = if ($observation.instrumented) { 'instrumented' } else { 'plain' }
        foreach ($key in $observation.shards.Keys) {
            $shard = $observation.shards[$key]
            if (-not $classValues.ContainsKey($key)) {
                $classValues[$key] = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
                $factorValues[$key] = @{ plain = [Collections.Generic.List[double]]::new(); instrumented = [Collections.Generic.List[double]]::new() }
            }
            foreach ($class in $shard.classes.Keys) {
                if (-not $classValues[$key].ContainsKey($class)) {
                    $classValues[$key][$class] = @{ plain = [Collections.Generic.List[double]]::new(); instrumented = [Collections.Generic.List[double]]::new() }
                }
                $values = $classValues[$key][$class][$rank]
                if ($values.Count -lt $MaxObservations) { $values.Add([double] $shard.classes[$class]) }
            }
            # A factor from a few seconds of tests is noise: the invocation cost dominates it.
            if ($null -ne $shard.wallSeconds -and $shard.trxSeconds -ge 30) {
                $factor = ([double] $shard.wallSeconds - $script:ShardPackInvocationSeconds) / $shard.trxSeconds
                $factor = [Math]::Min($script:ShardPackWallFactorRange[1], [Math]::Max($script:ShardPackWallFactorRange[0], $factor))
                if ($factorValues[$key][$rank].Count -lt $MaxObservations) { $factorValues[$key][$rank].Add($factor) }
            }
        }
    }

    $shards = [ordered]@{}
    foreach ($key in @($classValues.Keys | Sort-Object -CaseSensitive)) {
        $classes = [ordered]@{}
        $total = 0.0
        foreach ($class in @($classValues[$key].Keys | Sort-Object -CaseSensitive)) {
            $values = $classValues[$key][$class]
            # Assigned, not returned from an `if`: a one-element list would come back as its element.
            $used = $values.plain
            if ($used.Count -eq 0) { $used = $values.instrumented }
            $seconds = [Math]::Round((Get-ShardTimingMedian $used.ToArray()), 2)
            $classes[$class] = $seconds
            $total += $seconds
        }
        $factors = $factorValues[$key]
        $usedFactors = $factors.plain
        if ($usedFactors.Count -eq 0) { $usedFactors = $factors.instrumented }
        $entry = [ordered]@{ seconds = [Math]::Round($total, 2); classes = $classes }
        if ($usedFactors.Count -gt 0) { $entry['wallFactor'] = [Math]::Round((Get-ShardTimingMedian $usedFactors.ToArray()), 3) }
        $shards[$key] = $entry
    }
    return [ordered]@{
        schemaVersion     = $script:ShardPackSchemaVersion
        generatedUtc      = [DateTime]::UtcNow.ToString('o')
        invocationSeconds = $script:ShardPackInvocationSeconds
        defaultWallFactor = $script:ShardPackDefaultWallFactor
        sources           = @($Observations | ForEach-Object {
                [ordered]@{ source = [string] $_.source; sha = [string] $_.sha; generatedUtc = [string] $_.generatedUtc
                    instrumented = [bool] $_.instrumented; shards = $_.shards.Count }
            })
        shards            = $shards
    }
}

function Read-ShardTimingTable {
    <# The table at -Path, or $null (with the reason on the host) when there is none this code can read. #>
    param([AllowNull()] [AllowEmptyString()] [string] $Path)
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Write-Host "shard timing table: none at '$Path'"
        return $null
    }
    try {
        $table = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
        if ($table -isnot [Collections.IDictionary] -or [int] $table['schemaVersion'] -ne $script:ShardPackSchemaVersion -or
            $table['shards'] -isnot [Collections.IDictionary]) {
            throw 'it is not a schema version 1 timing table'
        }
        return $table
    }
    catch {
        Write-Host "::warning::shard timing table '$Path' is unusable ($($_.Exception.Message)); every slice is predicted to fill a job"
        return $null
    }
}

function Get-ShardSlicePrediction {
    <#
    .SYNOPSIS
        Pure. Predicted minutes of one slice's test step.
    .DESCRIPTION
        -Classes is the slice's test classes when it was narrowed (the TypeImpact plan's testClasses), and
        $null for a whole shard. Source says what the number rests on:
          timed    every class of the slice is in the table;
          partial  some classes are not, and count as the shard's mean class time, at least
                   ShardPackUnknownClassSeconds each;
          default  the table does not know the shard (or there is no table): the slice is predicted to fill
                   a job, so it is never packed with anything.
    #>
    param(
        [Parameter(Mandatory)] [object] $Slice,
        [AllowNull()] [object] $Table,
        [AllowNull()] [string[]] $Classes,
        [double] $TargetMinutes = $script:ShardPackTargetMinutes
    )
    $slug = ConvertTo-ShardSlug ([string] (Get-ShardSliceProperty $Slice 'name'))
    $entry = $null
    if ($null -ne $Table -and $Table['shards'].Contains($slug)) { $entry = $Table['shards'][$slug] }
    if ($null -eq $entry -or $entry['classes'] -isnot [Collections.IDictionary] -or $entry['classes'].Count -eq 0) {
        return [pscustomobject]@{ Minutes = [double] $TargetMinutes; Source = 'default'; Known = 0; Unknown = 0 }
    }
    $invocation = if ($Table.Contains('invocationSeconds')) { [double] $Table['invocationSeconds'] } else { $script:ShardPackInvocationSeconds }
    $factor = if ($entry.Contains('wallFactor')) { [double] $entry['wallFactor'] }
        elseif ($Table.Contains('defaultWallFactor')) { [double] $Table['defaultWallFactor'] }
        else { $script:ShardPackDefaultWallFactor }
    $known = 0; $unknown = 0
    if ($null -eq $Classes) {
        $seconds = [double] $entry['seconds']
        $known = $entry['classes'].Count
    }
    else {
        $seconds = 0.0
        $mean = [double] $entry['seconds'] / $entry['classes'].Count
        foreach ($class in $Classes) {
            if ($entry['classes'].Contains($class)) { $seconds += [double] $entry['classes'][$class]; $known++ }
            else { $seconds += [Math]::Max($script:ShardPackUnknownClassSeconds, $mean); $unknown++ }
        }
    }
    return [pscustomobject]@{
        Minutes = [Math]::Round(($invocation + $factor * $seconds) / 60, 2)
        Source  = $(if ($unknown -gt 0) { 'partial' } else { 'timed' })
        Known   = $known
        Unknown = $unknown
    }
}

# ------------------------------------------------------------------ packing

function Get-ShardPackTimeout {
    <# Job timeout, in minutes, of a packed job predicted to test for -Minutes. #>
    param([Parameter(Mandatory)] [double] $Minutes)
    $wanted = [Math]::Ceiling($script:ShardPackFixedMinutes + $script:ShardPackTimeoutFactor * $Minutes)
    return [int] [Math]::Min($script:ShardPackMaxTimeoutMinutes, [Math]::Max($script:ShardPackDefaultTimeoutMinutes, $wanted))
}

function Group-ShardSlices {
    <#
    .SYNOPSIS
        Pure. Which slices share a job: first-fit decreasing by predicted minutes.
    .DESCRIPTION
        Returns one group per job, in the order of each group's first slice in -Slices. Every input slice is in
        exactly one group. The rules are in the header of this file; -HeavyNames is the heavy list of
        Invoke-Shard.ps1 (a slice is also heavy when its own `heavy` is true) and -AloneNames the shards that
        measure timing.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Slices,
        [Parameter(Mandatory)] [hashtable] $Minutes,
        [string[]] $HeavyNames = @(),
        [string[]] $AloneNames = @(),
        [double] $TargetMinutes = $script:ShardPackTargetMinutes,
        [int] $MaxSegments = $script:ShardPackMaxSegments,
        [int] $MaxSegmentCharacters = $script:ShardPackMaxSegmentCharacters
    )
    if ($TargetMinutes -le 0 -or $MaxSegments -lt 1) { throw 'a packed job needs a positive size and at least one slice.' }
    $heavy = [Collections.Generic.HashSet[string]]::new([string[]] @($HeavyNames), [StringComparer]::Ordinal)
    $alone = [Collections.Generic.HashSet[string]]::new([string[]] @($AloneNames), [StringComparer]::Ordinal)
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $items = [Collections.Generic.List[object]]::new()
    for ($index = 0; $index -lt $Slices.Count; $index++) {
        $slice = $Slices[$index]
        $name = [string] (Get-ShardSliceProperty $slice 'name')
        if ([string]::IsNullOrWhiteSpace($name)) { throw 'a slice has no name.' }
        if (-not $seen.Add($name)) { throw "slice '$name' is listed twice." }
        if (-not $Minutes.ContainsKey($name)) { throw "slice '$name' has no predicted time." }
        $predicted = [double] $Minutes[$name]
        if ([double]::IsNaN($predicted) -or $predicted -lt 0) { throw "slice '$name' has an unusable predicted time." }
        $isHeavy = $heavy.Contains($name) -or ((Get-ShardSliceProperty $slice 'heavy') -eq $true)
        $characters = (ConvertTo-Json -InputObject $slice -Depth 8 -Compress).Length
        $why = ''
        if (Get-ShardSliceProperty $slice 'timeout') { $why = 'it has its own job timeout' }
        elseif ($alone.Contains($name)) { $why = 'it measures timing' }
        elseif ($null -ne (Get-ShardSliceProperty $slice 'workload')) { $why = 'it is not an ordinary test workload' }
        elseif ($null -ne (Get-ShardSliceProperty $slice 'segments')) { $why = 'it is already a packed job' }
        elseif ($characters -gt $MaxSegmentCharacters) { $why = 'its definition is too large to pass to a packed job' }
        elseif ($predicted -ge $TargetMinutes) { $why = 'it is predicted to fill a job' }
        $items.Add([pscustomobject]@{
                Index = $index; Name = $name; Slice = $slice; Minutes = $predicted; Characters = $characters; Alone = $why
                Heavy = $isHeavy
                Key   = (@([string] (Get-ShardSliceProperty $slice 'project'), [string] (Get-ShardSliceProperty $slice 'framework'),
                        [string] (Get-ShardSliceProperty $slice 'runner'), $(if ($isHeavy) { 'heavy' } else { 'ordinary' })) -join '|')
            })
    }

    $groups = [Collections.Generic.List[object]]::new()
    $open = @{}
    $ordered = @($items | Sort-Object -Stable -Property @{ Expression = { $_.Minutes }; Descending = $true }, @{ Expression = { $_.Index } })
    foreach ($item in $ordered) {
        $target = $null
        if (-not $item.Alone) {
            if (-not $open.ContainsKey($item.Key)) { $open[$item.Key] = [Collections.Generic.List[object]]::new() }
            foreach ($candidate in $open[$item.Key]) {
                if ($candidate.Items.Count -lt $MaxSegments -and
                    ($candidate.Minutes + $item.Minutes) -le $TargetMinutes -and
                    ($candidate.Characters + $item.Characters) -le $MaxSegmentCharacters) { $target = $candidate; break }
            }
        }
        if ($null -eq $target) {
            $target = [pscustomobject]@{ Key = $item.Key; Heavy = $item.Heavy; Minutes = 0.0; Characters = 0
                Items = [Collections.Generic.List[object]]::new(); Alone = $item.Alone }
            $groups.Add($target)
            if (-not $item.Alone) { $open[$item.Key].Add($target) }
        }
        $target.Items.Add($item)
        $target.Minutes += $item.Minutes
        $target.Characters += $item.Characters
    }

    return @($groups | ForEach-Object {
            $members = @($_.Items | Sort-Object -Property Index)
            [pscustomobject]@{
                First   = $members[0].Index
                Heavy   = $_.Heavy
                Minutes = [Math]::Round($_.Minutes, 2)
                Alone   = $(if ($members.Count -eq 1) { $_.Alone } else { '' })
                Slices  = @($members | ForEach-Object { $_.Slice })
            }
        } | Sort-Object -Property First)
}

function ConvertTo-PackedShardMatrix {
    <#
    .SYNOPSIS
        Pure. The matrix for Group-ShardSlices' groups.
    .DESCRIPTION
        A group of one is its slice, unchanged, so a job that is not packed is the job it always was. A larger
        group becomes one entry the test job runs as a packed job:
          name       'Pack 02 of 07 (5 slices)' - the job is 'Tests (<framework>) - Pack 02 of 07 (5 slices)';
          segments   the slices, each as it was plus its slug and predicted minutes;
          slotN      { name, slug } of segment N, for the workflow's per-slice steps (a property that is absent
                     reads as empty in a workflow expression, which an index past the end is not promised to);
          timeout    Get-ShardPackTimeout; runner and heavy as the slices have them.
    #>
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Groups, [Parameter(Mandatory)] [hashtable] $Minutes)
    $packs = @($Groups | Where-Object { @($_.Slices).Count -gt 1 })
    $slugs = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $number = 0
    $matrix = foreach ($group in $Groups) {
        $slices = @($group.Slices)
        if ($slices.Count -eq 1) { $slices[0]; continue }
        $number++
        $first = $slices[0]
        $entry = [ordered]@{
            name      = 'Pack {0:00} of {1:00} ({2} slices)' -f $number, $packs.Count, $slices.Count
            framework = [string] (Get-ShardSliceProperty $first 'framework')
            project   = [string] (Get-ShardSliceProperty $first 'project')
        }
        $runner = [string] (Get-ShardSliceProperty $first 'runner')
        if ($runner) { $entry['runner'] = $runner }
        if ($group.Heavy) { $entry['heavy'] = $true }
        $entry['timeout'] = Get-ShardPackTimeout -Minutes ([double] $group.Minutes)
        $entry['predictedMinutes'] = [double] $group.Minutes
        $segments = [Collections.Generic.List[object]]::new()
        for ($slot = 0; $slot -lt $slices.Count; $slot++) {
            $name = [string] (Get-ShardSliceProperty $slices[$slot] 'name')
            $slug = ConvertTo-ShardSlug $name
            if (-not $slugs.Add($slug)) { throw "two packed slices share the artifact slug '$slug'." }
            $segment = $slices[$slot].PSObject.Copy()
            $segment | Add-Member -NotePropertyName slug -NotePropertyValue $slug -Force
            $segment | Add-Member -NotePropertyName predictedMinutes -NotePropertyValue ([double] $Minutes[$name]) -Force
            $segments.Add($segment)
            $entry["slot$slot"] = [ordered]@{ name = $name; slug = $slug }
        }
        $entry['segments'] = $segments.ToArray()
        [pscustomobject] $entry
    }
    return @($matrix)
}

function Expand-PackedShardMatrix {
    <# The slices of a matrix, packed or not, each as it was before packing. #>
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Matrix)
    return @(foreach ($entry in $Matrix) {
            $segments = Get-ShardSliceProperty $entry 'segments'
            if ($null -eq $segments) { $entry; continue }
            foreach ($segment in @($segments)) {
                $slice = [ordered]@{}
                foreach ($property in $segment.PSObject.Properties) {
                    if ($property.Name -cnotin $script:ShardPackAddedSliceKeys) { $slice[$property.Name] = $property.Value }
                }
                [pscustomobject] $slice
            }
        })
}

function Assert-ShardPackComplete {
    <#
        Refuses a packed matrix that does not run exactly the slices it was given: none dropped, none twice, each
        with the definition it came with. Runs on every packing; a refusal leaves the matrix unpacked.
    #>
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Slices, [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Matrix)
    $expected = [Collections.Generic.Dictionary[string, string]]::new([StringComparer]::Ordinal)
    foreach ($slice in $Slices) {
        $expected[[string] (Get-ShardSliceProperty $slice 'name')] = ConvertTo-Json -InputObject $slice -Depth 8 -Compress
    }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($entry in $Matrix) {
        $segments = Get-ShardSliceProperty $entry 'segments'
        if ($null -ne $segments) {
            if (@($segments).Count -lt 2 -or @($segments).Count -gt $script:ShardPackMaxSegments) {
                throw "packed job '$($entry.name)' holds $(@($segments).Count) slice(s)."
            }
            if ([string] $entry.name -cnotmatch $script:ShardPackJobNamePattern) { throw "packed job '$($entry.name)' is not named as a packed job." }
        }
        elseif ([string] (Get-ShardSliceProperty $entry 'name') -cmatch $script:ShardPackJobNamePattern) {
            throw "shard '$($entry.name)' is named like a packed job."
        }
    }
    foreach ($slice in @(Expand-PackedShardMatrix -Matrix $Matrix)) {
        $name = [string] (Get-ShardSliceProperty $slice 'name')
        if (-not $names.Add($name)) { throw "slice '$name' is in the packed matrix twice." }
        if (-not $expected.ContainsKey($name)) { throw "slice '$name' is in the packed matrix but was not selected." }
        if ((ConvertTo-Json -InputObject $slice -Depth 8 -Compress) -cne $expected[$name]) {
            throw "slice '$name' does not have the definition it was selected with."
        }
    }
    if ($names.Count -ne $expected.Count) {
        $lost = @($expected.Keys | Where-Object { -not $names.Contains($_) } | Sort-Object)
        throw "the packed matrix drops $($lost.Count) slice(s): $(@($lost | Select-Object -First 5) -join ', ')."
    }
}

function Get-ShardPackWeights {
    <#
        Heavy and timing shards by name, read from the script that decides them (Invoke-Shard.ps1) through the
        rolling map's reader, so there is no second list to drift.
    #>
    param([Parameter(Mandatory)] [string] $ShardScriptPath)
    if (-not (Get-Command Get-NoCoverageShardLists -ErrorAction SilentlyContinue)) { . "$PSScriptRoot/RollingShardMap.ps1" }
    # The reader also folds in a manifest's `heavy: true` entries; Group-ShardSlices reads those from the
    # slices themselves, so it gets a manifest with none.
    $lists = Get-NoCoverageShardLists -ScriptLines @(Get-Content -LiteralPath $ShardScriptPath) -Manifest @([pscustomobject]@{ name = '' })
    $timing = @($lists.Timing)
    return [pscustomobject]@{
        Heavy  = @($lists.NoCoverage | Where-Object { $_ -cnotin $timing })
        Timing = $timing
    }
}

function New-ShardPackPlan {
    <#
    .SYNOPSIS
        Predicts, groups and checks: everything a caller needs to pack a matrix or to report what packing
        would do. Throws when anything is off; the caller then leaves the matrix as it was.
    .DESCRIPTION
        -Slices is the matrix as it would run unpacked. -PlanShards maps a shard name to its TypeImpact plan
        entry (testClasses), for slices whose `narrowed` is true. Returns Matrix (the packed matrix) and Plan
        (the record published as the shard-pack-plan artifact).
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Slices,
        [AllowNull()] [object] $Table,
        [hashtable] $PlanShards = @{},
        [string[]] $HeavyNames = @(),
        [string[]] $AloneNames = @(),
        [string] $Mode = 'report',
        [string] $EventName = '',
        [double] $TargetMinutes = $script:ShardPackTargetMinutes
    )
    $minutes = @{}
    $predictions = @{}
    foreach ($slice in $Slices) {
        $name = [string] (Get-ShardSliceProperty $slice 'name')
        $classes = $null
        if ((Get-ShardSliceProperty $slice 'narrowed') -eq $true -and $PlanShards.ContainsKey($name)) {
            $listed = Get-ShardSliceProperty $PlanShards[$name] 'testClasses'
            if ($null -ne $listed) { $classes = [string[]] @($listed | ForEach-Object { [string] $_ }) }
        }
        $prediction = Get-ShardSlicePrediction -Slice $slice -Table $Table -Classes $classes -TargetMinutes $TargetMinutes
        $predictions[$name] = $prediction
        $minutes[$name] = $prediction.Minutes
    }
    $groups = @(Group-ShardSlices -Slices $Slices -Minutes $minutes -HeavyNames $HeavyNames -AloneNames $AloneNames -TargetMinutes $TargetMinutes)
    $matrix = @(ConvertTo-PackedShardMatrix -Groups $groups -Minutes $minutes)
    Assert-ShardPackComplete -Slices $Slices -Matrix $matrix

    $jobs = for ($index = 0; $index -lt $groups.Count; $index++) {
        $group = $groups[$index]
        $entry = $matrix[$index]
        $packed = $null -ne (Get-ShardSliceProperty $entry 'segments')
        [ordered]@{
            name             = [string] (Get-ShardSliceProperty $entry 'name')
            packed           = $packed
            heavy            = [bool] $group.Heavy
            runner           = [string] (Get-ShardSliceProperty $entry 'runner')
            framework        = [string] (Get-ShardSliceProperty $entry 'framework')
            predictedMinutes = [double] $group.Minutes
            timeout          = $(if ($packed) { [int] $entry.timeout } else { $null })
            alone            = [string] $group.Alone
            slices           = @($group.Slices | ForEach-Object {
                    $name = [string] (Get-ShardSliceProperty $_ 'name')
                    [ordered]@{
                        name = $name; slug = ConvertTo-ShardSlug $name
                        predictedMinutes = [double] $predictions[$name].Minutes; source = [string] $predictions[$name].Source
                        narrowed = ((Get-ShardSliceProperty $_ 'narrowed') -eq $true)
                        unknownClasses = [int] $predictions[$name].Unknown
                    }
                })
        }
    }
    $plan = [ordered]@{
        schemaVersion  = $script:ShardPackSchemaVersion
        mode           = $Mode
        event          = $EventName
        targetMinutes  = $TargetMinutes
        maxSegments    = $script:ShardPackMaxSegments
        table          = $(if ($null -eq $Table) { $null } else { [ordered]@{ generatedUtc = [string] $Table['generatedUtc']; sources = @($Table['sources']).Count } })
        slices         = $Slices.Count
        jobs           = @($jobs)
        packedJobs     = @($jobs | Where-Object { $_.packed }).Count
        defaultSlices  = @($predictions.Values | Where-Object { $_.Source -ceq 'default' }).Count
    }
    return [pscustomobject]@{ Matrix = $matrix; Plan = $plan }
}

function Get-ShardPackSummary {
    <# Markdown for the step summary: what packing does, or would do, to this matrix. #>
    param([Parameter(Mandatory)] [object] $Plan, [string] $Applied = '')
    $jobs = @($Plan.jobs)
    $lines = [Collections.Generic.List[string]]::new()
    [void] $lines.Add('### Shard repacking')
    [void] $lines.Add('')
    $verb = if ($Applied) { $Applied } elseif ($Plan.mode -ceq 'on') { 'run as' } else { 'would run as' }
    [void] $lines.Add("**$($Plan.slices)** slice(s) $verb **$($jobs.Count)** job(s), $($Plan.packedJobs) of them packed (TEST_IMPACT_REPACK=$($Plan.mode), $($Plan.targetMinutes) predicted minutes a job).")
    if ($null -eq $Plan.table) {
        [void] $lines.Add('')
        [void] $lines.Add('There is no timing table, so every slice is predicted to fill a job and nothing is packed. The Shard timing table workflow builds it.')
    }
    elseif ($Plan.defaultSlices -gt 0) {
        [void] $lines.Add('')
        [void] $lines.Add("$($Plan.defaultSlices) slice(s) have no timing yet and stay alone.")
    }
    $packed = @($jobs | Where-Object { $_.packed })
    if ($packed.Count -gt 0) {
        [void] $lines.Add('')
        [void] $lines.Add('| Job | Runner | Predicted min | Slices |')
        [void] $lines.Add('|---|---|---|---|')
        foreach ($job in $packed) {
            $runner = if ($job.runner) { $job.runner } else { 'hosted' }
            $weight = if ($job.heavy) { ', heavy' } else { '' }
            $slices = @($job.slices | ForEach-Object { "$($_.name) ($($_.predictedMinutes))" }) -join '; '
            [void] $lines.Add("| $($job.name) | $runner$weight | $($job.predictedMinutes) | $slices |")
        }
    }
    return ($lines -join "`n")
}

# ------------------------------------------------------------------ reading a packed run

function Expand-PackedShardJobs {
    <#
    .SYNOPSIS
        Pure. A run's jobs with every packed job replaced by one job per slice, so tools that read
        'Tests (<framework>) - <shard>' jobs read a packed run as they read an unpacked one.
    .DESCRIPTION
        A packed job reports each slice in a step named 'Packed slice N: <shard>' whose conclusion is that
        slice's own. Each becomes a job named for the shard, with the step's conclusion and times and the packed
        job's url; packedIn names the job it ran in. A packed job with no such step yet (queued, or cancelled
        before it started) is returned as it is, so it stays visible. Other jobs pass through untouched.
    #>
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [AllowNull()] [object[]] $Jobs)
    return @(foreach ($job in @($Jobs | Where-Object { $null -ne $_ })) {
            $name = [string] (Get-ShardSliceProperty $job 'name')
            $match = [regex]::Match($name, '^(?<prefix>Tests \([^)]*\) - )(?<pack>.+)$')
            if (-not $match.Success -or $match.Groups['pack'].Value -cnotmatch $script:ShardPackJobNamePattern) { $job; continue }
            $steps = @(@(Get-ShardSliceProperty $job 'steps') | Where-Object { $null -ne $_ -and [string] $_.name -cmatch $script:ShardPackSliceStepPattern })
            if ($steps.Count -eq 0) { $job; continue }
            foreach ($step in $steps) {
                $shard = [regex]::Match([string] $step.name, $script:ShardPackSliceStepPattern).Groups['shard'].Value
                $slice = $job.PSObject.Copy()
                $slice.name = $match.Groups['prefix'].Value + $shard
                $conclusion = [string] (Get-ShardSliceProperty $step 'conclusion')
                $slice | Add-Member -NotePropertyName conclusion -NotePropertyValue $(if ($conclusion) { $conclusion } else { [string] (Get-ShardSliceProperty $job 'conclusion') }) -Force
                foreach ($time in 'started_at', 'completed_at') {
                    $value = Get-ShardSliceProperty $step $time
                    if ($null -ne $value) { $slice | Add-Member -NotePropertyName $time -NotePropertyValue $value -Force }
                }
                $slice | Add-Member -NotePropertyName packedIn -NotePropertyValue $name -Force
                $slice
            }
        })
}

# ------------------------------------------------------------------ predicted against actual

# The test job's step that runs the tests; its duration is a slice's (or a packed job's) actual test time.
$script:ShardPackTestStepName = 'Run tests (sharded) with coverage'

function Get-ShardJobTimings {
    <#
        Pure. Per test job of a run, by the name after 'Tests (<framework>) - ': minutes of the whole job and of
        its test step, from the jobs API. Step is $null when the test step did not finish.
    #>
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [AllowNull()] [object[]] $Jobs)
    $minutes = {
        param($From, $To)
        if (-not $From -or -not $To) { return $null }
        return ([DateTimeOffset] $To - [DateTimeOffset] $From).TotalMinutes
    }
    $timings = [hashtable]::new([StringComparer]::Ordinal)
    foreach ($job in @($Jobs | Where-Object { $null -ne $_ })) {
        $match = [regex]::Match([string] (Get-ShardSliceProperty $job 'name'), '^Tests \([^)]*\) - (?<name>.+)$')
        if (-not $match.Success) { continue }
        $step = @(@(Get-ShardSliceProperty $job 'steps') | Where-Object { $null -ne $_ -and [string] $_.name -ceq $script:ShardPackTestStepName }) | Select-Object -First 1
        $stepMinutes = $null
        if ($null -ne $step -and [string] $step.conclusion -cin @('success', 'failure')) { $stepMinutes = & $minutes $step.started_at $step.completed_at }
        $timings[$match.Groups['name'].Value] = [pscustomobject]@{
            Job        = & $minutes (Get-ShardSliceProperty $job 'started_at') (Get-ShardSliceProperty $job 'completed_at')
            Step       = $stepMinutes
            SelfHosted = @(Get-ShardSliceProperty $job 'labels') -cnotcontains 'ubuntu-latest'
            Conclusion = [string] (Get-ShardSliceProperty $job 'conclusion')
        }
    }
    return $timings
}

function Get-ShardPackReport {
    <#
    .SYNOPSIS
        Pure. Predicted against actual test minutes for every job of a shard-pack-plan, from the run's own jobs.
    .DESCRIPTION
        When the plan was applied (TEST_IMPACT_REPACK=on and the matrix was packed) a packed job's actual is its
        test step. Otherwise the run was one job per shard, and a would-be job's actual is the sum of its slices'
        test steps - what the packed job would have spent testing. The packed runner-minutes are then an
        estimate and say so: the measured test steps plus one measured median fixed cost (job minus test step)
        per job.
    #>
    param([Parameter(Mandatory)] [object] $Plan, [Parameter(Mandatory)] [AllowEmptyCollection()] [AllowNull()] [object[]] $Jobs)
    $timings = Get-ShardJobTimings -Jobs $Jobs
    $applied = (Get-ShardSliceProperty $Plan 'applied') -eq $true
    $fixed = @{ hosted = [Collections.Generic.List[double]]::new(); selfHosted = [Collections.Generic.List[double]]::new() }
    foreach ($timing in $timings.Values) {
        if ($null -ne $timing.Step -and $null -ne $timing.Job) {
            $fixed[$(if ($timing.SelfHosted) { 'selfHosted' } else { 'hosted' })].Add($timing.Job - $timing.Step)
        }
    }
    $fixedHosted = if ($fixed.hosted.Count -gt 0) { Get-ShardTimingMedian $fixed.hosted.ToArray() } else { $null }
    $fixedSelfHosted = if ($fixed.selfHosted.Count -gt 0) { Get-ShardTimingMedian $fixed.selfHosted.ToArray() } else { $null }
    if ($null -eq $fixedSelfHosted) { $fixedSelfHosted = $fixedHosted }
    if ($null -eq $fixedHosted) { $fixedHosted = $fixedSelfHosted }

    $rows = [Collections.Generic.List[object]]::new()
    $sliceErrors = [Collections.Generic.List[double]]::new()
    $jobsNow = 0; $minutesNow = 0.0; $minutesPacked = 0.0; $measuredJobs = 0
    foreach ($job in @($Plan.jobs)) {
        $slices = @($job.slices)
        $actual = $null
        $jobMinutes = 0.0
        $ran = 0
        $complete = $true
        if ($applied -and $job.packed) {
            $timing = $timings[[string] $job.name]
            if ($null -eq $timing -or $null -eq $timing.Step) { $complete = $false }
            else { $actual = $timing.Step; $jobMinutes = $timing.Job; $ran = 1 }
        }
        else {
            $sum = 0.0
            foreach ($slice in $slices) {
                $timing = $timings[[string] $slice.name]
                if ($null -eq $timing -or $null -eq $timing.Step) { $complete = $false; continue }
                $sum += $timing.Step; $jobMinutes += $timing.Job; $ran++
                if ([string] $slice.source -cne 'default') { $sliceErrors.Add([double] $slice.predictedMinutes - $timing.Step) }
            }
            if ($complete) { $actual = $sum }
        }
        if ($complete) {
            $measuredJobs++
            $jobsNow += $ran
            $minutesNow += $jobMinutes
            $perJobFixed = if ($job.runner) { $fixedSelfHosted } else { $fixedHosted }
            $minutesPacked += $(if ($job.packed -and -not $applied) { $actual + [double] $perJobFixed } else { $jobMinutes })
        }
        $rows.Add([ordered]@{
                name = [string] $job.name; packed = [bool] $job.packed; slices = $slices.Count
                predictedMinutes = [double] $job.predictedMinutes
                actualMinutes = $(if ($null -ne $actual) { [Math]::Round([double] $actual, 2) } else { $null })
                errorMinutes = $(if ($null -ne $actual) { [Math]::Round([double] $job.predictedMinutes - [double] $actual, 2) } else { $null })
                unknown = @($slices | Where-Object { [string] $_.source -ceq 'default' }).Count
            })
    }
    $measuredPacked = @($rows | Where-Object { $_.packed -and $null -ne $_.actualMinutes })
    $jobErrors = [double[]] @($measuredPacked | ForEach-Object { [Math]::Abs([double] $_.errorMinutes) })
    $absolute = [double[]] @($sliceErrors | ForEach-Object { [Math]::Abs($_) })
    $round = { param($Value) if ($null -eq $Value) { return $null }; return [Math]::Round([double] $Value, 2) }
    return [ordered]@{
        schemaVersion            = $script:ShardPackSchemaVersion
        mode                     = [string] $Plan.mode
        applied                  = $applied
        targetMinutes            = [double] $Plan.targetMinutes
        slices                   = [int] $Plan.slices
        plannedJobs              = @($Plan.jobs).Count
        packedJobs               = [int] $Plan.packedJobs
        measuredJobs             = $measuredJobs
        jobsRunForMeasured       = $jobsNow
        fixedMinutesHosted       = & $round $fixedHosted
        fixedMinutesSelfHosted   = & $round $fixedSelfHosted
        runnerMinutesMeasured    = [Math]::Round($minutesNow, 1)
        runnerMinutesPacked      = [Math]::Round($minutesPacked, 1)
        runnerMinutesPackedIs    = $(if ($applied) { 'measured' } else { 'estimated' })
        sliceMedianAbsoluteError = $(if ($absolute.Count -gt 0) { & $round (Get-ShardTimingMedian $absolute) } else { $null })
        sliceMedianSignedError   = $(if ($sliceErrors.Count -gt 0) { & $round (Get-ShardTimingMedian $sliceErrors.ToArray()) } else { $null })
        packedJobMedianAbsError  = $(if ($jobErrors.Count -gt 0) { & $round (Get-ShardTimingMedian $jobErrors) } else { $null })
        longestPackedJobMinutes  = $(if ($measuredPacked.Count -gt 0) { ($measuredPacked | ForEach-Object { [double] $_.actualMinutes } | Measure-Object -Maximum).Maximum } else { $null })
        packedJobsOverTarget     = @($measuredPacked | Where-Object { [double] $_.actualMinutes -gt [double] $Plan.targetMinutes }).Count
        jobs                     = $rows.ToArray()
    }
}

function Get-ShardPackReportSummary {
    <# Markdown for the step summary: predicted against actual, per packed (or would-be packed) job. #>
    param([Parameter(Mandatory)] [object] $Report)
    $lines = [Collections.Generic.List[string]]::new()
    $what = if ($Report.applied) { 'packed' } else { 'would-be packed' }
    [void] $lines.Add('### Shard repacking: predicted against actual')
    [void] $lines.Add('')
    [void] $lines.Add("TEST_IMPACT_REPACK=$($Report.mode). $($Report.slices) slice(s) in $($Report.plannedJobs) job(s), $($Report.packedJobs) of them $what; $($Report.measuredJobs) job(s) have every test step measured.")
    [void] $lines.Add('')
    if ($Report.applied) {
        [void] $lines.Add("Runner-minutes of the measured jobs: **$($Report.runnerMinutesMeasured)** (measured).")
    }
    else {
        [void] $lines.Add("Runner-minutes of the measured jobs: **$($Report.runnerMinutesMeasured)** as run, in $($Report.jobsRunForMeasured) job(s) (measured); **$($Report.runnerMinutesPacked)** packed (estimated: the measured test steps plus one fixed cost per packed job - the median of job minus test step in this run, $($Report.fixedMinutesHosted) min hosted and $($Report.fixedMinutesSelfHosted) min self-hosted).")
        [void] $lines.Add('')
        [void] $lines.Add("Prediction error per slice (predicted minus actual test step): median $($Report.sliceMedianSignedError) min, median absolute $($Report.sliceMedianAbsoluteError) min.")
    }
    [void] $lines.Add('')
    [void] $lines.Add("Per $what job: median absolute error $($Report.packedJobMedianAbsError) min; longest actual $($Report.longestPackedJobMinutes) min; $($Report.packedJobsOverTarget) over the $($Report.targetMinutes)-minute target.")
    $packed = @($Report.jobs | Where-Object { $_.packed })
    if ($packed.Count -gt 0) {
        [void] $lines.Add('')
        [void] $lines.Add('| Job | Slices | Predicted min | Actual min | Predicted - actual |')
        [void] $lines.Add('|---|---|---|---|---|')
        foreach ($job in $packed) {
            [void] $lines.Add("| $($job.name) | $($job.slices) | $($job.predictedMinutes) | $($job.actualMinutes) | $($job.errorMinutes) |")
        }
    }
    return ($lines -join "`n")
}
