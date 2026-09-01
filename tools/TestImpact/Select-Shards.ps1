<#
.SYNOPSIS
    Chooses which CI shards a pull request needs, from the coverage-derived shard map.

.DESCRIPTION
    Intersects changed line ranges against the map's file -> shard index. Selection is deliberately
    fail-safe: every changed file and every changed hunk must be accounted for, or the caller is told
    to run the full matrix.

.PARAMETER MapFile
    shard-map.json from New-ShardMap.ps1.

.PARAMETER ExpectedShards
    The complete current shard manifest. A map for a different shard universe is never trusted.

.PARAMETER SelfTest
    Runs the built-in adversarial checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string[]] $ExpectedShards,
    [Parameter(ParameterSetName = 'Select')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:SharedInfrastructure = @(
    '.github/',
    'Directory.Build.props', 'Directory.Build.targets', 'Directory.Packages.props',
    'global.json', 'nuget.config', 'NuGet.config',
    '.editorconfig'
)

function Test-SharedInfrastructure {
    param([string] $Path)
    # Two kinds of entry, matched differently on purpose:
    #
    #   trailing '/'  a directory - prefix match
    #   otherwise     a config FILE NAME - matched by basename at ANY depth, because MSBuild and
    #                 NuGet apply Directory.Build.props / Directory.Packages.props / nuget.config
    #                 per-directory, so src/Directory.Build.props changes what a subtree compiles
    #                 just as surely as the root one (and src/Directory.Build.props exists here).
    #
    # An earlier revision used StartsWith for everything, which missed those nested files AND
    # escalated on unrelated look-alikes such as Directory.Packages.props.backup.
    $name = [System.IO.Path]::GetFileName($Path)
    foreach ($entry in $script:SharedInfrastructure) {
        if ($entry.EndsWith('/')) {
            if ($Path.StartsWith($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
        elseif ($name -ieq $entry) { return $true }
    }
    return $false
}

function Test-RangeOverlap {
    param([int] $AStart, [int] $AEnd, [int] $BStart, [int] $BEnd)
    return -not ($AEnd -lt $BStart -or $BEnd -lt $AStart)
}

function ConvertTo-ValidatedInteger {
    param(
        [Parameter(Mandatory)] $Value,
        [Parameter(Mandatory)] [string] $What,
        [int] $Minimum = 0,
        [int] $Maximum = [int]::MaxValue
    )

    $isInteger = $Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
    if (-not $isInteger) { throw "$What must be an integer." }

    $number = [long] $Value
    if ($number -lt $Minimum -or $number -gt $Maximum) {
        throw "$What must be between $Minimum and $Maximum."
    }
    return [int] $number
}

function Assert-ShardMap {
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [string[]] $Expected
    )

    foreach ($required in 'schemaVersion', 'sha', 'knownShards', 'alwaysRun', 'files') {
        if (-not $Map.PSObject.Properties[$required]) { throw "no '$required' property" }
    }
    $schemaVersion = ConvertTo-ValidatedInteger -Value $Map.schemaVersion -What 'schemaVersion' -Minimum 1
    if ($schemaVersion -ne 1) { throw "unsupported schemaVersion $schemaVersion" }
    if ([string] $Map.sha -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
        throw 'sha must be a complete hexadecimal Git object id'
    }
    if ($Map.knownShards -isnot [array] -or $Map.alwaysRun -isnot [array]) {
        throw 'knownShards and alwaysRun must be arrays'
    }
    if ($Map.files -isnot [pscustomobject]) { throw 'files must be an object' }

    $expectedSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in @($Expected)) {
        if ([string]::IsNullOrWhiteSpace([string] $name)) { throw 'expected shard names must be nonempty' }
        if (-not $expectedSet.Add([string] $name)) { throw "duplicate expected shard '$name'" }
    }
    if ($expectedSet.Count -eq 0) { throw 'the expected shard manifest is empty' }

    $known = @($Map.knownShards)
    $always = @($Map.alwaysRun)
    if ($known.Count -eq 0) { throw 'knownShards is empty' }

    $mapSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($group in @(@{ Name = 'knownShards'; Values = $known }, @{ Name = 'alwaysRun'; Values = $always })) {
        foreach ($nameValue in $group.Values) {
            $name = [string] $nameValue
            if ([string]::IsNullOrWhiteSpace($name)) { throw "$($group.Name) contains an empty shard name" }
            if (-not $mapSet.Add($name)) { throw "duplicate or overlapping shard '$name'" }
        }
    }

    $missing = @($expectedSet | Where-Object { -not $mapSet.Contains($_) } | Sort-Object)
    $extra = @($mapSet | Where-Object { -not $expectedSet.Contains($_) } | Sort-Object)
    if ($missing.Count -gt 0 -or $extra.Count -gt 0) {
        throw "map shard universe differs from the manifest (missing: $($missing -join ', '); extra: $($extra -join ', '))"
    }

    $fileProperties = @($Map.files.PSObject.Properties)
    if ($fileProperties.Count -eq 0) { throw 'files index is empty' }
    if ($Map.PSObject.Properties['fileCount']) {
        $fileCount = ConvertTo-ValidatedInteger -Value $Map.fileCount -What 'fileCount'
        if ($fileCount -ne $fileProperties.Count) { throw 'fileCount does not match the files index' }
    }
    foreach ($fileProperty in $fileProperties) {
        if ([string]::IsNullOrWhiteSpace([string] $fileProperty.Name)) { throw 'files contains an empty path' }
        if ($fileProperty.Value -isnot [array]) { throw "'$($fileProperty.Name)' occurrences must be an array" }
        $occurrences = @($fileProperty.Value)
        if ($occurrences.Count -eq 0) { throw "'$($fileProperty.Name)' has no occurrences" }
        foreach ($occurrence in $occurrences) {
            if (-not $occurrence.PSObject.Properties['s'] -or -not $occurrence.PSObject.Properties['r']) {
                throw "'$($fileProperty.Name)' has an occurrence without s or r"
            }
            [void] (ConvertTo-ValidatedInteger -Value $occurrence.s -What "'$($fileProperty.Name)' shard index" `
                -Minimum 0 -Maximum ($known.Count - 1))
            if ($occurrence.r -isnot [array]) { throw "'$($fileProperty.Name)' ranges must be an array" }
            $ranges = @($occurrence.r)
            if ($ranges.Count -eq 0 -or $ranges.Count % 2 -ne 0) {
                throw "'$($fileProperty.Name)' ranges must be nonempty start/end pairs"
            }
            for ($i = 0; $i -lt $ranges.Count; $i += 2) {
                $start = ConvertTo-ValidatedInteger -Value $ranges[$i] -What "'$($fileProperty.Name)' range start" -Minimum 1
                $end = ConvertTo-ValidatedInteger -Value $ranges[$i + 1] -What "'$($fileProperty.Name)' range end" -Minimum 1
                if ($start -gt $end) { throw "'$($fileProperty.Name)' range start $start exceeds end $end" }
            }
        }
    }
}

function ConvertTo-ChangedRanges {
    <#
        Parses zero-context hunks in the map commit's coordinates. ChangedFiles is authoritative:
        rename-only, binary and mode-only changes do not necessarily have an @@ header, but they
        must still reach the fail-safe selector.
    #>
    param(
        [AllowEmptyCollection()] [string[]] $DiffLines,
        [AllowEmptyCollection()] [string[]] $ChangedFiles = @()
    )

    $changed = @{}
    $current = $null
    $deletedPath = $null
    # Hunk BODY lines still pending. Headers are only recognised while this is zero, because diff
    # body lines are raw file content behind a one-character prefix, and content can forge any
    # header: a REMOVED line whose text begins with '-- ' is rendered '--- ...', byte-identical to
    # an old-file header. Reproduced with real git: deleting the line '-- remove me' emitted
    # '--- remove me', the old parser took it as a header, nulled $current, and silently dropped
    # every later hunk of that file - under-selection with no escalation. A zero-context hunk
    # '@@ -a,n +b,m @@' is followed by exactly n+m body lines (plus uncounted '\ No newline'
    # markers), so counting them makes body content inert no matter what it says.
    $pendingBody = 0
    foreach ($line in $DiffLines) {
        if ($pendingBody -gt 0) {
            if (-not $line.StartsWith('\')) { $pendingBody-- }
            continue
        }
        if ($line.StartsWith('--- ')) {
            $from = $line.Substring(4).Trim()
            $deletedPath = if ($from -eq '/dev/null') { $null } else { $from -replace '^a/', '' }
            $current = $null
        }
        elseif ($line.StartsWith('+++ ')) {
            $to = $line.Substring(4).Trim()
            $current = if ($to -eq '/dev/null') { $deletedPath } else { $to -replace '^b/', '' }
        }
        elseif ($line.StartsWith('@@') -and $line -match '^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@') {
            $start = [int] $Matches[1]
            $count = if ($Matches[2]) { [int] $Matches[2] } else { 1 }
            $newCount = if ($Matches[4]) { [int] $Matches[4] } else { 1 }
            $pendingBody = $count + $newCount
            if ($current) {
                if (-not $changed.ContainsKey($current)) {
                    $changed[$current] = [System.Collections.Generic.List[int]]::new()
                }
                if ($count -gt 0) {
                    [void] $changed[$current].Add($start)
                    [void] $changed[$current].Add($start + $count - 1)
                }
                else {
                    [void] $changed[$current].Add([Math]::Max(1, $start))
                    [void] $changed[$current].Add([Math]::Max(1, $start + 1))
                }
            }
        }
    }

    foreach ($path in $ChangedFiles) {
        if (-not [string]::IsNullOrWhiteSpace($path) -and -not $changed.ContainsKey($path)) {
            $changed[$path] = [System.Collections.Generic.List[int]]::new()
        }
    }
    return $changed
}

function Get-ChangedRanges {
    param([string] $MapSha)

    $changedFiles = @(& git -c core.quotepath=false diff --name-only $MapSha HEAD --)
    if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from '$MapSha' failed" }
    $diff = @(& git -c core.quotepath=false diff --no-ext-diff -U0 $MapSha HEAD --)
    if ($LASTEXITCODE -ne 0) { throw "git diff from '$MapSha' failed" }
    return ConvertTo-ChangedRanges -DiffLines $diff -ChangedFiles $changedFiles
}

function Select-ImpactedShards {
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [hashtable] $Changed
    )

    $selected = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($shard in @($Map.alwaysRun)) { [void] $selected.Add([string] $shard) }
    $reasons = [System.Collections.Generic.List[string]]::new()
    $escalate = $false

    foreach ($path in ($Changed.Keys | Sort-Object)) {
        if (Test-SharedInfrastructure -Path $path) {
            $escalate = $true
            [void] $reasons.Add("shared infrastructure: $path")
            continue
        }

        $entry = $Map.files.PSObject.Properties[$path]
        if (-not $entry) {
            $escalate = $true
            [void] $reasons.Add("not executed by any mapped shard: $path")
            continue
        }

        $hunks = @($Changed[$path])
        if ($hunks.Count -eq 0 -or $hunks.Count % 2 -ne 0) {
            $escalate = $true
            [void] $reasons.Add("changed file has no trustworthy line hunks: $path")
            continue
        }

        $covered = [bool[]]::new([int] ($hunks.Count / 2))
        foreach ($occurrence in @($entry.Value)) {
            $ranges = @($occurrence.r)
            $shardName = [string] $Map.knownShards[[int] $occurrence.s]
            for ($h = 0; $h + 1 -lt $hunks.Count; $h += 2) {
                $hit = $false
                for ($i = 0; $i + 1 -lt $ranges.Count; $i += 2) {
                    if (Test-RangeOverlap -AStart ([int] $hunks[$h]) -AEnd ([int] $hunks[$h + 1]) `
                                          -BStart ([int] $ranges[$i]) -BEnd ([int] $ranges[$i + 1])) {
                        $hit = $true
                        break
                    }
                }
                if ($hit) {
                    $covered[[int] ($h / 2)] = $true
                    [void] $selected.Add($shardName)
                }
            }
        }

        for ($h = 0; $h + 1 -lt $hunks.Count; $h += 2) {
            if (-not $covered[[int] ($h / 2)]) {
                $escalate = $true
                [void] $reasons.Add("changed range $($hunks[$h])-$($hunks[$h + 1]) is not executed by any mapped shard: $path")
            }
        }
    }

    if ($selected.Count -eq 0) {
        $escalate = $true
        [void] $reasons.Add('selection was empty')
    }

    return [pscustomobject]@{
        Escalate = $escalate
        Reasons  = $reasons
        Shards   = @($selected | Sort-Object)
    }
}

# ---------------------------------------------------------------- self-test

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True {
        param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) }
    }
    function Assert-Throws {
        param([scriptblock] $Action, [string] $What)
        $threw = $false
        try { & $Action } catch { $threw = $true }
        Assert-True $threw $What
    }

    $map = @{
        schemaVersion = 1
        sha           = '0123456789abcdef0123456789abcdef01234567'
        knownShards   = @('Alpha', 'Beta')
        alwaysRun     = @('HeavyNoCoverage')
        files         = @{
            'src/Covered.cs' = @(
                @{ s = 0; r = @(10, 20) },
                @{ s = 1; r = @(100, 110) }
            )
            'src/SingleRange.cs' = @( @{ s = 0; r = @(5, 6) } )
        }
    } | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json
    $expected = @('Alpha', 'Beta', 'HeavyNoCoverage')
    try { Assert-ShardMap -Map $map -Expected $expected } catch { [void] $failures.Add("valid map rejected: $_") }

    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14) }
    Assert-True (-not $r.Escalate) 'a mapped, fully covered change must not escalate'
    Assert-True ($r.Shards -contains 'Alpha') 'a change on Alpha lines must select Alpha'
    Assert-True (-not ($r.Shards -contains 'Beta')) 'a change outside Beta lines must not select Beta'
    Assert-True ($r.Shards -contains 'HeavyNoCoverage') 'always-run shards must always be selected'

    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14, 50, 60) }
    Assert-True $r.Escalate 'mixed covered and uncovered hunks must escalate'
    Assert-True ($r.Shards -contains 'Alpha') 'covered hunks are still reported before escalation'

    foreach ($change in @(
        @{ Path = 'src/Unmapped.cs'; Why = 'an unmapped file must escalate' },
        @{ Path = 'Directory.Packages.props'; Why = 'dependency infrastructure must escalate' },
        @{ Path = '.github/workflows/ci.yml'; Why = 'workflow infrastructure must escalate' }
    )) {
        $r = Select-ImpactedShards -Map $map -Changed @{ $change.Path = @(1, 2) }
        Assert-True $r.Escalate $change.Why
    }

    foreach ($edge in @(10, 20)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True ($r.Shards -contains 'Alpha') "executed boundary line $edge must select Alpha"
    }
    foreach ($edge in @(9, 21)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True $r.Escalate "unexecuted boundary line $edge must escalate"
    }

    # Faithful to real git: every hunk header is followed by its body lines. An earlier revision
    # used header-only fixtures, which real git never emits - and which masked the forged-header
    # parse bug that body-line counting exists to prevent (see check 15).
    $diff = @('--- a/src/Covered.cs', '+++ b/src/Covered.cs', '@@ -1,0 +1,50 @@') +
            @(1..50 | ForEach-Object { "+inserted $_" }) +
            @('@@ -500 +550 @@', '-old text', '+new text')
    $parsed = ConvertTo-ChangedRanges -DiffLines $diff
    $ranges = @($parsed['src/Covered.cs'])
    Assert-True ($ranges -contains 500) 'the parser must report old-side line numbers'
    Assert-True (-not ($ranges -contains 550)) 'the parser must not report new-side line numbers'

    $diff = @('--- a/src/Covered.cs', '+++ b/src/Covered.cs', '@@ -100 +100 @@', '-x', '+y',
              '--- a/src/Deleted.cs', '+++ /dev/null', '@@ -1,200 +0,0 @@') +
            @(1..200 | ForEach-Object { "-gone $_" })
    $parsed = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($parsed.ContainsKey('src/Deleted.cs')) 'a deleted file must be reported under its own path'
    Assert-True (@($parsed['src/Covered.cs']).Count -eq 2) 'a deletion must not contaminate the prior file'

    $parsed = ConvertTo-ChangedRanges -DiffLines @() -ChangedFiles @('src/Renamed.cs', 'assets/blob.bin')
    Assert-True ($parsed.ContainsKey('src/Renamed.cs')) 'a rename-only file must be present without a hunk'
    Assert-True ($parsed.ContainsKey('assets/blob.bin')) 'a binary file must be present without a hunk'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @() }
    Assert-True $r.Escalate 'a mapped file without hunks must escalate'

    $badRange = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badRange.files.'src/Covered.cs'[0].r = @(10)
    Assert-Throws { Assert-ShardMap -Map $badRange -Expected $expected } 'an odd range list must be rejected'

    $badIndex = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badIndex.files.'src/Covered.cs'[0].s = 99
    Assert-Throws { Assert-ShardMap -Map $badIndex -Expected $expected } 'an out-of-range shard index must be rejected'

    $badSchema = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badSchema.schemaVersion = 2
    Assert-Throws { Assert-ShardMap -Map $badSchema -Expected $expected } 'an unknown map schema must be rejected'

    $overlap = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $overlap.alwaysRun = @('Alpha')
    Assert-Throws { Assert-ShardMap -Map $overlap -Expected @('Alpha', 'Beta') } `
        'known and always-run shard sets must be disjoint'

    $badLine = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badLine.files.'src/Covered.cs'[0].r = @(0, 10)
    Assert-Throws { Assert-ShardMap -Map $badLine -Expected $expected } 'non-positive map lines must be rejected'

    Assert-Throws { Assert-ShardMap -Map $map -Expected @('Alpha', 'Beta', 'HeavyNoCoverage', 'NewShard') } `
        'a stale shard universe must be rejected'

    $commaMap = @{
        schemaVersion = 1; sha = 'abcdefabcdefabcdefabcdefabcdefabcdefabcd'; knownShards = @('Comma, Shard'); alwaysRun = @()
        files = @{ 'src/Comma.cs' = @( @{ s = 0; r = @(1, 1) } ) }
    } | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    try { Assert-ShardMap -Map $commaMap -Expected @('Comma, Shard') } catch {
        [void] $failures.Add("a comma-containing shard name was rejected: $_")
    }
    $r = Select-ImpactedShards -Map $commaMap -Changed @{ 'src/Comma.cs' = @(1, 1) }
    Assert-True ($r.Shards -contains 'Comma, Shard') 'a comma-containing shard name must stay intact'

    # 14. Shared-infrastructure matching. File entries match by BASENAME at any depth - MSBuild
    #     and NuGet apply these files per-directory, and src/Directory.Build.props exists in this
    #     repo - while look-alike names must not match, and directories stay prefix-matched.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Directory.Build.props' = @(1, 2) }
    Assert-True $r.Escalate 'a NESTED Directory.Build.props must escalate'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/sub/nuget.config' = @(1, 2) }
    Assert-True $r.Escalate 'a nested nuget.config must escalate'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props.backup' = @(1, 2) }
    Assert-True (-not ($r.Reasons -join ';').Contains('shared infrastructure')) `
        'a look-alike suffix must not match shared infrastructure'
    $r = Select-ImpactedShards -Map $map -Changed @{ '.github/workflows/anything.yml' = @(1, 2) }
    Assert-True $r.Escalate 'the .github/ directory prefix still escalates'

    # 15. Hunk BODY content must be inert. A removed line whose text begins with '-- ' renders as
    #     '--- ...', byte-identical to an old-file header; the pre-fix parser nulled $current on it
    #     and silently dropped every later hunk of the file - under-selection with no escalation.
    #     This diff is verbatim real-git output for: delete the line '-- remove me', edit line 81.
    $diff = @(
        'diff --git a/F.cs b/F.cs',
        'index 9329992..2c35db3 100644',
        '--- a/F.cs',
        '+++ b/F.cs',
        '@@ -50 +49,0 @@ line 49',
        '--- remove me',
        '@@ -81 +80 @@ line 80',
        '-line 81',
        '+line 80 EDITED'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($r.ContainsKey('F.cs')) 'the file must be reported'
    Assert-True (@($r['F.cs']) -contains 81) 'the hunk AFTER the forged header line must survive'
    Assert-True (-not $r.ContainsKey('remove me')) 'body content must never become a path'

    # 16. And a forged header inside an ADDED body line must not smuggle a file in.
    $diff = @(
        '--- a/G.cs',
        '+++ b/G.cs',
        '@@ -5,0 +6,2 @@',
        '+--- a/EVIL.cs',
        '++++ b/EVIL.cs',
        '@@ -30 +32 @@',
        '-x',
        '+y'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True (-not $r.ContainsKey('EVIL.cs')) 'added body content must never become a path'
    Assert-True (@($r['G.cs']) -contains 30) 'the following real hunk still lands on the right file'

    if ($failures.Count -gt 0) {
        Write-Host 'Select-Shards self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'Select-Shards self-test passed.'
    exit 0
}

# ---------------------------------------------------------------- selection

function Exit-Escalated {
    param([Parameter(Mandatory)] [string] $Reason, [string] $Message)

    if ($Message) { Write-Host "::warning::$Message" }
    $result = [pscustomobject]@{ escalate = $true; reason = $Reason; reasons = @(); shards = @() }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

if (-not (Test-Path -LiteralPath $MapFile)) {
    Exit-Escalated -Reason 'map-missing' -Message "no shard map at $MapFile - running the full matrix"
}

$map = $null
try {
    $map = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
    Assert-ShardMap -Map $map -Expected $ExpectedShards
}
catch {
    Exit-Escalated -Reason 'map-unreadable' `
        -Message "the shard map at $MapFile could not be trusted - running the full matrix ($($_.Exception.Message))"
}

$mapSha = [string] $map.sha
& git cat-file -e "$mapSha^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Exit-Escalated -Reason 'map-unresolvable' `
        -Message "the map's commit $mapSha is not present in this checkout - running the full matrix"
}

try {
    $changed = Get-ChangedRanges -MapSha $mapSha
    Write-Host "changed files: $($changed.Count)"
    $selection = Select-ImpactedShards -Map $map -Changed $changed

    if ($selection.Escalate) {
        Write-Host '::warning::selection escalated to the full matrix'
        foreach ($reason in $selection.Reasons) { Write-Host "  reason: $reason" }
    }
    else {
        Write-Host "selected $($selection.Shards.Count) of $($ExpectedShards.Count) shard(s)"
        foreach ($shard in $selection.Shards) { Write-Host "  $shard" }
    }

    $result = [pscustomobject]@{
        escalate = $selection.Escalate
        reason   = $(if ($selection.Escalate) { 'impact-unknown' } else { 'selected' })
        reasons  = $selection.Reasons
        shards   = $selection.Shards
    }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
}
catch {
    Exit-Escalated -Reason 'selection-failed' `
        -Message "shard selection failed, so the full matrix will run: $($_.Exception.Message)"
}
