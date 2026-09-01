<#
.SYNOPSIS
    Chooses which CI shards a change needs, from the coverage-derived shard map.

.DESCRIPTION
    Intersects the PR's changed line ranges against the map's file -> shard index. A shard is
    selected when it executed at least one line the PR touched.

    FAIL SAFE IS THE POINT. Selection only ever saves time when it is confident; every uncertainty
    escalates to the full matrix and says why. A wrong escalation costs runner minutes, a wrong
    omission ships a regression - those are not symmetric, and the code treats them accordingly.
    Escalation triggers:

      no map                     nothing to select against
      map unreadable / malformed read or parse failed, or a required property is absent
      map commit not present     its line numbers cannot be resolved against this checkout
      changed file not in index  never executed by ANY shard, or new - impact unknown
      shared-infrastructure path build, workflow or global config, whose blast radius is not
                                 expressible as executed lines

    Line-level, not file-level, deliberately: on Integration H-L, 29,528 executed lines sit inside
    files also containing 169,732 unexecuted ones, so 85% of a file's lines belong to no shard by
    way of that file. LayerHelper.cs alone spans 25,980 lines with 703 executed and a 21,328-line
    gap. Selecting on the file would pull in shards that never run the changed code.

.PARAMETER MapFile
    shard-map.json from New-ShardMap.ps1.

.PARAMETER SelfTest
    Runs the built-in checks and exits. Proves selection both fires and refuses to fire.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $MapFile,
    [Parameter(ParameterSetName = 'Select')] [string] $OutFile,
    [Parameter(ParameterSetName = 'Select')] [string] $FileLevelFrom,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Paths whose effect is not expressible as executed source lines. A build property or workflow edit
# can change what every shard compiles or how it runs, so no coverage evidence can bound it.
$script:SharedInfrastructure = @(
    '.github/',
    'Directory.Build.props', 'Directory.Build.targets', 'Directory.Packages.props',
    'global.json', 'nuget.config', 'NuGet.config',
    '.editorconfig'
)

function Test-SharedInfrastructure {
    param([string] $Path)
    foreach ($prefix in $script:SharedInfrastructure) {
        if ($Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { return $true }
        if ($Path -ieq $prefix) { return $true }
    }
    return $false
}

function Test-RangeOverlap {
    param([int] $AStart, [int] $AEnd, [int] $BStart, [int] $BEnd)
    return -not ($AEnd -lt $BStart -or $BEnd -lt $AStart)
}

function ConvertTo-ChangedRanges {
    <#
        Changed line ranges per file, from unified-diff hunk headers with zero context.

        Pure so it can be self-tested: the coordinate choice below is the single most dangerous
        decision in the selector and a git fixture would hide it behind plumbing.
    #>
    param([string[]] $DiffLines)

    $changed = @{}
    $current = $null
    $deletedPath = $null
    foreach ($line in $DiffLines) {
        if ($line.StartsWith('--- ')) {
            # Both header lines are tracked, and $current is RESET here.
            #
            # A deletion is `--- a/<path>` followed by `+++ /dev/null`, so matching only `+++ b/`
            # left $current pointing at the PREVIOUS file: the deleted file's hunk was appended to
            # that file's ranges, and the deleted path was never reported at all. Shards executing
            # it were therefore not selected AND the unmapped-file escalation never fired - a
            # silent miss. Measured on a real two-file diff, an edit at line 100 of A plus a
            # deletion of B produced `src/A.cs -> 100,100,1,200` and no B.
            $from = $line.Substring(4).Trim()
            $deletedPath = if ($from -eq '/dev/null') { $null } else { $from -replace '^a/', '' }
            $current = $null
        }
        elseif ($line.StartsWith('+++ ')) {
            $to = $line.Substring(4).Trim()
            # `+++ /dev/null` means the file was deleted; attribute its hunks to the path it had.
            $current = if ($to -eq '/dev/null') { $deletedPath } else { $to -replace '^b/', '' }
        }
        elseif ($current -and $line.StartsWith('@@')) {
            # @@ -old,n +new,m @@ ; n omitted means 1, n = 0 means a pure insertion.
            #
            # The OLD side, deliberately. The map's ranges are in the coordinates of the commit it
            # was built at, and the caller diffs from exactly that commit, so the '-old' numbers are
            # already in the map's coordinate system. Reading '+new' instead looks correct and
            # silently misses: insert fifty lines above an edit and the edited line is looked up
            # fifty lines below where the map recorded it, so the shard covering it is not selected.
            if ($line -match '-(\d+)(?:,(\d+))?') {
                $start = [int] $Matches[1]
                $count = if ($Matches[2]) { [int] $Matches[2] } else { 1 }
                if (-not $changed.ContainsKey($current)) {
                    $changed[$current] = [System.Collections.Generic.List[int]]::new()
                }
                if ($count -gt 0) {
                    # Parenthesised deliberately: inside @(...) the comma binds tighter than +,
                    # so @($start, $start + $count - 1) parses as ($start, $start) + $count - 1,
                    # i.e. array concatenation followed by subtraction from an array.
                    [void] $changed[$current].Add($start)
                    [void] $changed[$current].Add($start + $count - 1)
                }
                else {
                    # A pure insertion occupies no line on this side, but it changed the meaning of
                    # the code around it. Treat the lines either side of the insertion point as
                    # touched rather than invisible.
                    [void] $changed[$current].Add([Math]::Max(1, $start))
                    [void] $changed[$current].Add([Math]::Max(1, $start + 1))
                }
            }
        }
    }
    return $changed
}

function Get-ChangedRanges {
    <#
        Diffs from the commit the MAP was built at, two-dot, so the '-old' side of every hunk is
        already in the map's coordinate system.

        Two-dot, not three-dot, and this is the whole design. Three-dot would diff from
        merge-base(map, HEAD), whose line numbers belong to neither the map nor the PR, which is the
        coordinate bug this function exists to avoid. Two-dot also drops the requirement that the
        map commit be an ancestor of HEAD - it is not, for any PR branched before last night's map -
        so no ancestry check is needed or wanted.

        The cost is over-selection: master commits between the PR's base and the map appear as
        changes too. That is one night of churn, it is the safe direction, and it is churn the PR
        merges with anyway.
    #>
    param([string] $MapSha)

    $diff = & git diff -U0 $MapSha HEAD
    if ($LASTEXITCODE -ne 0) { throw "git diff from the map commit '$MapSha' failed." }
    return ConvertTo-ChangedRanges -DiffLines $diff
}

function Select-ImpactedShardsByFile {
    <#
        FILE-level selection between two arbitrary trees, for the master-push case.

        Master pushes a merge commit whose tree nothing tested, but a PR run DID test a real tree;
        the only unverified part of master is diff(testedTree, master). A shard whose coverage
        touches none of those files behaves identically on both trees, so its result from that run
        still holds and it does not need running again.

        File-level, not line-level, and that is forced rather than lazy. The map's ranges are in
        the coordinates of the map's commit, while this diff is between two entirely different
        commits, so line numbers from it cannot be looked up in the map at all - the same
        coordinate mismatch that silently dropped covering shards before. File containment needs no
        coordinates, so it stays correct across any pair of trees. It over-selects, which is the
        safe direction.
    #>
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $ChangedFiles
    )

    $selected = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($shard in $Map.alwaysRun) { [void] $selected.Add([string] $shard) }

    $reasons = [System.Collections.Generic.List[string]]::new()
    $escalate = $false

    foreach ($path in ($ChangedFiles | Sort-Object -Unique)) {
        if (-not $path) { continue }
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
        foreach ($occurrence in @($entry.Value)) {
            [void] $selected.Add([string] $Map.knownShards[[int] $occurrence.s])
        }
    }

    return [pscustomobject]@{
        Escalate = $escalate
        Reasons  = $reasons
        Shards   = @($selected | Sort-Object)
    }
}

function Select-ImpactedShards {
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [hashtable] $Changed
    )

    $selected = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($shard in $Map.alwaysRun) { [void] $selected.Add([string] $shard) }

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

        foreach ($occurrence in @($entry.Value)) {
            $shardName = [string] $Map.knownShards[[int] $occurrence.s]
            if ($selected.Contains($shardName)) { continue }
            # Both sides are FLAT [start, end, start, end, ...]. Nested arrays are a trap: a file
            # with exactly one executed range comes back from ConvertFrom-Json as two loose
            # integers, and @() on a one-element collection flattens the same way, so every index
            # silently shifts. Flat pairs cannot be unrolled into something that still indexes.
            $ranges = @($occurrence.r)
            $hunks = @($Changed[$path])
            $hit = $false
            for ($i = 0; $i + 1 -lt $ranges.Count -and -not $hit; $i += 2) {
                for ($j = 0; $j + 1 -lt $hunks.Count; $j += 2) {
                    if (Test-RangeOverlap -AStart ([int] $hunks[$j]) -AEnd ([int] $hunks[$j + 1]) `
                                          -BStart ([int] $ranges[$i]) -BEnd ([int] $ranges[$i + 1])) {
                        $hit = $true; break
                    }
                }
            }
            if ($hit) { [void] $selected.Add($shardName) }
        }
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
    function Assert-True { param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) } }

    # Round-tripped through JSON: an in-memory map hides ConvertFrom-Json's single-element
    # unrolling, which is exactly what broke the real path while this self-test passed.
    $map = @{
        schemaVersion = 1
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

    # 1. A change on a line a shard executed selects that shard - the gate must FIRE.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14) }
    Assert-True (-not $r.Escalate) 'a mapped, covered change must not escalate'
    Assert-True ($r.Shards -contains 'Alpha') 'a change on Alpha''s executed lines must select Alpha'
    Assert-True (-not ($r.Shards -contains 'Beta')) 'a change outside Beta''s lines must NOT select Beta'


    # 2. A change in a GAP selects neither - the gate must REFUSE to fire. Without this check a
    #    selector that returned every shard for every input would satisfy check 1 and be useless.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(50, 60) }
    Assert-True (-not ($r.Shards -contains 'Alpha')) 'a change in an unexecuted gap must not select Alpha'
    Assert-True (-not ($r.Shards -contains 'Beta')) 'a change in an unexecuted gap must not select Beta'

    # 3. Always-run shards appear regardless, including when nothing else is selected.
    Assert-True ($r.Shards -contains 'HeavyNoCoverage') 'shards without coverage must always be selected'

    # 4. Fail safe: an unmapped file, and shared infrastructure, both escalate.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Unmapped.cs' = @(1, 5) }
    Assert-True $r.Escalate 'an unmapped file must escalate to the full matrix'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props' = @(1, 2) }
    Assert-True $r.Escalate 'a dependency-property change must escalate'
    $r = Select-ImpactedShards -Map $map -Changed @{ '.github/workflows/ci.yml' = @(1, 2) }
    Assert-True $r.Escalate 'a workflow change must escalate'

    # 5. Boundaries. The first and last executed line count as hits; the lines either side do not.
    foreach ($edge in @(10, 20)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True ($r.Shards -contains 'Alpha') "executed boundary line $edge must select Alpha"
    }
    foreach ($edge in @(9, 21)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True (-not ($r.Shards -contains 'Alpha')) "unexecuted boundary line $edge must not select Alpha"
    }

    # 6. COORDINATE SYSTEM. The map's ranges are in the coordinates of the commit the map was
    #    built at, so the diff must report line numbers on THAT side - the '-old' side. A PR that
    #    inserts lines above an edit shifts the two sides apart, and reading the '+new' side then
    #    looks up the edited line at a number the map has never heard of. The failure mode is a
    #    silent MISS: a shard that covers the edit is not selected, and the regression ships.
    #
    #    Shape below: 50 lines inserted at the top, then a one-line edit at old line 500, which
    #    the '+new' side calls 550.
    $diff = @(
        '--- a/src/Covered.cs',
        '+++ b/src/Covered.cs',
        '@@ -1,0 +1,50 @@',
        '@@ -500 +550 @@'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    $ranges = @($r['src/Covered.cs'])
    Assert-True ($ranges -contains 500) 'the edited line must be reported in the map''s coordinates (old side), not the new side'
    Assert-True (-not ($ranges -contains 550)) 'the new-side line number must NOT be reported - the map cannot resolve it'

    # 7. The two halves composed, because 6 alone only proves the parser reports a different
    #    number - not that the number changes the answer. Measured against the real map, a ONE-line
    #    insertion above an edit was enough to drop Integration H-L from the selection: executed
    #    line 67 shifts to 68, and 68 sits in a gap. That is an ordinary PR (add a using, edit a
    #    method below), so the pre-fix selector would have skipped covering shards routinely.
    $diff = @(
        '--- a/src/Covered.cs',
        '+++ b/src/Covered.cs',
        '@@ -0,0 +1 @@',
        '@@ -20 +21 @@'
    )
    $r = Select-ImpactedShards -Map $map -Changed (ConvertTo-ChangedRanges -DiffLines $diff)
    Assert-True ($r.Shards -contains 'Alpha') 'a one-line insertion above an edit must not lose the covering shard'

    # 8. FILE-level selection, used when master validates only its delta from a tested tree.
    #    It must select every shard touching a changed file regardless of WHERE in the file, since
    #    the two trees share no line numbering.
    $r = Select-ImpactedShardsByFile -Map $map -ChangedFiles @('src/Covered.cs')
    Assert-True ($r.Shards -contains 'Alpha' -and $r.Shards -contains 'Beta') `
        'file-level selection must select every shard executing that file'
    Assert-True (-not $r.Escalate) 'a mapped file must not escalate'

    # 9. And it must still refuse: an untouched file selects nothing beyond always-run. Without
    #    this a selector returning everything would satisfy 8.
    $r = Select-ImpactedShardsByFile -Map $map -ChangedFiles @('src/SingleRange.cs')
    Assert-True ($r.Shards -contains 'Alpha') 'the shard executing the changed file is selected'
    Assert-True (-not ($r.Shards -contains 'Beta')) 'a shard not executing the changed file is not selected'
    Assert-True ($r.Shards -contains 'HeavyNoCoverage') 'always-run shards still appear'

    # 10. Same fail-safes as line-level selection.
    $r = Select-ImpactedShardsByFile -Map $map -ChangedFiles @('src/Unmapped.cs')
    Assert-True $r.Escalate 'an unmapped file must escalate'
    $r = Select-ImpactedShardsByFile -Map $map -ChangedFiles @('Directory.Packages.props')
    Assert-True $r.Escalate 'shared infrastructure must escalate'

    # 11. No changed files at all - the delta is empty, so only always-run shards remain. This is
    #     the case worth the most: master merged a tree whose difference from the tested one
    #     touches nothing, and the entire matrix collapses.
    $r = Select-ImpactedShardsByFile -Map $map -ChangedFiles @()
    Assert-True (-not $r.Escalate) 'an empty delta must not escalate'
    Assert-True ($r.Shards.Count -eq 1 -and $r.Shards -contains 'HeavyNoCoverage') `
        'an empty delta selects only the always-run shards'

    # 12. DELETIONS. git writes `--- a/<path>` then `+++ /dev/null`, so a parser keyed only on
    #     `+++ b/` leaves the previous file current: the deletion's hunk lands on THAT file and the
    #     deleted path is never reported, so shards executing it are not selected and the
    #     unmapped-file escalation never fires. Both halves are checked.
    $diff = @(
        '--- a/src/Covered.cs',
        '+++ b/src/Covered.cs',
        '@@ -100 +100 @@',
        '--- a/src/Deleted.cs',
        '+++ /dev/null',
        '@@ -1,200 +0,0 @@'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($r.ContainsKey('src/Deleted.cs')) 'a deleted file must be reported under its own path'
    Assert-True (@($r['src/Covered.cs']).Count -eq 2) 'a deletion must not append its ranges to the previous file'

    # 13. Additions are the mirror: `--- /dev/null` then `+++ b/<path>`.
    $diff = @('--- /dev/null', '+++ b/src/Added.cs', '@@ -0,0 +1,50 @@')
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($r.ContainsKey('src/Added.cs')) 'an added file must be reported under its new path'
    Assert-True (-not $r.ContainsKey('/dev/null')) '/dev/null must never appear as a path'

    if ($failures.Count -gt 0) {
        Write-Host 'Select-Shards self-test FAILED:'
        foreach ($f in $failures) { Write-Host "  - $f" }
        exit 1
    }
    Write-Host 'Select-Shards self-test passed.'
    exit 0
}

# ---------------------------------------------------------------- selection

function Exit-Escalated {
    <#
        Emit the full-matrix result and stop. Every "we cannot be confident" path ends here, and
        they must all write the SAME artifact: the caller reads selection.json to decide the matrix,
        so a path that exits without writing it does not escalate - it kills the job, and the test
        matrix never runs at all.
    #>
    param([Parameter(Mandatory)] [string] $Reason, [string] $Message)

    if ($Message) { Write-Host "::warning::$Message" }
    $result = [pscustomobject]@{ escalate = $true; reason = $Reason; reasons = @(); shards = @() }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

if (-not (Test-Path -LiteralPath $MapFile)) {
    Exit-Escalated -Reason 'map-missing' -Message "no shard map at $MapFile - running the full matrix"
}

# Read and shape-check together, because both failures mean the same thing: the map cannot be
# trusted to say which shards cover a line.
#
# The try/catch is load-bearing, not defensive habit. $ErrorActionPreference is Stop, so a truncated
# or malformed map made ConvertFrom-Json THROW - and a throw exits before any result is written, so
# the caller sees no selection.json, the job fails, and the matrix never runs. The header has always
# listed "no map / unreadable" as an escalation trigger; only the missing half was implemented.
$map = $null
try {
    $map = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json

    # A map missing any of these is not a map. Checked explicitly because the failure is otherwise
    # silent and much later: a null .files makes every changed file look unmapped, which escalates
    # for a misleading reason, and a null .knownShards indexes as empty so every selected shard
    # resolves to a blank name.
    foreach ($required in 'sha', 'knownShards', 'alwaysRun', 'files') {
        if (-not $map.PSObject.Properties[$required]) { throw "no '$required' property" }
    }
}
catch {
    Exit-Escalated -Reason 'map-unreadable' `
        -Message "the shard map at $MapFile could not be read - running the full matrix ($($_.Exception.Message))"
}

# The map's own commit is the reference, NOT the merge base. An earlier revision required
# map.sha -eq mergeBase, which reads like prudence and is in fact a switch that pins the feature
# off: a map built nightly on master equals a PR's merge base essentially never, so every run
# escalated and the saving was always zero while the logs said "map-stale" and looked healthy.
#
# What actually matters is that the map's line numbers can be resolved, and diffing from that exact
# commit guarantees they can. So the only real precondition is that the commit is present locally -
# a shallow checkout, or a map from a branch that has since been pruned, is what fails here.
$mapSha = [string] $map.sha
& git cat-file -e "$mapSha^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Exit-Escalated -Reason 'map-unresolvable' `
        -Message "the map's commit $mapSha is not present in this checkout - running the full matrix"
}

if ($FileLevelFrom) {
    # Master-push mode: validate only what differs from the tree a PR run already tested.
    & git cat-file -e "$FileLevelFrom^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Exit-Escalated -Reason 'tested-sha-unresolvable' `
            -Message "the tested commit $FileLevelFrom is not present in this checkout - running the full matrix"
    }
    $files = @(& git diff --name-only $FileLevelFrom HEAD)
    if ($LASTEXITCODE -ne 0) { Exit-Escalated -Reason 'diff-failed' -Message "git diff from $FileLevelFrom failed" }
    Write-Host "delta from $FileLevelFrom : $($files.Count) changed file(s)"
    $selection = Select-ImpactedShardsByFile -Map $map -ChangedFiles $files
}
else {
    $changed = Get-ChangedRanges -MapSha $mapSha
    Write-Host "changed files: $($changed.Count)"
    $selection = Select-ImpactedShards -Map $map -Changed $changed
}
if ($selection.Escalate) {
    Write-Host '::warning::selection escalated to the full matrix'
    foreach ($reason in $selection.Reasons) { Write-Host "  reason: $reason" }
}
else {
    # Denominator counts every shard that COULD run - mapped plus always-run - otherwise a
    # selection including an always-run shard reads as "2 of 1".
    $total = @($map.knownShards).Count + @($map.alwaysRun).Count
    Write-Host "selected $($selection.Shards.Count) of $total shard(s)"
    foreach ($shard in $selection.Shards) { Write-Host "  $shard" }
}

$result = [pscustomobject]@{
    escalate = $selection.Escalate
    reason   = $(if ($selection.Escalate) { 'impact-unknown' } else { 'selected' })
    reasons  = $selection.Reasons
    shards   = $selection.Shards
}
if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
