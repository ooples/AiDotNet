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

      no map / unreadable        nothing to select against
      map sha != merge base      the map describes a different tree, so its line numbers are lies
      changed file not in index  never executed by ANY shard, or new - impact unknown
      shared-infrastructure path build, workflow or global config, whose blast radius is not
                                 expressible as executed lines

    Line-level, not file-level, deliberately: on Integration H-L, 29,528 executed lines sit inside
    files also containing 169,732 unexecuted ones, so 85% of a file's lines belong to no shard by
    way of that file. LayerHelper.cs alone spans 25,980 lines with 703 executed and a 21,328-line
    gap. Selecting on the file would pull in shards that never run the changed code.

.PARAMETER MapFile
    shard-map.json from New-ShardMap.ps1.

.PARAMETER BaseRef
    The merge base to diff against.

.PARAMETER SelfTest
    Runs the built-in checks and exits. Proves selection both fires and refuses to fire.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $BaseRef,
    [Parameter(ParameterSetName = 'Select')] [string] $OutFile,
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

function Get-ChangedRanges {
    <# Changed line ranges per file, from unified-diff hunk headers with zero context. #>
    param([string] $BaseRef)

    $diff = & git diff -U0 "$BaseRef...HEAD"
    if ($LASTEXITCODE -ne 0) { throw "git diff against '$BaseRef' failed." }

    $changed = @{}
    $current = $null
    foreach ($line in $diff) {
        if ($line.StartsWith('+++ b/')) {
            $current = $line.Substring(6).Trim()
        }
        elseif ($current -and $line.StartsWith('@@')) {
            # @@ -old,n +new,m @@ ; m omitted means 1, m = 0 means pure deletion
            if ($line -match '\+(\d+)(?:,(\d+))?') {
                $start = [int] $Matches[1]
                $count = if ($Matches[2]) { [int] $Matches[2] } else { 1 }
                if ($count -gt 0) {
                    if (-not $changed.ContainsKey($current)) {
                        $changed[$current] = [System.Collections.Generic.List[int]]::new()
                    }
                    # Parenthesised deliberately: inside @(...) the comma binds tighter than +,
                    # so @($start, $start + $count - 1) parses as ($start, $start) + $count - 1,
                    # i.e. array concatenation followed by subtraction from an array.
                    [void] $changed[$current].Add($start)
                    [void] $changed[$current].Add($start + $count - 1)
                }
                else {
                    # A deletion executes no new line, but the surrounding code changed meaning.
                    # Treat the deletion point as touched rather than invisible.
                    if (-not $changed.ContainsKey($current)) {
                        $changed[$current] = [System.Collections.Generic.List[int]]::new()
                    }
                    [void] $changed[$current].Add([Math]::Max(1, $start))
                    [void] $changed[$current].Add([Math]::Max(1, $start + 1))
                }
            }
        }
    }
    return $changed
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

    if ($failures.Count -gt 0) {
        Write-Host 'Select-Shards self-test FAILED:'
        foreach ($f in $failures) { Write-Host "  - $f" }
        exit 1
    }
    Write-Host 'Select-Shards self-test passed.'
    exit 0
}

# ---------------------------------------------------------------- selection

if (-not (Test-Path -LiteralPath $MapFile)) {
    Write-Host "::warning::no shard map at $MapFile - running the full matrix"
    $result = [pscustomobject]@{ escalate = $true; reason = 'map-missing'; reasons = @(); shards = @() }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

$map = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
$mergeBase = (& git merge-base HEAD $BaseRef).Trim()

# A map built from a different tree describes different line numbers. Selecting against it would be
# worse than not selecting: the ranges would look valid and mean nothing.
if ($map.sha -ne $mergeBase) {
    Write-Host "::warning::map built from $($map.sha) but the merge base is $mergeBase - running the full matrix"
    $result = [pscustomobject]@{ escalate = $true; reason = 'map-stale'; reasons = @(); shards = @() }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

$changed = Get-ChangedRanges -BaseRef $BaseRef
Write-Host "changed files: $($changed.Count)"

$selection = Select-ImpactedShards -Map $map -Changed $changed
if ($selection.Escalate) {
    Write-Host '::warning::selection escalated to the full matrix'
    foreach ($reason in $selection.Reasons) { Write-Host "  reason: $reason" }
}
else {
    Write-Host "selected $($selection.Shards.Count) of $($map.knownShards.Count) shard(s)"
    foreach ($shard in $selection.Shards) { Write-Host "  $shard" }
}

$result = [pscustomobject]@{
    escalate = $selection.Escalate
    reason   = $(if ($selection.Escalate) { 'impact-unknown' } else { 'selected' })
    reasons  = $selection.Reasons
    shards   = $selection.Shards
}
if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
