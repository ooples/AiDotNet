<#
.SYNOPSIS
    Decides which shards must be re-instrumented, and carries the rest forward from the previous map.

.DESCRIPTION
    Rebuilding the map means running the whole matrix with coverage on, and instrumentation costs
    3.21x on heavy shards - about 4.8 hours of queue occupancy. Most of that is wasted: on a typical
    night only a fraction of shards execute code that changed at all.

    A shard can be carried forward when EVERY file in its digest is byte-identical between the
    previous map's commit and the new one. That is the exact condition under which its recorded line
    ranges still mean what they said, so the carried digest is valid at the new commit rather than
    merely plausible - which matters, because a map mixing coordinate systems is the failure this
    whole feature has been bitten by twice.

    Everything else is instrumented:

      a shard whose covered files changed     its ranges may have moved
      a shard absent from the previous map    never mapped, or its last run failed
      any shard, when there is no previous map  first build

    Conservative by construction: a shard is only skipped on positive evidence that nothing it
    touches moved. Any doubt - unreadable map, unknown shard, missing digest - instruments.

    NOTE this narrows nothing about SELECTION. A carried shard is as fully mapped as a freshly
    instrumented one; the only thing saved is re-measuring what cannot have changed.

.PARAMETER PreviousMap
    shard-map.json from the last successful build. Absent or unreadable means instrument everything.

.PARAMETER ChangedFiles
    Repo-relative paths changed between the previous map's commit and the one being mapped.

.PARAMETER AllShards
    The current shard manifest.

.PARAMETER CarryOnly
    Restrict carrying to these shards. The nightly passes the heavy and timing shards here, because
    every OTHER shard collects coverage in any run and produces a fresh digest on success - and a
    carried digest colliding with a fresh one makes New-ShardMap abort the whole map, by design.
    A clean mapped shard NOT in this list is simply left to re-produce its own digest; it is neither
    instrumented by force nor carried. Omit to allow carrying everything (the self-contained case).

.PARAMETER CarryForwardDirectory
    Where to write reconstructed digests for the shards being carried forward. A digest is inverted
    straight out of the previous map, so New-ShardMap consumes it exactly like a fresh one and needs
    no knowledge of any of this.

.PARAMETER OutFile
    JSON: { instrument: [...], carried: [...] }.

.PARAMETER SelfTest
    Runs the built-in checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $PreviousMap,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [AllowEmptyCollection()] [string[]] $ChangedFiles,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string[]] $AllShards,
    [Parameter(ParameterSetName = 'Select')] [AllowEmptyCollection()] [string[]] $CarryOnly,
    [Parameter(ParameterSetName = 'Select')] [string] $CarryForwardDirectory,
    [Parameter(ParameterSetName = 'Select')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Split-CoverageWork {
    <#
        Pure: given a parsed map, the changed paths and the shard manifest, decide what to
        instrument and what may be carried. Returns names only; writing digests is the caller's job.
    #>
    param(
        [Parameter(Mandatory)] [AllowNull()] $Map,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $ChangedFiles,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $AllShards
    )

    if ($null -eq $Map) {
        return [pscustomobject]@{ Instrument = @($AllShards); Carried = @(); Reason = 'no previous map' }
    }

    $changed = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($f in $ChangedFiles) { if ($f) { [void] $changed.Add([string] $f) } }

    $known = @($Map.knownShards)

    # Which shards touch a changed file. Walked file-first because the map is indexed that way, and
    # a shard is disqualified by ONE changed file - there is no need to enumerate the rest.
    $dirty = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($prop in $Map.files.PSObject.Properties) {
        if (-not $changed.Contains([string] $prop.Name)) { continue }
        foreach ($occurrence in @($prop.Value)) {
            $i = [int] $occurrence.s
            if ($i -ge 0 -and $i -lt $known.Count) { [void] $dirty.Add([string] $known[$i]) }
        }
    }

    $mapped = [System.Collections.Generic.HashSet[string]]::new([string[]] $known, [System.StringComparer]::Ordinal)

    $instrument = [System.Collections.Generic.List[string]]::new()
    $carried    = [System.Collections.Generic.List[string]]::new()
    foreach ($s in $AllShards) {
        $name = [string] $s
        # Absent from the map means never mapped or last run failed - it has nothing to carry.
        if (-not $mapped.Contains($name)) { [void] $instrument.Add($name); continue }
        if ($dirty.Contains($name))       { [void] $instrument.Add($name); continue }
        [void] $carried.Add($name)
    }

    return [pscustomobject]@{
        Instrument = @($instrument)
        Carried    = @($carried)
        Reason     = "$($changed.Count) changed file(s)"
    }
}

function Export-ShardDigest {
    <#
        Rebuild one shard's digest by inverting the map. The map stores file -> [{s,r}], so a
        shard's digest is every file whose occurrences name it, with those ranges.
    #>
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [string] $Shard,
        [Parameter(Mandatory)] [string] $Path
    )

    $known = @($Map.knownShards)
    $index = [array]::IndexOf($known, $Shard)
    if ($index -lt 0) { throw "shard '$Shard' is not in the previous map" }

    $files = [ordered]@{}
    foreach ($prop in $Map.files.PSObject.Properties) {
        foreach ($occurrence in @($prop.Value)) {
            if ([int] $occurrence.s -ne $index) { continue }
            $files[$prop.Name] = @($occurrence.r)
        }
    }

    $digest = [ordered]@{
        schemaVersion = 1
        shard         = $Shard
        generatedUtc  = [string] $Map.generatedUtc
        carriedFrom   = [string] $Map.sha
        fileCount     = $files.Count
        files         = $files
    }
    $digest | ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath $Path -Encoding utf8 -NoNewline
    return $files.Count
}

# ---------------------------------------------------------------- self-test

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) } }

    $map = @{
        schemaVersion = 1
        sha           = 'abc123'
        generatedUtc  = '2026-09-01T00:00:00Z'
        knownShards   = @('Alpha', 'Beta')
        alwaysRun     = @('Heavy')
        files         = @{
            'src/A.cs' = @(@{ s = 0; r = @(10, 20) })
            'src/B.cs' = @(@{ s = 1; r = @(5, 6) })
            'src/C.cs' = @(@{ s = 0; r = @(1, 2) }, @{ s = 1; r = @(1, 2) })
        }
    } | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $all = @('Alpha', 'Beta', 'Heavy')

    # 1. Nothing changed: everything mapped is carried, and only the unmapped shard is instrumented.
    $r = Split-CoverageWork -Map $map -ChangedFiles @() -AllShards $all
    Assert-True ($r.Carried.Count -eq 2) 'with no changes both mapped shards are carried'
    Assert-True ($r.Instrument -contains 'Heavy') 'a shard absent from the map must be instrumented'
    Assert-True (-not ($r.Instrument -contains 'Alpha')) 'an unaffected shard must not be instrumented'

    # 2. A changed file dirties exactly the shards that execute it - the core of the saving, and
    #    the thing that would silently carry a stale digest if it were wrong.
    $r = Split-CoverageWork -Map $map -ChangedFiles @('src/A.cs') -AllShards $all
    Assert-True ($r.Instrument -contains 'Alpha') 'a shard executing a changed file must be instrumented'
    Assert-True ($r.Carried -contains 'Beta') 'a shard not executing it may still be carried'

    # 3. A file executed by SEVERAL shards dirties all of them.
    $r = Split-CoverageWork -Map $map -ChangedFiles @('src/C.cs') -AllShards $all
    Assert-True ($r.Instrument -contains 'Alpha' -and $r.Instrument -contains 'Beta') `
        'every shard executing a changed file must be instrumented'
    Assert-True ($r.Carried.Count -eq 0) 'nothing may be carried when all mapped shards are dirty'

    # 4. A changed file no shard executes carries everything - it cannot have moved any ranges.
    #    (Selection escalates on such a file separately; that is not this script's job.)
    $r = Split-CoverageWork -Map $map -ChangedFiles @('src/Unmapped.cs') -AllShards $all
    Assert-True ($r.Carried.Count -eq 2) 'a change to an unmapped file dirties no shard'

    # 5. No previous map instruments everything. Without this the first build would carry nothing
    #    forward and silently produce an empty map.
    $r = Split-CoverageWork -Map $null -ChangedFiles @('src/A.cs') -AllShards $all
    Assert-True ($r.Instrument.Count -eq 3 -and $r.Carried.Count -eq 0) 'no map means instrument everything'

    # 6. Ordinal matching: shard names differing only in case are different shards.
    $r = Split-CoverageWork -Map $map -ChangedFiles @('src/A.cs') -AllShards @('alpha')
    Assert-True ($r.Instrument -contains 'alpha') 'shard matching must be ordinal'

    # 7. Round-trip: a carried digest must reproduce exactly the ranges the map held, or carrying
    #    forward would quietly corrupt the very thing it is trying to preserve.
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "carry-$PID.json"
    [void] (Export-ShardDigest -Map $map -Shard 'Alpha' -Path $tmp)
    $d = Get-Content -LiteralPath $tmp -Raw | ConvertFrom-Json
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
    Assert-True ($d.shard -eq 'Alpha') 'the carried digest names its shard'
    Assert-True ($d.fileCount -eq 2) "Alpha's digest carries both files it executes"
    Assert-True (@($d.files.'src/A.cs') -join ',' -eq '10,20') 'carried ranges match the map exactly'
    Assert-True (-not $d.files.PSObject.Properties['src/B.cs']) "another shard's file must not leak in"
    Assert-True ($d.carriedFrom -eq 'abc123') 'the carried digest records the commit it came from'

    # 8. CarryOnly restricts carrying WITHOUT touching the instrument set. A shard displaced by
    #    the filter is a natural producer: it re-creates its own digest in any run, so carrying it
    #    would collide with the fresh one and New-ShardMap aborts the whole map on duplicates.
    $r = Split-CoverageWork -Map $map -ChangedFiles @() -AllShards $all
    $allow = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    [void] $allow.Add('Alpha')
    $restricted = @($r.Carried | Where-Object { $allow.Contains([string] $_) })
    Assert-True ($restricted.Count -eq 1 -and $restricted -contains 'Alpha') `
        'CarryOnly must keep exactly the intersection'
    Assert-True ($r.Instrument -contains 'Heavy' -and $r.Instrument.Count -eq 1) `
        'CarryOnly must not move displaced shards into the instrument set'

    if ($failures.Count -gt 0) {
        Write-Host 'Select-CoverageShards self-test FAILED:'
        foreach ($f in $failures) { Write-Host "  - $f" }
        exit 1
    }
    Write-Host 'Select-CoverageShards self-test passed.'
    exit 0
}

# ---------------------------------------------------------------- split

$map = $null
if ($PreviousMap -and (Test-Path -LiteralPath $PreviousMap)) {
    try {
        $map = Get-Content -LiteralPath $PreviousMap -Raw | ConvertFrom-Json
        foreach ($required in 'sha', 'knownShards', 'files') {
            if (-not $map.PSObject.Properties[$required]) { throw "no '$required' property" }
        }
    }
    catch {
        Write-Host "::warning::previous map unreadable ($($_.Exception.Message)) - instrumenting every shard"
        $map = $null
    }
}

$split = Split-CoverageWork -Map $map -ChangedFiles $ChangedFiles -AllShards $AllShards

$carriedFinal = @($split.Carried)
if ($PSBoundParameters.ContainsKey('CarryOnly')) {
    $allow = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($s in $CarryOnly) { if ($s) { [void] $allow.Add([string] $s) } }
    $carriedFinal = @($split.Carried | Where-Object { $allow.Contains([string] $_) })
}
$selfProducing = @($split.Carried).Count - $carriedFinal.Count
Write-Host "instrument $($split.Instrument.Count), carry forward $($carriedFinal.Count), self-producing $selfProducing  [$($split.Reason)]"

if ($CarryForwardDirectory -and $carriedFinal.Count -gt 0) {
    if (-not (Test-Path -LiteralPath $CarryForwardDirectory)) {
        New-Item -ItemType Directory -Path $CarryForwardDirectory -Force | Out-Null
    }
    foreach ($shard in $carriedFinal) {
        $slug = $shard -replace '[\\/:*?"<>|\s-]+', '_'
        $n = Export-ShardDigest -Map $map -Shard $shard -Path (Join-Path $CarryForwardDirectory "$slug.digest.json")
        Write-Verbose "carried $shard ($n file(s))"
    }
    Write-Host "wrote $($carriedFinal.Count) carried digest(s) to $CarryForwardDirectory"
}

if ($OutFile) {
    [pscustomobject]@{ instrument = @($split.Instrument); carried = @($carriedFinal) } |
        ConvertTo-Json -Depth 4 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8
}
