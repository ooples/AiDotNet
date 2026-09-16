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

    Every shard receives exactly one typed disposition:

      Instrument          collect a fresh digest because its ranges may have moved, it has no
                          previous result, or it produces coverage naturally
      Carried             reconstruct its byte-identical digest from the previous map
      RunWithoutCoverage  execute all correctness tests, but do not instrument a known heavy or
                          timing shard that the previous map already classified as always-run

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

.PARAMETER NoCoverageShards
    Shards that do not collect coverage in an ordinary run because instrumentation exceeds their
    memory envelope or invalidates timing assertions. A mapped, unchanged member is carried; a
    dirty or previously unseen member is instrumented; and a member already in the map's alwaysRun
    set runs without coverage. Shards outside this list produce coverage naturally and therefore
    receive Instrument. Omit to treat every shard as eligible for carrying (the self-contained
    compatibility case).

.PARAMETER GlobalDirtyPrefixes
    Path prefixes whose changes invalidate EVERY carry (default: tests/). Coverage digests contain
    only product code - coverlet excludes the test assemblies - so a change that only touches test
    code is invisible to the per-shard dirty check, yet a new test can create coverage edges the
    map must learn. Reviewed and confirmed as a permanent silent-miss vector in enforce mode: a
    heavy shard whose suite gained a test covering src lines mapped to another shard would be
    re-carried nightly, and a PR touching those lines would skip it forever. When any changed path
    matches, nothing is carried and the night instruments everything, which re-measures all
    coverage edges including the new ones.

.PARAMETER CarryForwardDirectory
    Where to write reconstructed digests for the shards being carried forward. A digest is inverted
    straight out of the previous map, so New-ShardMap consumes it exactly like a fresh one and needs
    no knowledge of any of this.

.PARAMETER OutFile
    JSON: { instrument: [...], carried: [...], runWithoutCoverage: [...] }.

.PARAMETER SelfTest
    Runs the built-in checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $PreviousMap,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [AllowEmptyCollection()] [string[]] $ChangedFiles,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string[]] $AllShards,
    [Parameter(ParameterSetName = 'Select')] [AllowEmptyCollection()] [string[]] $NoCoverageShards,
    [Parameter(ParameterSetName = 'Select')] [string[]] $GlobalDirtyPrefixes = @('tests/'),
    [Parameter(ParameterSetName = 'Select')] [string] $CarryForwardDirectory,
    [Parameter(ParameterSetName = 'Select')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum CoverageDisposition {
    Instrument
    Carried
    RunWithoutCoverage
}

function Split-CoverageWork {
    <#
        Pure: given a parsed map, the changed paths and the shard manifest, decide what to
        instrument and what may be carried. Returns names only; writing digests is the caller's job.
    #>
    param(
        [Parameter(Mandatory)] [AllowNull()] $Map,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $ChangedFiles,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $AllShards,
        [AllowEmptyCollection()] [string[]] $NoCoverageShards,
        [switch] $DisableCarry
    )

    if ($null -eq $Map) {
        return [pscustomobject]@{
            Instrument        = @($AllShards)
            Carried           = @()
            RunWithoutCoverage = @()
            Reason            = 'no previous map'
        }
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
    $alwaysRun = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    if ($Map.PSObject.Properties['alwaysRun']) {
        foreach ($s in @($Map.alwaysRun)) { if ($s) { [void] $alwaysRun.Add([string] $s) } }
    }

    $noCoverage = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    if ($PSBoundParameters.ContainsKey('NoCoverageShards')) {
        foreach ($s in $NoCoverageShards) { if ($s) { [void] $noCoverage.Add([string] $s) } }
    }
    else {
        foreach ($s in $AllShards) { if ($s) { [void] $noCoverage.Add([string] $s) } }
    }

    $instrument = [System.Collections.Generic.List[string]]::new()
    $carried    = [System.Collections.Generic.List[string]]::new()
    $runWithoutCoverage = [System.Collections.Generic.List[string]]::new()
    foreach ($s in $AllShards) {
        $name = [string] $s
        $disposition = [CoverageDisposition]::Instrument

        if ($noCoverage.Contains($name) -and $alwaysRun.Contains($name)) {
            # The preceding complete map run already proved this no-coverage shard could not
            # produce a digest. Repeating the same memory-hungry instrumentation cannot teach the
            # map anything: execute its complete correctness suite and retain alwaysRun instead.
            $disposition = [CoverageDisposition]::RunWithoutCoverage
        }
        elseif ($noCoverage.Contains($name) -and -not $DisableCarry -and
                $mapped.Contains($name) -and -not $dirty.Contains($name)) {
            $disposition = [CoverageDisposition]::Carried
        }

        switch ($disposition) {
            ([CoverageDisposition]::Carried) { [void] $carried.Add($name); break }
            ([CoverageDisposition]::RunWithoutCoverage) {
                [void] $runWithoutCoverage.Add($name)
                break
            }
            default { [void] $instrument.Add($name) }
        }
    }

    return [pscustomobject]@{
        Instrument         = @($instrument)
        Carried            = @($carried)
        RunWithoutCoverage = @($runWithoutCoverage)
        Reason             = "$($changed.Count) changed file(s)"
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
            # Merged, not overwritten. The current builder cannot emit two occurrences of one
            # shard for one path, but this reads HISTORICAL maps, and silently dropping ranges
            # is the one corruption a carried digest must never introduce.
            if ($files.Contains($prop.Name)) { $files[$prop.Name] = @($files[$prop.Name]) + @($occurrence.r) }
            else { $files[$prop.Name] = @($occurrence.r) }
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

    # 1. Nothing changed: mapped no-coverage shards are carried, while an explicitly known
    #    always-run shard executes without the instrumentation that already failed to map it.
    $r = Split-CoverageWork -Map $map -ChangedFiles @() -AllShards $all
    Assert-True ($r.Carried.Count -eq 2) 'with no changes both mapped shards are carried'
    Assert-True ($r.RunWithoutCoverage -contains 'Heavy') `
        'a known always-run no-coverage shard must execute without instrumentation'
    Assert-True ($r.Instrument.Count -eq 0) 'the three dispositions must not overlap'
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
    Assert-True ($r.Instrument.Count -eq 3 -and $r.Carried.Count -eq 0 -and
        $r.RunWithoutCoverage.Count -eq 0) 'no map means instrument everything'

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

    # 8. The no-coverage boundary creates a complete, disjoint three-way partition. A mapped shard
    #    outside the boundary is a natural coverage producer, so it must not be carried.
    $r = Split-CoverageWork -Map $map -ChangedFiles @() -AllShards $all `
        -NoCoverageShards @('Alpha', 'Heavy')
    Assert-True ($r.Carried.Count -eq 1 -and $r.Carried -contains 'Alpha') `
        'the mapped no-coverage shard must be carried'
    Assert-True ($r.Instrument.Count -eq 1 -and $r.Instrument -contains 'Beta') `
        'a natural coverage producer must receive Instrument'
    Assert-True ($r.RunWithoutCoverage.Count -eq 1 -and $r.RunWithoutCoverage -contains 'Heavy') `
        'the prior always-run no-coverage shard must receive RunWithoutCoverage'
    Assert-True (($r.Instrument.Count + $r.Carried.Count + $r.RunWithoutCoverage.Count) -eq $all.Count) `
        'the typed coverage dispositions must cover every shard exactly once'

    # 9. alwaysRun alone is not permission to suppress instrumentation. The shard must also be in
    #    the explicit workflow-derived no-coverage boundary; otherwise it remains Instrument and
    #    gets another opportunity to produce a digest.
    $r = Split-CoverageWork -Map $map -ChangedFiles @() -AllShards $all `
        -NoCoverageShards @('Alpha')
    Assert-True ($r.Instrument -contains 'Heavy' -and
        -not ($r.RunWithoutCoverage -contains 'Heavy')) `
        'an always-run shard outside the no-coverage boundary must remain Instrument'

    # 10. A global-dirty tests/** change invalidates carries but cannot make a known memory-bound
    #    always-run shard safely instrumentable. The former becomes Instrument; the latter retains
    #    RunWithoutCoverage and therefore stays permanently selected.
    $r = Split-CoverageWork -Map $map -ChangedFiles @('tests/NewTest.cs') -AllShards $all `
        -NoCoverageShards @('Alpha', 'Heavy') -DisableCarry
    Assert-True ($r.Instrument -contains 'Alpha' -and $r.Instrument -contains 'Beta') `
        'global dirty must turn every coverage-capable shard into Instrument'
    Assert-True ($r.RunWithoutCoverage -contains 'Heavy') `
        'global dirty must not retry instrumentation already known to exceed the runner envelope'

    # 11. A historical map holding two occurrences of one shard for one path must MERGE their
    #    ranges into the carried digest, never overwrite - dropped ranges would make the map
    #    claim the shard does not execute lines it does, which is a silent selection miss.
    $dupMap = @{
        schemaVersion = 1
        sha           = 'abc123'
        generatedUtc  = '2026-09-01T00:00:00Z'
        knownShards   = @('Alpha')
        alwaysRun     = @()
        files         = @{ 'src/Dup.cs' = @(@{ s = 0; r = @(1, 2) }, @{ s = 0; r = @(9, 9) }) }
    } | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $tmp2 = Join-Path ([System.IO.Path]::GetTempPath()) "carry-dup-$PID.json"
    [void] (Export-ShardDigest -Map $dupMap -Shard 'Alpha' -Path $tmp2)
    $d2 = Get-Content -LiteralPath $tmp2 -Raw | ConvertFrom-Json
    Remove-Item -LiteralPath $tmp2 -ErrorAction SilentlyContinue
    Assert-True ((@($d2.files.'src/Dup.cs') -join ',') -eq '1,2,9,9') `
        'duplicate occurrences must merge ranges, not overwrite'

    # 12. TEST-CODE changes must invalidate every carry. Digests hold only product code, so a
    #     tests/** change is invisible to the per-shard dirty check - yet a new test can create
    #     coverage edges the map must learn, and re-carrying would hide them from enforce-mode
    #     selection permanently. The pure split cannot see this; the caller-level rule must.
    #     (Exercised via the CLI path in the workflow; here the rule's building blocks:)
    $r = Split-CoverageWork -Map $map -ChangedFiles @('tests/AiDotNet.Tests/NewTest.cs') -AllShards $all
    Assert-True ($r.Carried.Count -eq 2) 'the pure split alone does NOT see test files (by design)'
    $hit = $false
    foreach ($p in @('tests/')) {
        if ('tests/AiDotNet.Tests/NewTest.cs'.StartsWith($p, [StringComparison]::OrdinalIgnoreCase)) { $hit = $true }
    }
    Assert-True $hit 'the global-dirty prefix rule must catch the same path the split misses'

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
        # An unknown schema is treated as unreadable, not as best-effort: carrying is the one
        # operation here where a misread map produces a STALE map rather than an escalation, so
        # any doubt must land on instrument-everything.
        if (-not $map.PSObject.Properties['schemaVersion'] -or [int] $map.schemaVersion -ne 1) {
            throw "unsupported schemaVersion '$($map.schemaVersion)'"
        }
        if ($map.files -isnot [pscustomobject]) { throw 'files must be an object' }
    }
    catch {
        Write-Host "::warning::previous map unreadable ($($_.Exception.Message)) - instrumenting every shard"
        $map = $null
    }
}

$disableCarry = $false
foreach ($path in $ChangedFiles) {
    if (-not $path) { continue }
    foreach ($prefix in $GlobalDirtyPrefixes) {
        if (([string] $path).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            Write-Host "change under '$prefix' ($path) - test code is invisible to digests, so nothing is carried tonight"
            $disableCarry = $true
            break
        }
    }
    if ($disableCarry) { break }
}

$splitArgs = @{
    Map          = $map
    ChangedFiles = $ChangedFiles
    AllShards    = $AllShards
    DisableCarry = $disableCarry
}
if ($PSBoundParameters.ContainsKey('NoCoverageShards')) {
    $splitArgs.NoCoverageShards = $NoCoverageShards
}
$split = Split-CoverageWork @splitArgs

$carriedFinal = @($split.Carried)
$runWithoutCoverage = @($split.RunWithoutCoverage)
Write-Host "instrument $($split.Instrument.Count), carry forward $($carriedFinal.Count), run without coverage $($runWithoutCoverage.Count)  [$($split.Reason)]"

if ($CarryForwardDirectory -and $carriedFinal.Count -gt 0) {
    # .NET call rather than New-Item: New-Item has no -LiteralPath, and its -Path is
    # wildcard-expanded, so a directory name containing [ or ] would be created somewhere else.
    # Resolve through PowerShell first: Directory.CreateDirectory resolves a relative path from
    # Environment.CurrentDirectory, which does NOT follow Push-Location. The generated-repository
    # proof deliberately changes the PowerShell location and caught the resulting split-brain path.
    $resolvedCarryDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
        $CarryForwardDirectory)
    [void] [System.IO.Directory]::CreateDirectory($resolvedCarryDirectory)
    foreach ($shard in $carriedFinal) {
        $slug = $shard -replace '[\\/:*?"<>|\s-]+', '_'
        $n = Export-ShardDigest -Map $map -Shard $shard -Path (Join-Path $resolvedCarryDirectory "$slug.digest.json")
        Write-Verbose "carried $shard ($n file(s))"
    }
    Write-Host "wrote $($carriedFinal.Count) carried digest(s) to $CarryForwardDirectory"
}

if ($OutFile) {
    [pscustomobject]@{
        schemaVersion      = 1
        instrument         = @($split.Instrument)
        carried            = @($carriedFinal)
        runWithoutCoverage = @($runWithoutCoverage)
    } |
        ConvertTo-Json -Depth 4 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8
}
