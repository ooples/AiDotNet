<#
.SYNOPSIS
    Merges per-shard coverage digests into the single map that drives test selection.

.DESCRIPTION
    Inverts "shard -> executed lines" into "file -> the shards that execute it, and where". Selection
    then reduces to intersecting a PR's changed line ranges against that index.

    The map also records what CANNOT be selected against, which matters as much as what can:

      alwaysRun   Shards that produce no coverage and therefore can never appear in the index.
                  $collectCoverage in the workflow is `-not ($heavyShard -or $measuresTiming)`, so
                  every heavy and timing shard emits nothing. Left implicit, the map would silently
                  under-select exactly the shards that have OOM-killed runners before. They are
                  named here so the selector always includes them.

      knownShards Every shard the map was built from. A shard absent from BOTH this list and
                  alwaysRun is unknown to the map, which is a stale-map signal the selector escalates
                  on rather than quietly skipping.

    SCALE, measured on a synthetic full-size set (106 digests, ~11 MB, built from a real shard's
    file list so the shapes are realistic):

        map build      6s      ->  9.4 MB
        selection      2.8s        including parsing that map

    Both are negligible against the ~2h matrix they decide, so neither is worth optimising. The
    synthetic set deliberately gives every shard an overlapping subset of ONE file list, so its
    selection RATE means nothing - only the timings transfer.

.PARAMETER DigestDirectory
    Directory holding *.digest.json files, one per shard.

.PARAMETER AlwaysRun
    Shards that never produce coverage and must always run.

.PARAMETER Sha
    The commit the digests were produced from. Selection refuses a map built from an unrelated tree.

.PARAMETER OutFile
    Where to write shard-map.json.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $DigestDirectory,
    [string[]] $AlwaysRun = @(),
    [Parameter(Mandatory)] [string] $Sha,
    [Parameter(Mandatory)] [string] $OutFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# NOT split on commas. An earlier revision did, to guard against a caller passing "A,B" instead of
# @('A','B') - and that guard corrupted real data, because shard names legitimately contain commas:
#
#   ModelFamily - Generated Layers P A,C,I,L,N,P        -> 6 fragments
#   ModelFamily - Generated Layers P E,H,O,R,S,U,W,Y    -> 8 fragments
#
# Measured on the first real map: 91 always-run shards became 103 entries, none of the fragments
# matching a real shard. The selector then finds a selected name absent from the shard list and
# escalates - every run, silently, with the feature switched off and the logs looking healthy.
#
# The caller passes a real array; blanks are dropped and entries are trimmed, nothing more.
$AlwaysRun = @(
    $AlwaysRun |
        Where-Object { $_ } |
        ForEach-Object { ([string] $_).Trim() } |
        Where-Object { $_ }
)

$digestFiles = @(Get-ChildItem -LiteralPath $DigestDirectory -Filter '*.digest.json' -Recurse -File)
if ($digestFiles.Count -eq 0) {
    throw "No *.digest.json under $DigestDirectory - refusing to write an empty map, which would select nothing."
}

$shards = [System.Collections.Generic.List[string]]::new()
$index = @{}   # path -> list of @{ s = shardIndex; r = ranges }

foreach ($file in $digestFiles) {
    $digest = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    $name = [string] $digest.shard
    if ([string]::IsNullOrWhiteSpace($name)) { throw "Digest $($file.Name) has no shard name." }

    $shardIndex = $shards.Count
    [void] $shards.Add($name)

    foreach ($property in $digest.files.PSObject.Properties) {
        $path = $property.Name
        if (-not $index.ContainsKey($path)) {
            $index[$path] = [System.Collections.Generic.List[object]]::new()
        }
        [void] $index[$path].Add([ordered]@{ s = $shardIndex; r = $property.Value })
    }
}

$files = [ordered]@{}
foreach ($path in ($index.Keys | Sort-Object)) { $files[$path] = $index[$path] }

$map = [ordered]@{
    schemaVersion = 1
    sha           = $Sha
    generatedUtc  = [DateTimeOffset]::UtcNow.ToString('o')
    knownShards   = $shards
    alwaysRun     = $AlwaysRun
    fileCount     = $files.Count
    files         = $files
}

$dir = Split-Path -Parent $OutFile
if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
$map | ConvertTo-Json -Depth 8 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 -NoNewline

Write-Host "map: $($shards.Count) shard(s), $($files.Count) file(s), $($AlwaysRun.Count) always-run -> $OutFile"
