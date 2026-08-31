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

# A caller that passes "A,B" or "A;B" instead of @('A','B') would otherwise store one bogus shard
# name and silently drop the real always-run shards - under-selecting exactly the shards that cannot
# be mapped, which is the dangerous direction. Split defensively and drop blanks.
$AlwaysRun = @(
    $AlwaysRun |
        Where-Object { $_ } |
        ForEach-Object { $_ -split '[,;]' } |
        ForEach-Object { $_.Trim() } |
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
