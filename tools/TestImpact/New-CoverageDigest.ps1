<#
.SYNOPSIS
    Reduces a shard's OpenCover XML to the executed source lines, per file.

.DESCRIPTION
    The map that drives test selection needs one fact per shard: which source lines that shard
    actually executed. The OpenCover XML carries that, but it is ~234 MB per shard - too large to
    keep, download and re-parse on every selection. This distils it to ~200 KB, measured 1111x
    smaller on Integration H-L, which is what makes a nightly map build cheap.

    Streamed with XmlReader on purpose. A DOM load of 234 MB is not viable on a 4-core runner, and
    the reduction is single-pass: every SequencePoint carries the fileid and line it belongs to, so
    nothing needs to be held except the accumulating line set.

    VISITED lines only. A file appearing in the <Files> table means it was COMPILED, not executed -
    every shard compiles the whole solution, so keying the map on that would map every file to every
    shard and select everything, every time. vc="0" is compiled-not-executed and is dropped.

.PARAMETER CoverageXml
    Path to coverage.opencover.xml.

.PARAMETER Shard
    The shard's display name, recorded in the digest so the map knows what to select.

.PARAMETER OutFile
    Where to write the digest JSON.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $CoverageXml,
    [Parameter(Mandatory)] [string] $Shard,
    [Parameter(Mandatory)] [string] $OutFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $CoverageXml)) {
    throw "Coverage file not found: $CoverageXml"
}

# uid -> repo-relative path, and uid -> sorted set of executed line numbers.
$paths = @{}
$lines = @{}

$settings = [System.Xml.XmlReaderSettings]::new()
$settings.IgnoreComments = $true
$settings.IgnoreWhitespace = $true
$settings.DtdProcessing = [System.Xml.DtdProcessing]::Prohibit

$reader = [System.Xml.XmlReader]::Create($CoverageXml, $settings)
try {
    while ($reader.Read()) {
        if ($reader.NodeType -ne [System.Xml.XmlNodeType]::Element) { continue }

        switch ($reader.Name) {
            'File' {
                $uid = $reader.GetAttribute('uid')
                $full = $reader.GetAttribute('fullPath')
                if ($uid -and $full) {
                    # Normalise to a repo-relative path so the map survives the runner's absolute
                    # workspace prefix, which differs between CI and any local rebuild.
                    # String.Replace, not -replace: the operator takes a REGEX and a lone
                    # backslash is not a valid pattern.
                    $norm = $full.Replace([char]92, [char]47)
                    $cut = -1
                    foreach ($anchor in @('/src/', '/tests/', '/tools/')) {
                        $i = $norm.LastIndexOf($anchor)
                        if ($i -ge 0 -and $i -gt $cut) { $cut = $i }
                    }
                    if ($cut -ge 0) { $paths[$uid] = $norm.Substring($cut + 1) }
                }
            }
            'SequencePoint' {
                # vc is the visit count. Zero means compiled but never executed by this shard.
                $vc = $reader.GetAttribute('vc')
                if ($vc -and $vc -ne '0') {
                    $fid = $reader.GetAttribute('fileid')
                    $sl = $reader.GetAttribute('sl')
                    if ($fid -and $sl) {
                        if (-not $lines.ContainsKey($fid)) {
                            $lines[$fid] = [System.Collections.Generic.HashSet[int]]::new()
                        }
                        [void] $lines[$fid].Add([int]$sl)
                    }
                }
            }
        }
    }
}
finally {
    $reader.Dispose()
}

# Collapse each file's executed lines into contiguous ranges. Ranges rather than a line list because
# executed code is overwhelmingly contiguous, and it keeps the digest small enough to commit.
$files = [ordered]@{}
foreach ($fid in $lines.Keys) {
    if (-not $paths.ContainsKey($fid)) { continue }
    $path = $paths[$fid]

    $sorted = [int[]] ($lines[$fid] | Sort-Object)
    $ranges = [System.Collections.Generic.List[object]]::new()
    $start = $sorted[0]
    $prev = $sorted[0]
    for ($i = 1; $i -lt $sorted.Length; $i++) {
        if ($sorted[$i] -eq $prev + 1) { $prev = $sorted[$i]; continue }
        [void] $ranges.Add(@($start, $prev))
        $start = $sorted[$i]; $prev = $sorted[$i]
    }
    [void] $ranges.Add(@($start, $prev))

    # A partial class compiles into several <File> entries under one path; merge rather than replace.
    if ($files.Contains($path)) {
        $merged = [System.Collections.Generic.List[object]] $files[$path]
        foreach ($r in $ranges) { [void] $merged.Add($r) }
        $files[$path] = $merged
    }
    else {
        $files[$path] = $ranges
    }
}

$digest = [ordered]@{
    schemaVersion = 1
    shard         = $Shard
    generatedUtc  = [DateTimeOffset]::UtcNow.ToString('o')
    fileCount     = $files.Count
    files         = $files
}

$dir = Split-Path -Parent $OutFile
if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
$digest | ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 -NoNewline

Write-Host "digest: $($files.Count) executed file(s) -> $OutFile"
