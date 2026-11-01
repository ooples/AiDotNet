<#
.SYNOPSIS
  Rolls the AiDotNet public API contract forward after a release.

.DESCRIPTION
  src/AiDotNet.csproj tracks its public API with Microsoft.CodeAnalysis.PublicApiAnalyzers
  (PublicAPI.Shipped.txt / PublicAPI.Unshipped.txt, at the project root and per target framework under
  PublicAPI/<tfm>/) and validates the package against the last release (PackageValidationBaselineVersion).
  Once a version is published, the API it shipped has to become the new baseline:

  1. Every entry that PublicAPI.Unshipped.txt held AT THE RELEASE TAG moves into the matching
     PublicAPI.Shipped.txt. A '*REMOVED*<entry>' line removes <entry> from Shipped instead.
     Entries added to Unshipped on master after the tag stay in Unshipped: they have not shipped.
  2. PackageValidationBaselineVersion in src/AiDotNet.csproj becomes the released version.

  The suppressions in src/CompatibilitySuppressions.xml are NOT edited here: which of them are still needed
  against the new baseline is only known by running package validation. Regenerate them afterwards with a full
  build of src/AiDotNet.csproj and -p:GenerateCompatibilitySuppressionFile=true (the release workflow does).

  Entries are sorted the way the analyzer's code fix sorts them (ordinal ignoring case, then ordinal), and
  files are written with LF line endings and the '#nullable enable' header the analyzer reads.

.PARAMETER Version
  The version that was just released, e.g. 0.235.0.

.PARAMETER ReleasedRef
  The git ref of the release (default: v<Version>). Its PublicAPI.Unshipped.txt files define what shipped.

.PARAMETER RepoRoot
  The repository root (default: the parent of this script's directory).

.EXAMPLE
  pwsh scripts/Roll-PublicApiBaseline.ps1 -Version 0.235.0
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Version,
    [string]$ReleasedRef,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($Version -notmatch '^\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$') { throw "Version '$Version' is not a semantic version." }
if (-not $ReleasedRef) { $ReleasedRef = "v$Version" }

$projectDir = Join-Path $RepoRoot 'src'
$csproj = Join-Path $projectDir 'AiDotNet.csproj'
$header = '#nullable enable'
$removedPrefix = '*REMOVED*'

& git -C $RepoRoot rev-parse --verify --quiet "$ReleasedRef^{commit}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Release ref '$ReleasedRef' does not exist in $RepoRoot (fetch tags first)." }

# The analyzer's code fix orders entries with OrdinalIgnoreCase, breaking ties with Ordinal.
$comparison = [System.Comparison[string]] {
    param($x, $y)
    $c = [string]::Compare($x, $y, [StringComparison]::OrdinalIgnoreCase)
    if ($c -eq 0) { $c = [string]::CompareOrdinal($x, $y) }
    $c
}

function Get-ApiEntries([string[]]$lines) {
    # Everything except the header, blank lines and comments is an entry; keep them verbatim.
    @($lines | ForEach-Object { $_.TrimEnd("`r") } | Where-Object { $_ -and $_ -ne $header -and -not $_.StartsWith('//') })
}

function Read-ApiFile([string]$path) {
    if (Test-Path -LiteralPath $path) { Get-ApiEntries (Get-Content -LiteralPath $path) } else { @() }
}

function Write-ApiFile([string]$path, [string[]]$entries) {
    $list = [System.Collections.Generic.List[string]]::new()
    foreach ($e in ($entries | Select-Object -Unique)) { $list.Add($e) }
    $list.Sort($comparison)
    $text = (@($header) + $list) -join "`n"
    $dir = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    [System.IO.File]::WriteAllText($path, $text + "`n", [System.Text.UTF8Encoding]::new($false))
}

# Every Unshipped file that exists at the release tag or on the working tree.
$relativeUnshipped = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
foreach ($p in (& git -C $RepoRoot ls-tree -r --name-only $ReleasedRef -- src)) {
    if ($p -match '^src/(PublicAPI/[^/]+/)?PublicAPI\.Unshipped\.txt$') { [void]$relativeUnshipped.Add($p) }
}
if ($LASTEXITCODE -ne 0) { throw "git ls-tree failed for '$ReleasedRef'." }
Get-ChildItem -LiteralPath $projectDir -Recurse -Filter 'PublicAPI.Unshipped.txt' -File |
    ForEach-Object { [System.IO.Path]::GetRelativePath($RepoRoot, $_.FullName).Replace('\', '/') } |
    Where-Object { $_ -match '^src/(PublicAPI/[^/]+/)?PublicAPI\.Unshipped\.txt$' } |
    ForEach-Object { [void]$relativeUnshipped.Add($_) }

$moved = 0; $removed = 0
foreach ($rel in $relativeUnshipped) {
    $released = @(& git -C $RepoRoot show "${ReleasedRef}:$rel" 2>$null)
    if ($LASTEXITCODE -ne 0) { $released = @() }
    $released = @(Get-ApiEntries $released)
    if ($released.Count -eq 0) { continue }

    $unshippedPath = Join-Path $RepoRoot $rel
    $shippedPath = Join-Path (Split-Path -Parent $unshippedPath) 'PublicAPI.Shipped.txt'

    $shipped = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($e in @(Read-ApiFile $shippedPath)) { [void]$shipped.Add($e) }
    foreach ($e in $released) {
        if ($e.StartsWith($removedPrefix, [StringComparison]::Ordinal)) {
            $target = $e.Substring($removedPrefix.Length)
            if (-not $shipped.Remove($target)) { Write-Warning "$rel removes '$target', which $shippedPath does not list." }
            $removed++
        }
        else {
            [void]$shipped.Add($e)
            $moved++
        }
    }
    Write-ApiFile $shippedPath @($shipped)

    # Master may have gained entries since the tag; only the released ones leave Unshipped.
    $releasedSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$released, [StringComparer]::Ordinal)
    $remaining = @(Read-ApiFile $unshippedPath | Where-Object { -not $releasedSet.Contains($_) })
    if ($rel -eq 'src/PublicAPI.Unshipped.txt' -or $remaining.Count -gt 0) {
        Write-ApiFile $unshippedPath $remaining
    }
    elseif (Test-Path -LiteralPath $unshippedPath) {
        # A per-framework Unshipped file exists only while it has entries (see src/AiDotNet.csproj).
        Remove-Item -LiteralPath $unshippedPath
    }
}

# Point package validation at the release.
$xml = [System.IO.File]::ReadAllText($csproj)
$pattern = '<PackageValidationBaselineVersion>[^<]*</PackageValidationBaselineVersion>'
$found = [regex]::Matches($xml, $pattern).Count
if ($found -ne 1) { throw "Expected exactly one PackageValidationBaselineVersion in $csproj, found $found." }
$xml = [regex]::Replace($xml, $pattern, "<PackageValidationBaselineVersion>$Version</PackageValidationBaselineVersion>")
[System.IO.File]::WriteAllText($csproj, $xml, [System.Text.UTF8Encoding]::new($false))

Write-Host "Moved $moved entries into Shipped, applied $removed removals, baseline is now $Version (from $ReleasedRef)."
