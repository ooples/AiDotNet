[CmdletBinding()]
<#
.SYNOPSIS
Reports how much of the shard matrix a certified map can actually select, by family.

.DESCRIPTION
THE NUMBER NOBODY WAS LOOKING AT. Selection can only exclude a shard the map knows about. A shard
absent from `knownShards` is added to `alwaysRun` and therefore runs on every pull request that needs
validation, forever, silently -- there is no warning, no failed check, and the job counts look like a
selector making choices.

Measured on the certified map from run 35047041346: 113 of 164 shards were mapped. The 51 that were
not are the entire Conformance family (35) and the entire Sweep family (13), plus the three
Integration shards that are always-run by design. That is a 29% floor under every run, and it is why
a one-line comment change to one activation function still ran 49 shards.

Those two families are not an oversight in the map builder. Their work happens in a child worker
process (see `coverageIncludeDirectory` in .github/test-shards.yml), and the builder deliberately
keeps such a shard always-run until its digest is shown to have reached the model code -- a coverage
gap costs runs rather than risking a miss. The gap is the right default and the wrong resting place,
so this makes it visible and measurable instead of leaving it to be rediscovered.

.PARAMETER MapFile
A certified or candidate shard-map.json.

.PARAMETER ManifestFile
.github/test-shards.yml. Defaults to the one beside this script's repository root.

.PARAMETER FailOnEmptyFamily
Exit non-zero when an entire family is unmapped. Off by default so the report can be read on a
machine where the floor is still expected; the CI gate turns it on once a family is fixed, which is
what stops it regressing.
#>
param(
    [Parameter(Mandatory)] [string] $MapFile,
    [string] $ManifestFile,
    [switch] $FailOnEmptyFamily
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $ManifestFile) {
    $ManifestFile = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) '.github/test-shards.yml'
}
foreach ($file in @($MapFile, $ManifestFile)) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Not found: $file" }
}

$map = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
$known = [Collections.Generic.HashSet[string]]::new([string[]] @($map.knownShards), [StringComparer]::Ordinal)

# PARSED, NOT PATTERN-MATCHED. This first used a single '^\s*-\s*name:' regex on the grounds that it
# would run anywhere PowerShell does. That was the wrong trade: the pattern silently skips flow-style
# entries -- "- { name: 'Parameter sweep - Count 0/8', ... }" -- and the manifest carried 45 of them,
# so the universe read 164 where it was 209. An undercounted universe makes coverage look BETTER than
# it is and can hide an entire unmapped family, which is precisely what this script exists to expose.
#
# yq is how the selector and the map workflow read this file, so it is the authority here too; the
# text fallback is for a developer machine without it and handles both styles.
$raw = Get-Content -LiteralPath $ManifestFile -Raw
$names = @()
if (Get-Command yq -CommandType Application -ErrorAction SilentlyContinue) {
    $names = @(& yq -o=json -I=0 '.shard' $ManifestFile | ConvertFrom-Json | ForEach-Object { $_.name })
    if ($LASTEXITCODE -ne 0) { throw "yq failed to read $ManifestFile" }
}
if ($names.Count -eq 0) {
    $block = [regex]::Matches($raw, '(?m)^\s*-\s*name:\s*(?<v>.+?)\s*$')
    $flow = [regex]::Matches($raw, '(?m)^\s*-\s*\{\s*name:\s*(?<v>''[^'']*''|"[^"]*"|[^,}]+)')
    $names = @(@($block) + @($flow) | ForEach-Object { $_.Groups['v'].Value.Trim().Trim("'", '"') })
}
if ($names.Count -eq 0) { throw "No shard names found in $ManifestFile" }

# THE TRIP-WIRE. Whatever did the parsing, the number of names must equal the number of list items
# under 'shard:'. Without this a parser that quietly skips a style reports a rosy number and nothing
# ever contradicts it -- the failure mode that produced the 164/209 undercount in the first place.
$items = ([regex]::Matches($raw, '(?m)^\s{2}-\s')).Count
if ($names.Count -ne $items) {
    throw ("parsed {0} shard names but {1} holds {2} list items; the parser is missing entries" -f `
        $names.Count, $ManifestFile, $items)
}

# Family is the part before the first ' - ', which is how these names are constructed.
$rows = $names | ForEach-Object {
    [pscustomobject]@{
        Name   = $_
        Family = ($_ -split ' - ')[0].Trim()
        Mapped = $known.Contains($_)
    }
}

$families = $rows | Group-Object Family | ForEach-Object {
    $mapped = @($_.Group | Where-Object Mapped).Count
    [pscustomobject]@{
        Family    = $_.Name
        Total     = $_.Count
        Mapped    = $mapped
        Unmapped  = $_.Count - $mapped
        Selectable = if ($_.Count) { [math]::Round(100 * $mapped / $_.Count, 1) } else { 0 }
    }
} | Sort-Object -Property @{ Expression = 'Unmapped'; Descending = $true }, @{ Expression = 'Family'; Descending = $false }

$totalShards = $rows.Count
$totalMapped = @($rows | Where-Object Mapped).Count
$floor = $totalShards - $totalMapped

Write-Host ''
Write-Host ("Shard map coverage: {0} of {1} shards selectable ({2}%)" -f `
    $totalMapped, $totalShards, [math]::Round(100 * $totalMapped / $totalShards, 1))
Write-Host ("Always-run floor:   {0} shards ({1}% of every validated run)" -f `
    $floor, [math]::Round(100 * $floor / $totalShards, 1))
Write-Host ''
$families | Format-Table Family, Total, Mapped, Unmapped, @{ n = 'Selectable %'; e = { $_.Selectable } } -AutoSize | Out-Host

$empty = @($families | Where-Object { $_.Mapped -eq 0 -and $_.Total -gt 0 })
foreach ($family in $empty) {
    Write-Host ("::warning::the entire '{0}' family ({1} shards) is absent from the map, so it runs on every validated pull request" -f `
        $family.Family, $family.Total)
}

if ($FailOnEmptyFamily -and $empty.Count -gt 0) {
    Write-Error ("{0} shard famil{1} entirely unmapped: {2}" -f `
        $empty.Count, $(if ($empty.Count -eq 1) { 'y is' } else { 'ies are' }), (($empty.Family) -join ', '))
    exit 1
}
exit 0
