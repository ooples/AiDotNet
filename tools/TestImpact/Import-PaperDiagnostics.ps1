[CmdletBinding()]
param([Parameter(Mandatory)] [string] $SlicesJson, [Parameter(Mandatory)] [string] $ArtifactRoot,
    [Parameter(Mandatory)] [string] $Destination, [Parameter(Mandatory)] [string] $TestedSha,
    [Parameter(Mandatory)] [string] $OutputDirectory, [Parameter(Mandatory)] [string] $ClassListFile,
    [ValidateRange(1, 100000)] [int] $ChunkSize = 10)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/../PaperFidelity/PaperFidelity.ps1"
$slices = @($SlicesJson | ConvertFrom-Json)
if ($slices.Count -eq 0 -or @($slices | Select-Object -Unique).Count -ne $slices.Count) { throw 'Paper slice inventory is empty or duplicated.' }
$allClasses = @(Get-Content -LiteralPath $ClassListFile | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$expected = @()
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
foreach ($slice in $slices) {
    if ([string]$slice -notmatch '^\d+$') { throw 'Paper slice identity is not a nonnegative integer.' }
    $name = "Paper fidelity - slice $slice"; $slug = "Paper_fidelity_slice_$slice"; $expected += $name
    $source = Join-Path $ArtifactRoot "paper-fidelity-results-pr-$slice"
    # actions/download-artifact extracts a pattern's ONLY match straight into its path, with no directory named after
    # the artifact. A one-slice plan therefore lands in the root; its summary still has to name this slice below.
    if (-not (Test-Path -LiteralPath $source -PathType Container) -and $slices.Count -eq 1 -and
        (Test-Path -LiteralPath (Join-Path $ArtifactRoot 'slice-summary.json') -PathType Leaf)) { $source = $ArtifactRoot }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) { continue }
    $target = Join-Path $Destination "coverage-$TestedSha-$slug"
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    $classes = @(Select-PaperFidelitySlice -Classes $allClasses -Slice ([int]$slice) -SliceCount $slices.Count)
    $chunks = @()
    for ($index = 0; $index -lt $classes.Count; $index += $ChunkSize) {
        $chunks += [pscustomobject]@{ chunk = ('chunk{0:d3}' -f ($chunks.Count + 1)); classes = @($classes[$index..([Math]::Min($index + $ChunkSize, $classes.Count) - 1)]) }
    }
    $errors = [Collections.Generic.List[string]]::new()
    $summary = $null
    try {
        $summary = Get-Content (Join-Path $source 'slice-summary.json') -Raw -ErrorAction Stop | ConvertFrom-Json
        if ($summary.schemaVersion -ne 1 -or $summary.slice -ne $slice -or $summary.sliceCount -ne $slices.Count -or
            $summary.classes -ne $classes.Count -or @($summary.chunks).Count -ne $chunks.Count) { throw 'Paper summary does not match the planned slice/class/chunk inventory.' }
        for ($index = 0; $index -lt $chunks.Count; $index++) {
            $actual = $summary.chunks[$index]; $planned = $chunks[$index]
            if ($actual.chunk -cne $planned.chunk -or (ConvertTo-Json -InputObject @($actual.classes) -Compress) -cne
                (ConvertTo-Json -InputObject @($planned.classes) -Compress)) { throw 'Paper summary chunk identity/classes do not match the independent class plan.' }
            if ([string]$actual.exit -cnotin @('0', '1')) { $errors.Add("$($planned.chunk) did not complete normally: $($actual.exit).") }
        }
        Copy-Item -LiteralPath (Join-Path $source 'slice-summary.json') -Destination $target
    } catch { $errors.Add("Paper summary is missing or invalid: $($_.Exception.Message)") }
    $initialPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($chunk in $chunks) {
        $initial = Join-Path $source "$($chunk.chunk).trx"
        [void]$initialPaths.Add([IO.Path]::GetFullPath($initial))
        if (Test-Path -LiteralPath $initial -PathType Leaf) { Copy-Item -LiteralPath $initial -Destination $target }
        else { $errors.Add("Missing planned initial chunk $($chunk.chunk).trx.") }
    }
    $excluded = @(Get-ChildItem -LiteralPath $source -Recurse -Filter '*.trx' -File | Where-Object { -not $initialPaths.Contains($_.FullName) })
    if ($excluded.Count) { $errors.Add("Excluded $($excluded.Count) retry/duplicate/undeclared TRX file(s).") }
    [ordered]@{ shard = $name; slug = $slug; diagnosticOnly = $true; testStepOutcome = ''
        expectedInitialChunks = @($chunks.chunk); artifactIncomplete = $errors.Count -gt 0; diagnosticErrors = @($errors) } | ConvertTo-Json -Depth 10 |
        Set-Content (Join-Path $target 'shard-metadata.json') -Encoding utf8
}
$expected | ConvertTo-Json -AsArray | Set-Content (Join-Path $OutputDirectory 'expected-shards.json') -Encoding utf8
[ordered]@{ schemaVersion = 1; verified = $false; diagnosticOnly = $true; testedSha = $TestedSha
    reasons = @('Paper fidelity is a separate diagnostic inventory and cannot certify the standard shard baseline.') } |
    ConvertTo-Json | Set-Content (Join-Path $OutputDirectory 'evidence-provenance.json') -Encoding utf8
