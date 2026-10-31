<#
.SYNOPSIS
    Runs one slice of the paper-fidelity tests and leaves a machine-readable result for every case.

.DESCRIPTION
    The classes come from -ClassListFile (a pull request's reachable classes) or, without one, from discovering
    every class with a Category=PaperFidelity test (the full sweep). The slice owns every -SliceCount-th class of
    the sorted list, so the slices of one sweep cover each class exactly once.

    The slice's classes run in chunks of -ChunkSize, each chunk in its own test host under a wall-clock cap, the
    way heavy-timeout-nightly.yml runs its lane: a host's memory only grows across models, so a chunk bounds it,
    and a chunk that hangs or dies is reported on its own row instead of taking the slice with it.

    Results, all under -ResultsDirectory:
      paper-fidelity-<pid>.jsonl  one record per case, written by the tests themselves
                                  (tests/AiDotNet.Tests/ModelFamilyTests/Base/PaperFidelityResultLog.cs)
      chunkNNN.trx                the VSTest results of each chunk
      slice-summary.json          per chunk: classes, exit, seconds; and the slice's wall time

    tools/PaperFidelity/ConvertTo-RuntimeFidelityBaseline.ps1 reads these.

    Exits 1 when any chunk failed (a test failed, timed out or the host died), 0 otherwise.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [int] $Slice,
    [Parameter(Mandatory)] [int] $SliceCount,
    [string] $ClassListFile = '',
    [string] $ResultsDirectory = 'artifacts/paper-fidelity',
    [int] $ChunkSize = 10,
    [int] $ChunkCapMinutes = 90,
    # The slice's own budget, inside the job's timeout-minutes; a chunk that could not finish inside it is
    # recorded as not run rather than cancelled mid-way with no summary.
    [int] $BudgetMinutes = 330
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'PaperFidelity.ps1')

if ($ChunkSize -lt 1) { throw "ChunkSize must be positive, not $ChunkSize" }
$sliceStarted = Get-Date
$project = $script:PaperFidelityProject
$framework = $script:PaperFidelityFramework
$category = $script:PaperFidelityCategory
New-Item -ItemType Directory -Force -Path $ResultsDirectory | Out-Null
$resultsRoot = (Resolve-Path -LiteralPath $ResultsDirectory).Path

if ($ClassListFile) {
    $all = @(Get-Content -LiteralPath $ClassListFile | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    Write-Host "--- $($all.Count) class(es) from $ClassListFile ---"
}
else {
    Write-Host "--- discovering $category test classes ---"
    $all = @(Find-PaperFidelityClasses)
    Write-Host "--- $($all.Count) class(es) discovered ---"
}

$classes = @(Select-PaperFidelitySlice -Classes $all -Slice $Slice -SliceCount $SliceCount)
Write-Host "--- slice $Slice of ${SliceCount}: $($classes.Count) class(es) in chunks of $ChunkSize ---"

$memTotal = Select-String -Path '/proc/meminfo' -Pattern '^MemTotal:\s+(\d+) kB' -ErrorAction SilentlyContinue
if ($null -ne $memTotal) {
    # The managed heap is capped below the runner's memory so an exhausted chunk fails with an
    # OutOfMemoryException in the host instead of the kernel killing the runner (see heavy-timeout-nightly.yml).
    $env:DOTNET_GCHeapHardLimit = '0x{0:X}' -f [long]([long]$memTotal.Matches[0].Groups[1].Value * 1024 * 0.7)
}
$env:AIDOTNET_PAPER_FIDELITY_RESULTS = $resultsRoot

$rows = [Collections.Generic.List[object]]::new()
$failed = 0
for ($i = 0; $i -lt $classes.Count; $i += $ChunkSize) {
    $chunk = @($classes[$i..([Math]::Min($i + $ChunkSize, $classes.Count) - 1)])
    $label = 'chunk{0:d3}' -f ($rows.Count + 1)
    if (((Get-Date) - $sliceStarted).TotalMinutes + $ChunkCapMinutes -gt $BudgetMinutes) {
        $rows.Add([pscustomobject]@{ chunk = $label; classes = $chunk; exit = 'not run (budget)'; seconds = 0 })
        $failed++
        Write-Host "$label -> not run: the slice's $BudgetMinutes-minute budget is spent"
        continue
    }

    $filter = "Category=$category&(" + (($chunk | ForEach-Object { "FullyQualifiedName~$_." }) -join '|') + ')'
    $started = Get-Date
    Write-Host "::group::$label ($($chunk.Count)): $($chunk -join ', ')"
    & timeout --kill-after=60s "$($ChunkCapMinutes)m" `
        dotnet test $project -c Release --framework $framework --no-build --no-restore `
        --blame-hang --blame-hang-timeout 15m --blame-hang-dump-type none `
        --filter $filter `
        --logger "trx;LogFileName=$label.trx" `
        --logger 'console;verbosity=normal' `
        --results-directory $resultsRoot
    $exit = $LASTEXITCODE
    Write-Host '::endgroup::'
    # A filter that matches no test exits 0 too. Count the TRX's total (not 'executed': a chunk of skipped cases
    # legitimately executes none) and treat zero as a failure, so a renamed trait or class cannot pass silently.
    if ($exit -eq 0) {
        $trx = Join-Path $resultsRoot "$label.trx"
        $total = 0
        if (Test-Path $trx) {
            $counters = ([xml](Get-Content -Raw $trx)).TestRun.ResultSummary.Counters
            if ($counters) { $total = [int]$counters.total }
        }
        if ($total -eq 0) {
            Write-Host "::error::$label matched no paper-fidelity test (filter: $filter)"
            $exit = 3
        }
    }
    $seconds = [int]((Get-Date) - $started).TotalSeconds
    # 124: timeout stopped the chunk at its cap; 137: SIGKILL (the kill-after escalation or the OOM killer).
    $described = switch ($exit) { 0 { '0' } 3 { 'no tests matched' } 124 { "timeout after $ChunkCapMinutes min" } 137 { 'SIGKILL (timeout escalation or OOM)' } default { [string] $exit } }
    if ($exit -ne 0) { $failed++ }
    $rows.Add([pscustomobject]@{ chunk = $label; classes = $chunk; exit = $described; seconds = $seconds })
    Write-Host "$label -> exit $described in ${seconds}s"
}

$wall = [int]((Get-Date) - $sliceStarted).TotalSeconds
[pscustomobject]@{
    schemaVersion = 1
    slice = $Slice
    sliceCount = $SliceCount
    classes = $classes.Count
    wallSeconds = $wall
    failedChunks = $failed
    chunks = $rows.ToArray()
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultsRoot 'slice-summary.json') -Encoding utf8

if ($env:GITHUB_STEP_SUMMARY) {
    $summary = @("## Paper fidelity - slice $Slice of $SliceCount", '',
        "$($classes.Count) class(es), $($rows.Count) chunk(s), $failed failed, ${wall}s wall.", '',
        '| Chunk | Exit | Seconds | Classes |', '|---|---|---:|---|')
    foreach ($r in $rows) {
        $summary += "| $($r.chunk) | $($r.exit) | $($r.seconds) | $(@($r.classes | ForEach-Object { $_.Substring($_.LastIndexOf('.') + 1) }) -join ', ') |"
    }
    $summary -join "`n" | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8
}

Write-Host "=== slice ${Slice}: $($rows.Count) chunk(s), $failed failed, ${wall}s ==="
exit $(if ($failed -gt 0) { 1 } else { 0 })
