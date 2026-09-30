<#
.SYNOPSIS
    Fails when a test's verdict depends on the instruction set it ran under.

.DESCRIPTION
    The cross-ISA agreement lane runs the same tests once per ISA configuration (default,
    DOTNET_EnableAVX512=0, DOTNET_EnableAVX2=0), each in its own process, writing TRX files under
    <ResultsRoot>/<configuration>/. A model is correct or not independently of which SIMD kernels
    computed it; a test that passes under one configuration and fails under another is decided by
    the CPU, not by the model. RepViTSAM was one: ten AdamW steps failed on Intel with AVX-512 and
    passed with it disabled.

    Outcomes compared: Passed and Failed. A test skipped everywhere is ignored. A test that has a
    result in one configuration and none in another fails closed - a crashed or timed-out process
    must not read as agreement.

    Writes a markdown report to -SummaryFile when given, and exits 1 on any disagreement.
#>
[CmdletBinding()]
param(
    [string] $ResultsRoot,
    [string] $SummaryFile = '',
    [switch] $SelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Read-Verdicts([string] $Root) {
    $verdicts = @{}
    $configurations = @(Get-ChildItem -LiteralPath $Root -Directory | Sort-Object Name)
    foreach ($configuration in $configurations) {
        foreach ($trx in Get-ChildItem -LiteralPath $configuration.FullName -Filter '*.trx' -Recurse -File) {
            [xml] $doc = Get-Content -LiteralPath $trx.FullName -Raw
            foreach ($result in $doc.GetElementsByTagName('UnitTestResult')) {
                $name = [string] $result.GetAttribute('testName')
                $outcome = [string] $result.GetAttribute('outcome')
                if (-not $verdicts.ContainsKey($name)) { $verdicts[$name] = @{} }
                # A retried test can appear twice; any failure in a configuration is its verdict there.
                $previous = $verdicts[$name][$configuration.Name]
                if ($previous -ne 'Failed') { $verdicts[$name][$configuration.Name] = $outcome }
            }
        }
    }

    return [pscustomobject]@{ Configurations = @($configurations.Name); Verdicts = $verdicts }
}

function Find-Disagreements($Read) {
    $rows = [Collections.Generic.List[object]]::new()
    foreach ($name in ($Read.Verdicts.Keys | Sort-Object)) {
        $byConfiguration = $Read.Verdicts[$name]
        $decided = @($byConfiguration.Values | Where-Object { $_ -in @('Passed', 'Failed') } | Sort-Object -Unique)
        $missing = @($Read.Configurations | Where-Object { -not $byConfiguration.ContainsKey($_) })
        if ($decided.Count -gt 1 -or ($missing.Count -gt 0 -and $decided.Count -gt 0)) {
            $rows.Add([pscustomobject]@{
                Test = $name
                Outcomes = (@($Read.Configurations | ForEach-Object {
                    $o = if ($byConfiguration.ContainsKey($_)) { $byConfiguration[$_] } else { 'no result' }
                    "$($_)=$o"
                }) -join ', ')
            })
        }
    }

    return $rows.ToArray()
}

if ($SelfTest) {
    $root = Join-Path ([IO.Path]::GetTempPath()) "isa-verdicts-selftest-$([Guid]::NewGuid().ToString('N'))"
    function Write-Trx([string] $Configuration, [hashtable] $Outcomes) {
        $dir = Join-Path $root $Configuration
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $results = ($Outcomes.GetEnumerator() | ForEach-Object {
            "<UnitTestResult testName=`"$($_.Key)`" outcome=`"$($_.Value)`" />"
        }) -join "`n"
        Set-Content -LiteralPath (Join-Path $dir 'r.trx') -Encoding utf8 -Value "<TestRun><Results>$results</Results></TestRun>"
    }
    try {
        Write-Trx 'default' @{ 'A.Agrees' = 'Passed'; 'B.Flips' = 'Failed'; 'C.Crashed' = 'Passed'; 'D.Skipped' = 'NotExecuted' }
        Write-Trx 'avx512-off' @{ 'A.Agrees' = 'Passed'; 'B.Flips' = 'Passed'; 'D.Skipped' = 'NotExecuted' }
        Write-Trx 'avx2-off' @{ 'A.Agrees' = 'Passed'; 'B.Flips' = 'Passed'; 'C.Crashed' = 'Passed'; 'D.Skipped' = 'NotExecuted' }
        $found = @(Find-Disagreements (Read-Verdicts $root) | ForEach-Object Test)
        if (($found -join ',') -cne 'B.Flips,C.Crashed') {
            throw "expected B.Flips and C.Crashed to be flagged; got [$($found -join ', ')]"
        }
        Write-Host 'Compare-IsaVerdicts self-test passed (a flip and a missing result are flagged; agreement and skips are not).'
        exit 0
    }
    finally {
        Remove-Item -Recurse -Force -LiteralPath $root -ErrorAction SilentlyContinue
    }
}

if (-not $ResultsRoot) { throw '-ResultsRoot is required' }
$read = Read-Verdicts $ResultsRoot
if ($read.Configurations.Count -lt 2) { throw "need at least two configurations under $ResultsRoot; found $($read.Configurations.Count)" }
if ($read.Verdicts.Count -eq 0) { throw "no test results under $ResultsRoot - the lane would report agreement having compared nothing" }

$rows = @(Find-Disagreements $read)
$lines = @('### Cross-ISA verdict agreement', '',
    "$($read.Verdicts.Count) test(s) compared across $($read.Configurations -join ', ').", '')
if ($rows.Count -eq 0) {
    $lines += 'Every verdict agrees.'
}
else {
    $lines += "**$($rows.Count) test(s) whose verdict depends on the instruction set.** Measure the per-step loss " +
        'under each configuration; a start-up transient that outlasts the budget is declared with ' +
        '`MeasuredTransientRecoveryBudget`, and a kernel that computes differently is fixed at the kernel.'
    $lines += '', '| Test | Outcomes |', '|---|---|'
    $lines += @($rows | ForEach-Object { "| $($_.Test) | $($_.Outcomes) |" })
}

$report = $lines -join "`n"
Write-Host $report
if ($SummaryFile) { $report | Out-File -FilePath $SummaryFile -Append -Encoding utf8 }
exit $(if ($rows.Count -gt 0) { 1 } else { 0 })
