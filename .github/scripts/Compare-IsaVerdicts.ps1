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

    Coverage is checked against what was EXPECTED, not only against what was observed: every
    expected configuration must have produced results, and every test in -ExpectedTestsFile must
    have a result in every configuration. A test is compared only when every configuration reports
    Passed or Failed; a test skipped in every configuration is ignored. Anything else - a missing
    result, a skip in some configurations, an Error/Timeout/Aborted outcome - is INCOMPLETE and fails,
    reported separately from a verdict disagreement, because a crashed or half-run process must not
    read as agreement.

    Writes a markdown report to -SummaryFile when given, and exits 1 on any disagreement or
    incomplete result.
#>
[CmdletBinding()]
param(
    [string] $ResultsRoot,
    # The configuration directories that must exist under -ResultsRoot.
    [string[]] $ExpectedConfigurations = @(),
    # Optional: one fully qualified test name per line that every configuration must report.
    [string] $ExpectedTestsFile = '',
    [string] $SummaryFile = '',
    [switch] $SelfTest
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# A test reported more than once in one configuration (a retry, or a second results file) keeps its most severe
# outcome: anything incomplete (Error, Timeout, Aborted, ...) over Failed, Failed over Passed, Passed over NotExecuted.
# Incomplete must win in both orders: Error then Passed would otherwise read as agreement, and Failed then Error would
# hide that the run never finished.
function Get-OutcomeSeverity([string] $Outcome) {
    switch ($Outcome) {
        'NotExecuted' { return 0 }
        'Passed' { return 1 }
        'Failed' { return 2 }
        default { return 3 }
    }
}

function Read-Verdicts([string] $Root, [string[]] $Configurations) {
    $verdicts = @{}
    $configurationsWithResults = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($configuration in $Configurations) {
        $dir = Join-Path $Root $configuration
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { continue }
        foreach ($trx in Get-ChildItem -LiteralPath $dir -Filter '*.trx' -Recurse -File) {
            [xml] $doc = Get-Content -LiteralPath $trx.FullName -Raw
            foreach ($result in $doc.GetElementsByTagName('UnitTestResult')) {
                $name = [string] $result.GetAttribute('testName')
                $outcome = [string] $result.GetAttribute('outcome')
                [void] $configurationsWithResults.Add($configuration)
                if (-not $verdicts.ContainsKey($name)) { $verdicts[$name] = @{} }
                # A retried test can appear twice; the most severe outcome is its verdict there (Get-OutcomeSeverity).
                $previous = $verdicts[$name][$configuration]
                if ($null -eq $previous -or (Get-OutcomeSeverity $outcome) -gt (Get-OutcomeSeverity $previous)) {
                    $verdicts[$name][$configuration] = $outcome
                }
            }
        }
    }

    return [pscustomobject]@{ Verdicts = $verdicts; ConfigurationsWithResults = $configurationsWithResults }
}

function Compare-Verdicts([object] $Read, [string[]] $Configurations, [string[]] $ExpectedTests) {
    $disagreements = [Collections.Generic.List[object]]::new()
    $incomplete = [Collections.Generic.List[object]]::new()
    foreach ($configuration in $Configurations) {
        if (-not $Read.ConfigurationsWithResults.Contains($configuration)) {
            $incomplete.Add([pscustomobject]@{ Test = "(configuration $configuration)"; Outcomes = 'produced no results' })
        }
    }

    $names = [Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in $Read.Verdicts.Keys) { [void] $names.Add($name) }
    foreach ($name in $ExpectedTests) { [void] $names.Add($name) }
    foreach ($name in $names) {
        $byConfiguration = if ($Read.Verdicts.ContainsKey($name)) { $Read.Verdicts[$name] } else { @{} }
        $outcomes = @($Configurations | ForEach-Object {
            if ($byConfiguration.ContainsKey($_)) { $byConfiguration[$_] } else { 'no result' }
        })
        $row = [pscustomobject]@{
            Test = $name
            Outcomes = (@(for ($i = 0; $i -lt $Configurations.Count; $i++) { "$($Configurations[$i])=$($outcomes[$i])" }) -join ', ')
        }
        if (@($outcomes | Where-Object { $_ -ne 'NotExecuted' }).Count -eq 0) { continue }   # skipped everywhere
        if (@($outcomes | Where-Object { $_ -notin @('Passed', 'Failed') }).Count -gt 0) { $incomplete.Add($row); continue }
        if (@($outcomes | Sort-Object -Unique).Count -gt 1) { $disagreements.Add($row) }
    }

    return [pscustomobject]@{ Disagreements = $disagreements.ToArray(); Incomplete = $incomplete.ToArray() }
}

if ($SelfTest) {
    $root = Join-Path ([IO.Path]::GetTempPath()) "isa-verdicts-selftest-$([Guid]::NewGuid().ToString('N'))"
    function Write-Trx([string] $Base, [string] $Configuration, [hashtable] $Outcomes, [string] $File = 'r.trx') {
        $dir = Join-Path $Base $Configuration
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $results = ($Outcomes.GetEnumerator() | ForEach-Object {
            "<UnitTestResult testName=`"$($_.Key)`" outcome=`"$($_.Value)`" />"
        }) -join "`n"
        Set-Content -LiteralPath (Join-Path $dir $File) -Encoding utf8 -Value "<TestRun><Results>$results</Results></TestRun>"
    }
    $configs = @('default', 'avx512-off', 'avx2-off')
    $failures = [Collections.Generic.List[string]]::new()
    function Check([string] $Name, [object] $Result, [string[]] $Disagree, [string[]] $Incomplete) {
        $d = @($Result.Disagreements | ForEach-Object Test) -join ','
        $n = @($Result.Incomplete | ForEach-Object Test) -join ','
        if ($d -cne ($Disagree -join ',') -or $n -cne ($Incomplete -join ',')) {
            $failures.Add("$($Name): disagreements [$d] expected [$($Disagree -join ',')]; incomplete [$n] expected [$($Incomplete -join ',')]")
        }
    }
    try {
        $a = Join-Path $root 'a'
        Write-Trx $a 'default' @{ 'A.Agrees' = 'Passed'; 'B.Flips' = 'Failed'; 'C.Crashed' = 'Passed'; 'D.Skipped' = 'NotExecuted'; 'E.PartSkip' = 'Passed'; 'F.Errored' = 'Passed' }
        Write-Trx $a 'avx512-off' @{ 'A.Agrees' = 'Passed'; 'B.Flips' = 'Passed'; 'D.Skipped' = 'NotExecuted'; 'E.PartSkip' = 'NotExecuted'; 'F.Errored' = 'Error' }
        Write-Trx $a 'avx2-off' @{ 'A.Agrees' = 'Passed'; 'B.Flips' = 'Passed'; 'C.Crashed' = 'Passed'; 'D.Skipped' = 'NotExecuted'; 'E.PartSkip' = 'Passed'; 'F.Errored' = 'Passed' }
        Check 'outcomes' (Compare-Verdicts (Read-Verdicts $a $configs) $configs @('G.NeverRan')) `
            @('B.Flips') @('C.Crashed', 'E.PartSkip', 'F.Errored', 'G.NeverRan')

        $b = Join-Path $root 'b'
        Write-Trx $b 'default' @{ 'A.Agrees' = 'Passed' }
        Write-Trx $b 'avx2-off' @{ 'A.Agrees' = 'Passed' }
        Check 'missing configuration' (Compare-Verdicts (Read-Verdicts $b $configs) $configs @()) `
            @() @('(configuration avx512-off)', 'A.Agrees')

        # A test reported twice in one configuration: an incomplete outcome must survive in either order.
        $c = Join-Path $root 'c'
        foreach ($config in $configs) { Write-Trx $c $config @{ 'H.ErrorThenPass' = 'Passed'; 'I.FailThenError' = 'Failed'; 'J.RetriedPass' = 'Passed' } }
        Write-Trx $c 'default' @{ 'H.ErrorThenPass' = 'Error'; 'I.FailThenError' = 'Failed'; 'J.RetriedPass' = 'Passed' } 'a-first.trx'
        Write-Trx $c 'default' @{ 'H.ErrorThenPass' = 'Passed'; 'I.FailThenError' = 'Error'; 'J.RetriedPass' = 'Passed' } 'z-second.trx'
        Check 'repeated results' (Compare-Verdicts (Read-Verdicts $c $configs) $configs @()) `
            @() @('H.ErrorThenPass', 'I.FailThenError')

        if ($failures.Count -gt 0) { $failures | ForEach-Object { Write-Host "FAIL: $_" }; exit 1 }
        Write-Host 'Compare-IsaVerdicts self-test passed (flip, missing result, partial skip, execution error, never-run test, missing configuration and an incomplete repeat in either order are all caught; agreement and a skip everywhere are not).'
        exit 0
    }
    finally {
        Remove-Item -Recurse -Force -LiteralPath $root -ErrorAction SilentlyContinue
    }
}

if (-not $ResultsRoot) { throw '-ResultsRoot is required' }
$configurations = if ($ExpectedConfigurations.Count -gt 0) { $ExpectedConfigurations }
                  else { @(Get-ChildItem -LiteralPath $ResultsRoot -Directory | Sort-Object Name | ForEach-Object Name) }
if ($configurations.Count -lt 2) { throw "need at least two configurations; got $($configurations.Count)" }
$expectedTests = if ($ExpectedTestsFile) { @(Get-Content -LiteralPath $ExpectedTestsFile | Where-Object { $_.Trim() }) } else { @() }

$read = Read-Verdicts $ResultsRoot $configurations
if ($read.Verdicts.Count -eq 0 -and $expectedTests.Count -eq 0) {
    throw "no test results under $ResultsRoot - the lane would report agreement having compared nothing"
}

$result = Compare-Verdicts $read $configurations $expectedTests
$compared = @($read.Verdicts.Keys) + $expectedTests | Sort-Object -Unique
$lines = @('### Cross-ISA verdict agreement', '',
    "$(@($compared).Count) test(s) across $($configurations -join ', ').", '')
if ($result.Disagreements.Count -eq 0 -and $result.Incomplete.Count -eq 0) {
    $lines += 'Every verdict agrees, and every expected test ran under every configuration.'
}
if ($result.Disagreements.Count -gt 0) {
    $lines += "**$($result.Disagreements.Count) test(s) whose verdict depends on the instruction set.** Measure the " +
        'per-step loss under each configuration; a start-up transient that outlasts the budget is declared with ' +
        '`MeasuredTransientRecoveryBudget`, and a kernel that computes differently is fixed at the kernel.'
    $lines += '', '| Test | Outcomes |', '|---|---|'
    $lines += @($result.Disagreements | ForEach-Object { "| $($_.Test) | $($_.Outcomes) |" })
    $lines += ''
}
if ($result.Incomplete.Count -gt 0) {
    $lines += "**$($result.Incomplete.Count) incomplete result(s)** - a missing, partially skipped or errored run " +
        'is an infrastructure failure, not agreement.'
    $lines += '', '| Test | Outcomes |', '|---|---|'
    $lines += @($result.Incomplete | ForEach-Object { "| $($_.Test) | $($_.Outcomes) |" })
}

$report = $lines -join "`n"
Write-Host $report
if ($SummaryFile) { $report | Out-File -FilePath $SummaryFile -Append -Encoding utf8 }
exit $(if ($result.Disagreements.Count -gt 0 -or $result.Incomplete.Count -gt 0) { 1 } else { 0 })
