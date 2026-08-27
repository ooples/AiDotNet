# Report aggregate test totals across every shard artifact downloaded by CI.
#
# This is a reporter, not a gate. It always exits zero and writes stable outputs for the CI Gate.
# Counts are explicitly labelled as a floor when a TRX is missing/unreadable, a blame-hang sequence
# is present, or the TRX counters show that not every discovered test executed.

[CmdletBinding()]
param(
    [string]$ResultsPath = 'TestResults',
    [string]$WorkflowPath = '.github/workflows/sonarcloud.yml',
    [int]$ExpectedShardCount = 0
)

$ErrorActionPreference = 'Stop'

function Add-Summary {
    param([string]$Line)
    Write-Host $Line
    if ($env:GITHUB_STEP_SUMMARY) {
        try { Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Value $Line -ErrorAction Stop }
        catch { Write-Host "::warning::Could not append aggregate test totals to the step summary: $($_.Exception.Message)" }
    }
}

function Set-CiOutput {
    param([string]$Name, [string]$Value)
    if ($env:GITHUB_OUTPUT) {
        try { Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value "$Name=$Value" -ErrorAction Stop }
        catch { Write-Host "::warning::Could not write aggregate output '$Name': $($_.Exception.Message)" }
    }
}

$body = {
    $expectedShards = $ExpectedShardCount
    if ($expectedShards -le 0 -and (Test-Path -LiteralPath $WorkflowPath)) {
        # Keep this self-updating with the matrix instead of hard-coding 110. Job keys are indented
        # two spaces and shard entries ten, so the next job key closes the section unambiguously.
        $insideShardedJob = $false
        foreach ($line in Get-Content -LiteralPath $WorkflowPath) {
            if ($line -match '^  test-net10-sharded:\s*$') {
                $insideShardedJob = $true
                continue
            }
            if ($insideShardedJob -and $line -match '^  [A-Za-z0-9_-]+:\s*$') { break }
            if ($insideShardedJob -and $line -match '^          - name:\s+') { $expectedShards++ }
        }
    }

    $trxFiles = @(Get-ChildItem -LiteralPath $ResultsPath -Recurse -Filter '*.trx' -ErrorAction SilentlyContinue)
    $sequenceFiles = @(Get-ChildItem -LiteralPath $ResultsPath -Recurse -Filter 'Sequence_*.xml' -ErrorAction SilentlyContinue)
    $failedOccurrences = 0
    $failedNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $hangVictims = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $unparseableTrx = [System.Collections.Generic.List[string]]::new()
    $unparseableSequence = [System.Collections.Generic.List[string]]::new()
    $executed = 0
    $discovered = 0
    $missingCounters = 0

    foreach ($trx in $trxFiles) {
        try {
            [xml]$document = Get-Content -LiteralPath $trx.FullName -Raw
            $failures = @($document.SelectNodes("//*[local-name()='UnitTestResult' and @outcome='Failed']"))
            $failedOccurrences += $failures.Count
            foreach ($failure in $failures) {
                $name = [string]$failure.testName
                if (-not [string]::IsNullOrWhiteSpace($name)) { [void]$failedNames.Add($name) }
            }

            $counters = $document.SelectSingleNode("//*[local-name()='Counters']")
            if ($counters) {
                $executed += [int]$counters.executed
                $discovered += [int]$counters.total
            } else {
                $missingCounters++
            }
        } catch {
            $unparseableTrx.Add("$($trx.FullName): $($_.Exception.Message)")
        }
    }

    foreach ($sequence in $sequenceFiles) {
        try {
            [xml]$document = Get-Content -LiteralPath $sequence.FullName -Raw
            $elements = @($document.SelectNodes("//*[local-name()='UnitTestElement']"))
            if ($elements.Count -eq 0) { continue }
            $last = $elements[$elements.Count - 1]
            $name = [string]$last.FullyQualifiedName
            if ([string]::IsNullOrWhiteSpace($name)) { $name = [string]$last.InnerText }
            if (-not [string]::IsNullOrWhiteSpace($name)) { [void]$hangVictims.Add($name.Trim()) }
        } catch {
            $unparseableSequence.Add("$($sequence.FullName): $($_.Exception.Message)")
        }
    }

    # A blame-hang victim normally has no failed TRX row because the host was killed while it ran.
    # Count only victims not already present so the aggregate matches the normalized baseline logic.
    $additionalHangFailures = 0
    foreach ($victim in $hangVictims) {
        if ($failedNames.Add($victim)) { $additionalHangFailures++ }
    }

    $knownFailures = $failedOccurrences + $additionalHangFailures
    $incomplete = $trxFiles.Count -eq 0 -or $expectedShards -le 0 -or
        $trxFiles.Count -ne $expectedShards -or $unparseableTrx.Count -gt 0 -or
        $unparseableSequence.Count -gt 0 -or
        $missingCounters -gt 0 -or $hangVictims.Count -gt 0 -or
        ($discovered -gt 0 -and $executed -lt $discovered)

    $failureOutput = if ($trxFiles.Count -eq 0) { 'unknown' } else { [string]$knownFailures }
    $distinctOutput = if ($trxFiles.Count -eq 0) { 'unknown' } else { [string]$failedNames.Count }
    $expectedShardOutput = if ($expectedShards -gt 0) { [string]$expectedShards } else { 'unknown' }
    Set-CiOutput 'failed_test_occurrences' $failureOutput
    Set-CiOutput 'distinct_failing_tests' $distinctOutput
    Set-CiOutput 'executed_tests' ([string]$executed)
    Set-CiOutput 'discovered_tests' ([string]$discovered)
    Set-CiOutput 'reported_shards' ([string]$trxFiles.Count)
    Set-CiOutput 'expected_shards' $expectedShardOutput
    Set-CiOutput 'test_totals_incomplete' $incomplete.ToString().ToLowerInvariant()

    Add-Summary '## Aggregate test totals'
    Add-Summary ''
    Add-Summary '| Metric | Total |'
    Add-Summary '|---|---:|'
    Add-Summary "| Failed test occurrences (including unique hang victims) | $failureOutput |"
    Add-Summary "| Distinct failing tests | $distinctOutput |"
    Add-Summary "| Executed tests | $executed |"
    Add-Summary "| Discovered tests | $discovered |"
    Add-Summary "| Parsed TRX files | $($trxFiles.Count - $unparseableTrx.Count) / $($trxFiles.Count) |"
    Add-Summary "| Reported shards | $($trxFiles.Count) / $expectedShardOutput |"
    Add-Summary "| Blame-hang victims | $($hangVictims.Count) |"

    if ($incomplete) {
        Add-Summary ''
        Add-Summary ':warning: **These failure totals are incomplete or unknown.** Treat numeric values as a floor.'
    }
    if ($unparseableTrx.Count -gt 0) {
        foreach ($errorText in $unparseableTrx) { Add-Summary ('    ' + $errorText) }
    }
    if ($unparseableSequence.Count -gt 0) {
        foreach ($errorText in $unparseableSequence) { Add-Summary ('    ' + $errorText) }
    }
}

try {
    & $body
} catch {
    Write-Host "::warning::report-test-totals.ps1 could not complete: $($_.Exception.Message)"
    Set-CiOutput 'failed_test_occurrences' 'unknown'
    Set-CiOutput 'distinct_failing_tests' 'unknown'
    Set-CiOutput 'executed_tests' 'unknown'
    Set-CiOutput 'discovered_tests' 'unknown'
    Set-CiOutput 'reported_shards' 'unknown'
    Set-CiOutput 'expected_shards' 'unknown'
    Set-CiOutput 'test_totals_incomplete' 'true'
}

exit 0
