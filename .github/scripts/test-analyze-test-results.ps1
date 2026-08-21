$ErrorActionPreference = 'Stop'

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("aidotnet-ci-analysis-tests-" + [guid]::NewGuid().ToString('N'))
$results = Join-Path $tempRoot 'results'
$artifact = Join-Path $results 'test-results-current-Unit_Example'
$output = Join-Path $tempRoot 'output'
New-Item -ItemType Directory -Path $artifact, $output -Force | Out-Null

try {
  @'
<?xml version="1.0" encoding="utf-8"?>
<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">
  <TestDefinitions>
    <UnitTest name="Case" id="1"><TestMethod className="Example.Tests" name="Case" /></UnitTest>
  </TestDefinitions>
  <Results>
    <UnitTestResult testId="1" testName="Case(1)" outcome="Failed"><Output><ErrorInfo><Message>Assert.Equal() Failure</Message></ErrorInfo></Output></UnitTestResult>
    <UnitTestResult testId="1" testName="Case(2)" outcome="Failed"><Output><ErrorInfo><Message>Loss was NaN</Message></ErrorInfo></Output></UnitTestResult>
    <UnitTestResult testId="1" testName="Case(3)" outcome="Error"><Output><ErrorInfo><Message>Shape mismatch</Message></ErrorInfo></Output></UnitTestResult>
    <UnitTestResult testId="1" testName="Case(4)" outcome="Timeout"><Output><ErrorInfo><Message>Test timed out</Message></ErrorInfo></Output></UnitTestResult>
    <UnitTestResult testId="1" testName="Case(5)" outcome="Aborted"><Output><ErrorInfo><Message>Out of memory</Message></ErrorInfo></Output></UnitTestResult>
  </Results>
  <ResultSummary><Counters total="6" executed="6" passed="1" failed="5" notExecuted="0" /></ResultSummary>
</TestRun>
'@ | Set-Content -LiteralPath (Join-Path $artifact 'test-results.trx') -Encoding utf8

  @'
[
  {
    "name": "Tests (net10.0) - Unit Example",
    "status": "completed",
    "conclusion": "failure",
    "html_url": "https://example.invalid/job"
  }
]
'@ | Set-Content -LiteralPath (Join-Path $tempRoot 'jobs.json') -Encoding utf8

  @'
{
  "schemaVersion": 1,
  "source": { "commitSha": "baseline" },
  "summary": { "uniqueFailedTests": 1, "uniqueFailedMethods": 1, "failedShards": 1 },
  "shards": [{ "name": "Unit Example", "conclusion": "failure" }],
  "failures": [{ "identity": "Example.Tests::Case(1)" }]
}
'@ | Set-Content -LiteralPath (Join-Path $tempRoot 'baseline.json') -Encoding utf8

  & (Join-Path $PSScriptRoot 'analyze-test-results.ps1') `
    -ResultsPath $results `
    -OutputDirectory $output `
    -JobsPath (Join-Path $tempRoot 'jobs.json') `
    -BaselinePath (Join-Path $tempRoot 'baseline.json') `
    -Repository 'ooples/AiDotNet' `
    -CommitSha 'current'

  $report = Get-Content -LiteralPath (Join-Path $output 'ci-test-analysis.json') -Raw | ConvertFrom-Json
  if ($report.summary.totalTests -ne 6) { throw 'Expected six discovered tests.' }
  if ($report.summary.failedResultRecords -ne 5) { throw 'Expected all five terminal failure records.' }
  if ($report.summary.uniqueFailedTests -ne 5) { throw 'Expected five unique failing test cases.' }
  if ($report.summary.uniqueFailedMethods -ne 1) { throw 'Expected one unique failing test method.' }
  if ($report.summary.failedShards -ne 1) { throw 'Expected one failing shard.' }
  if ($report.regression.status -ne 'regressed') { throw 'Expected a regressed comparison.' }
  if (@($report.regression.newFailedTests).Count -ne 4) { throw 'Expected four new failures.' }
  if (@($report.regression.persistentFailedTests).Count -ne 1) { throw 'Expected one persistent failure.' }
  if ($report.categories.failedTests.'Numerical stability' -ne 1) { throw 'Expected numerical categorization.' }
  if ($report.categories.failedTests.'Assertion mismatch' -ne 1) { throw 'Expected assertion categorization.' }
  if ($report.categories.failedTests.'Shape or tensor contract' -ne 1) { throw 'Expected error categorization.' }
  if ($report.categories.failedTests.'Timeout or cancellation' -ne 1) { throw 'Expected timeout categorization.' }
  if ($report.categories.failedTests.'Memory or resource exhaustion' -ne 1) { throw 'Expected aborted categorization.' }
  $outcomes = @($report.failures.outcome | Sort-Object -Unique)
  $missingOutcomes = @('Failed', 'Error', 'Timeout', 'Aborted') |
    Where-Object { $_ -notin $outcomes }
  if (@($outcomes).Count -ne 4 -or @($missingOutcomes).Count -ne 0) {
    throw "Failure report omitted a terminal outcome: $($outcomes -join ', ')."
  }

  $markdown = Get-Content -LiteralPath (Join-Path $output 'ci-test-analysis.md') -Raw
  if ($markdown -notmatch 'New failures \(4\)') { throw 'Markdown omitted the new-failure section.' }
  Write-Host 'analyze-test-results.ps1 self-test passed.'
}
finally {
  if (Test-Path -LiteralPath $tempRoot) {
    $resolved = (Resolve-Path -LiteralPath $tempRoot).Path
    $tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (-not $resolved.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
      throw "Refusing to remove test directory outside the system temp directory: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
  }
}
