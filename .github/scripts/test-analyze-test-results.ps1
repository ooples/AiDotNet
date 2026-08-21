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
  </Results>
  <ResultSummary><Counters total="3" executed="3" passed="1" failed="2" notExecuted="0" /></ResultSummary>
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
  if ($report.summary.totalTests -ne 3) { throw 'Expected three discovered tests.' }
  if ($report.summary.uniqueFailedTests -ne 2) { throw 'Expected two unique failing test cases.' }
  if ($report.summary.uniqueFailedMethods -ne 1) { throw 'Expected one unique failing test method.' }
  if ($report.summary.failedShards -ne 1) { throw 'Expected one failing shard.' }
  if ($report.regression.status -ne 'regressed') { throw 'Expected a regressed comparison.' }
  if (@($report.regression.newFailedTests).Count -ne 1) { throw 'Expected one new failure.' }
  if (@($report.regression.persistentFailedTests).Count -ne 1) { throw 'Expected one persistent failure.' }
  if ($report.categories.failedTests.'Numerical stability' -ne 1) { throw 'Expected numerical categorization.' }
  if ($report.categories.failedTests.'Assertion mismatch' -ne 1) { throw 'Expected assertion categorization.' }

  $markdown = Get-Content -LiteralPath (Join-Path $output 'ci-test-analysis.md') -Raw
  if ($markdown -notmatch 'New failures \(1\)') { throw 'Markdown omitted the new-failure section.' }
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
