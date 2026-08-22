$ErrorActionPreference = 'Stop'

$analyzer = Join-Path $PSScriptRoot 'test-regression-analysis.ps1'
$retrySelector = Join-Path $PSScriptRoot 'find-pr-new-failures.ps1'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("aidotnet-trx-analysis-" + [Guid]::NewGuid().ToString('N'))

function Assert-Equal {
    param($Expected, $Actual, [string] $Because)
    if ($Expected -ne $Actual) {
        throw "Expected '$Expected', got '$Actual': $Because"
    }
}

function Write-SyntheticShard {
    param(
        [string] $Root,
        [string] $Sha,
        [string] $Name,
        [string] $TestName,
        [ValidateSet('Passed', 'Failed')] [string] $Outcome,
        [int] $Total = 1,
        [int] $Executed = 1,
        [int] $NotExecuted = 0,
        [string] $FileName = 'test-results.trx',
        [ValidateSet('coverage', 'test-outcome')] [string] $ArtifactPrefix = 'coverage'
    )

    $slug = ($Name -replace '[\\/:*?"<>|\s-]+', '_').Trim('_')
    $directory = Join-Path $Root "$ArtifactPrefix-$Sha-$slug/TestResults/$slug"
    New-Item -Path $directory -ItemType Directory -Force | Out-Null
    [PSCustomObject]@{
        shard = $Name
        slug = $slug
        testStepOutcome = if ($Outcome -eq 'Failed') { 'failure' } else { 'success' }
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path (Split-Path $directory -Parent) 'shard-metadata.json') -Encoding utf8

    $failed = if ($Outcome -eq 'Failed') { 1 } else { 0 }
    $error = if ($Outcome -eq 'Failed') {
        '<Output><ErrorInfo><Message>synthetic assertion</Message></ErrorInfo></Output>'
    } else { '' }
    $trx = @"
<?xml version="1.0" encoding="utf-8"?>
<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">
  <TestDefinitions>
    <UnitTest name="$TestName" id="test-id">
      <TestMethod className="Synthetic.Fixture" name="$TestName" />
    </UnitTest>
  </TestDefinitions>
  <Results>
    <UnitTestResult executionId="execution-id" testId="test-id" testName="$TestName" outcome="$Outcome">$error</UnitTestResult>
  </Results>
  <ResultSummary outcome="$Outcome">
    <Counters total="$Total" executed="$Executed" passed="$($Executed - $failed)" failed="$failed" notExecuted="$NotExecuted" />
  </ResultSummary>
</TestRun>
"@
    $trx | Set-Content -LiteralPath (Join-Path $directory $FileName) -Encoding utf8
}

try {
    $baselineRoot = Join-Path $testRoot 'baseline'
    $currentRoot = Join-Path $testRoot 'current'
    $baselineOutput = Join-Path $testRoot 'baseline-output'
    $comparisonOutput = Join-Path $testRoot 'comparison-output'
    $roundTripOutput = Join-Path $testRoot 'roundtrip-output'

    Write-SyntheticShard $baselineRoot 'aaaa' 'Shard Green' 'GreenTest' 'Passed' -Total 2 -Executed 1 -NotExecuted 1
    Write-SyntheticShard $baselineRoot 'aaaa' 'Shard Existing Red' 'ExistingFailure' 'Failed'
    Write-SyntheticShard $currentRoot 'bbbb' 'Shard Green' 'NewFailure' 'Failed'
    Write-SyntheticShard $currentRoot 'bbbb' 'Shard Existing Red' 'ExistingFailure' 'Passed' -Total 2 -Executed 1

    & $analyzer -CurrentResultsPath $baselineRoot -OutputDirectory $baselineOutput -CurrentSha 'aaaa'
    & $analyzer -CurrentResultsPath $currentRoot -BaselineResultsPath $baselineRoot `
        -OutputDirectory $comparisonOutput -CurrentSha 'bbbb'

    $comparison = Get-Content -LiteralPath (Join-Path $comparisonOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal 1 $comparison.counts.baselineDistinctFailures 'baseline failure count comes from TRX'
    Assert-Equal 1 $comparison.counts.currentDistinctFailures 'current failure count comes from TRX'
    Assert-Equal 1 $comparison.counts.baselinePassedShards 'baseline passed-shard count comes from TRX'
    Assert-Equal 0 $comparison.counts.currentPassedShards 'an incomplete or failed shard is not counted as passed'
    Assert-Equal 1 $comparison.counts.currentReportedFailureResults 'reported failing results remain visible separately from distinct tests'
    Assert-Equal 1 $comparison.counts.newFailures 'a failure absent from the baseline is classified as new'
    Assert-Equal 1 $comparison.counts.fixedFailures 'an explicit current pass classifies the old failure as fixed'
    Assert-Equal 1 $comparison.counts.greenToRedShards 'a previously-green shard becoming red is a hard regression'
    Assert-Equal 1 $comparison.counts.currentIncompleteShards 'executed < total is never treated as a complete shard'
    Assert-Equal $false $comparison.criteria.noConfirmedNewFailures 'the strict verdict exposes any confirmed-new failure'
    Assert-Equal $false $comparison.policyPassed 'the hybrid policy blocks the synthetic regression'
    Assert-Equal 1 $comparison.currentFailureCategories[0].count 'failures are grouped into error categories'
    Assert-Equal 'synthetic assertion' $comparison.currentFailureCategories[0].message 'the first error line names the category'
    Assert-Equal $true (Test-Path -LiteralPath (Join-Path $comparisonOutput 'failures.csv')) 'a spreadsheet-ready failure inventory is emitted'
    Assert-Equal $true (Test-Path -LiteralPath (Join-Path $comparisonOutput 'shards.csv')) 'a spreadsheet-ready shard inventory is emitted'

    $retryOutput = Join-Path $testRoot 'retry-candidates.json'
    $actionOutput = Join-Path $testRoot 'github-output.txt'
    $previousActionOutput = $env:GITHUB_OUTPUT
    try {
        $env:GITHUB_OUTPUT = $actionOutput
        & $retrySelector -CurrentResultsPath $currentRoot -BaselineResultsPath $baselineRoot `
            -OutputFile $retryOutput
    }
    finally {
        $env:GITHUB_OUTPUT = $previousActionOutput
    }
    $retry = Get-Content -LiteralPath $retryOutput -Raw | ConvertFrom-Json
    Assert-Equal 1 $retry.candidateCount 'only the PR-new failure is selected for retry'
    Assert-Equal 'Synthetic.Fixture.NewFailure' $retry.candidates[0].fullyQualifiedName 'retry uses the TRX definition FQN'
    Assert-Equal $true ((Get-Content -LiteralPath $actionOutput -Raw) -match 'rerun_count=1') 'retry count is published to Actions'

    & $analyzer -CurrentResultsPath $currentRoot `
        -BaselineLedgerPath (Join-Path $baselineOutput 'ledger.json') `
        -OutputDirectory $roundTripOutput -CurrentSha 'bbbb'
    $roundTrip = Get-Content -LiteralPath (Join-Path $roundTripOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal $comparison.counts.newFailures $roundTrip.counts.newFailures 'serialized ledger comparison matches direct TRX comparison'
    Assert-Equal $comparison.counts.greenToRedShards $roundTrip.counts.greenToRedShards 'shard transitions survive ledger round-trip'

    $workflowPath = Join-Path $testRoot 'synthetic-workflow.yml'
    $inventoryOutput = Join-Path $testRoot 'inventory-output'
    $inventoryRoot = Join-Path $testRoot 'inventory-current'
    Write-SyntheticShard $inventoryRoot 'cccc' 'Shard Green' 'GreenTest' 'Passed' `
        -ArtifactPrefix 'test-outcome'
    @'
jobs:
  test-net10-sharded:
    strategy:
      matrix:
        shard:
          - name: Shard Green
            project: tests.csproj
          - name: Shard Never Uploaded
            project: tests.csproj
    steps:
      - name: This workflow step is not a shard
        run: echo test
  next-job:
    runs-on: ubuntu-latest
'@ | Set-Content -LiteralPath $workflowPath -Encoding utf8
    & $analyzer -CurrentResultsPath $inventoryRoot -OutputDirectory $inventoryOutput `
        -CurrentSha 'cccc' -CurrentWorkflowPath $workflowPath
    $inventoryLedger = Get-Content -LiteralPath (Join-Path $inventoryOutput 'ledger.json') -Raw | ConvertFrom-Json
    Assert-Equal 2 @($inventoryLedger.shards).Count 'the ledger contains every expected matrix shard'
    Assert-Equal 1 @($inventoryLedger.shards | Where-Object status -eq 'Incomplete').Count 'an expected shard with no artifact is synthesized as incomplete'
    Assert-Equal 'missing-artifact' @($inventoryLedger.shards | Where-Object name -eq 'Shard Never Uploaded')[0].testStepOutcome 'the missing shard is identifiable in the proof'

    $missingOutput = Join-Path $testRoot 'missing-output'
    $missingRoot = Join-Path $testRoot 'missing-current'
    Write-SyntheticShard $missingRoot 'cccc' 'Shard Green' 'GreenTest' 'Passed'
    & $analyzer -CurrentResultsPath $missingRoot -BaselineResultsPath $baselineRoot `
        -OutputDirectory $missingOutput -CurrentSha 'cccc'
    $missing = Get-Content -LiteralPath (Join-Path $missingOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal 1 $missing.counts.missingCurrentShardArtifacts 'an absent red shard artifact is explicitly counted'
    Assert-Equal 1 $missing.counts.currentIncompleteShards 'a missing artifact cannot hide a baseline failure'
    Assert-Equal $false $missing.criteria.verifiedFailureBalanceShrank 'a missing failure result does not earn fix credit'

    $flakeBaselineRoot = Join-Path $testRoot 'flake-baseline'
    $flakeCurrentRoot = Join-Path $testRoot 'flake-current'
    $flakeOutput = Join-Path $testRoot 'flake-output'
    Write-SyntheticShard $flakeBaselineRoot 'dddd' 'Shard Retry' 'RetryTest' 'Passed'
    Write-SyntheticShard $flakeCurrentRoot 'eeee' 'Shard Retry' 'RetryTest' 'Failed'
    Write-SyntheticShard $flakeCurrentRoot 'eeee' 'Shard Retry' 'RetryTest' 'Passed' -FileName 'rerun.trx'
    & $analyzer -CurrentResultsPath $flakeCurrentRoot -BaselineResultsPath $flakeBaselineRoot `
        -OutputDirectory $flakeOutput -CurrentSha 'eeee'
    $flake = Get-Content -LiteralPath (Join-Path $flakeOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal 1 $flake.counts.newFailures 'the original failure remains visible in the report'
    Assert-Equal 0 $flake.counts.confirmedNewFailures 'an identical explicit retry pass clears reproducibility'
    Assert-Equal 1 $flake.counts.rerunPassedNewFailures 'the retry pass is classified as a one-run failure'
    Assert-Equal 0 $flake.counts.greenToRedShards 'a successful targeted retry keeps a green shard green for policy'
    Assert-Equal $true $flake.criteria.noConfirmedNewFailures 'an identical retry pass clears the strict confirmed-new verdict'

    Write-Host 'test-regression-analysis.tests.ps1: all assertions passed.'
}
finally {
    $resolvedTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $resolvedTestRoot = [IO.Path]::GetFullPath($testRoot)
    if ($resolvedTestRoot.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedTestRoot)) {
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
