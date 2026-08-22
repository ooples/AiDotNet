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

function Assert-Matches {
    param([string] $Pattern, [string] $Actual, [string] $Because)
    if ($Actual -notmatch $Pattern) {
        throw "Expected '$Actual' to match '$Pattern': $Because"
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
        [string] $ClassName = 'Synthetic.Fixture'
    )

    $slug = $Name -replace '[\\/:*?"<>|\s-]+', '_'
    $directory = Join-Path $Root "coverage-$Sha-$slug/TestResults/$slug"
    New-Item -Path $directory -ItemType Directory -Force | Out-Null
    [PSCustomObject]@{
        shard = $Name
        slug = $slug
        testStepOutcome = if ($Outcome -eq 'Failed') { 'failure' } else { 'success' }
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path (Split-Path $directory -Parent) 'shard-metadata.json') -Encoding utf8

    $failed = if ($Outcome -eq 'Failed') { 1 } else { 0 }
    $errorXml = if ($Outcome -eq 'Failed') {
        '<Output><ErrorInfo><Message>synthetic assertion</Message></ErrorInfo></Output>'
    } else { '' }
    $escapedClassName = [Security.SecurityElement]::Escape($ClassName)
    $escapedTestName = [Security.SecurityElement]::Escape($TestName)
    $trx = @"
<?xml version="1.0" encoding="utf-8"?>
<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">
  <TestDefinitions>
    <UnitTest name="$escapedTestName" id="test-id">
      <TestMethod className="$escapedClassName" name="$escapedTestName" />
    </UnitTest>
  </TestDefinitions>
  <Results>
    <UnitTestResult executionId="execution-id" testId="test-id" testName="$escapedTestName" outcome="$Outcome">$errorXml</UnitTestResult>
  </Results>
  <ResultSummary outcome="$Outcome">
    <Counters total="$Total" executed="$Executed" passed="$($Executed - $failed)" failed="$failed" notExecuted="$NotExecuted" />
  </ResultSummary>
</TestRun>
"@
    $trx | Set-Content -LiteralPath (Join-Path $directory $FileName) -Encoding utf8
}

function Write-SyntheticMetadataOnlyShard {
    param([string] $Root, [string] $Sha, [string] $Name)

    $slug = $Name -replace '[\\/:*?"<>|\s-]+', '_'
    $directory = Join-Path $Root "coverage-$Sha-$slug/TestResults/$slug"
    New-Item -Path $directory -ItemType Directory -Force | Out-Null
    [PSCustomObject]@{
        shard = $Name
        slug = $slug
        testStepOutcome = 'failure'
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path (Split-Path $directory -Parent) 'shard-metadata.json') -Encoding utf8
}

function Invoke-Git {
    param([string] $Repository, [Parameter(ValueFromRemainingArguments)] [string[]] $Arguments)

    $result = & git -C $Repository @Arguments
    if ($LASTEXITCODE -ne 0) { throw "git $($Arguments -join ' ') failed in $Repository" }
    return $result
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
    Assert-Equal $false $comparison.policyPassed 'the hybrid policy blocks the synthetic regression'
    Assert-Equal $false $comparison.touchedTokenDiscovery.success 'missing git comparison inputs are reported as unknown'
    Assert-Equal $false $comparison.criteria.noTouchedSurfaceRegression 'unknown discovery cannot clear a confirmed new failure'
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

    $escapedBaselineRoot = Join-Path $testRoot 'escaped-baseline'
    $escapedCurrentRoot = Join-Path $testRoot 'escaped-current'
    $escapedRetryOutput = Join-Path $testRoot 'escaped-retry.json'
    $escapedActionOutput = Join-Path $testRoot 'escaped-github-output.txt'
    $operatorClassName = 'Synthetic.Fix(ture)!&|=~\With,Comma'
    Write-SyntheticShard $escapedBaselineRoot 'fa11' 'Shard Operators' 'OperatorTest' 'Passed' -ClassName $operatorClassName
    Write-SyntheticShard $escapedCurrentRoot 'fa12' 'Shard Operators' 'OperatorTest' 'Failed' -ClassName $operatorClassName
    try {
        $env:GITHUB_OUTPUT = $escapedActionOutput
        & $retrySelector -CurrentResultsPath $escapedCurrentRoot -BaselineResultsPath $escapedBaselineRoot `
            -OutputFile $escapedRetryOutput
    }
    finally {
        $env:GITHUB_OUTPUT = $previousActionOutput
    }
    $escapedActionText = Get-Content -LiteralPath $escapedActionOutput -Raw
    $expectedFilter = 'filter=FullyQualifiedName~Synthetic.Fix\(ture\)\!\&\|\=\~\\With,Comma.OperatorTest'
    Assert-Equal $true $escapedActionText.Contains($expectedFilter) 'VSTest operators are escaped while commas are preserved'

    $cappedBaselineRoot = Join-Path $testRoot 'capped-baseline'
    $cappedCurrentRoot = Join-Path $testRoot 'capped-current'
    $cappedRetryOutput = Join-Path $testRoot 'capped-retry.json'
    $cappedActionOutput = Join-Path $testRoot 'capped-github-output.txt'
    Write-SyntheticShard $cappedBaselineRoot 'ca11' 'Shard Cap' 'FirstNewFailure' 'Passed' -FileName 'first.trx'
    Write-SyntheticShard $cappedBaselineRoot 'ca11' 'Shard Cap' 'SecondNewFailure' 'Passed' -FileName 'second.trx'
    Write-SyntheticShard $cappedCurrentRoot 'ca12' 'Shard Cap' 'FirstNewFailure' 'Failed' -FileName 'first.trx'
    Write-SyntheticShard $cappedCurrentRoot 'ca12' 'Shard Cap' 'SecondNewFailure' 'Failed' -FileName 'second.trx'
    try {
        $env:GITHUB_OUTPUT = $cappedActionOutput
        & $retrySelector -CurrentResultsPath $cappedCurrentRoot -BaselineResultsPath $cappedBaselineRoot `
            -OutputFile $cappedRetryOutput -MaxRerunMethods 1
    }
    finally {
        $env:GITHUB_OUTPUT = $previousActionOutput
    }
    $cappedRetry = Get-Content -LiteralPath $cappedRetryOutput -Raw | ConvertFrom-Json
    $cappedActionText = Get-Content -LiteralPath $cappedActionOutput -Raw
    Assert-Equal $true $cappedRetry.retrySkipped 'oversized retry selections are explicitly skipped'
    Assert-Matches 'exceed.*cap' $cappedRetry.retrySkipReason 'the candidate inventory explains why retry was skipped'
    Assert-Equal $true $cappedActionText.Contains('rerun_count=0') 'oversized retry selections publish zero rerun methods'
    Assert-Equal $true ($cappedActionText -match '(?m)^filter=\r?$') 'oversized retry selections publish an empty filter'

    & $analyzer -CurrentResultsPath $currentRoot `
        -BaselineLedgerPath (Join-Path $baselineOutput 'ledger.json') `
        -OutputDirectory $roundTripOutput -CurrentSha 'bbbb'
    $roundTrip = Get-Content -LiteralPath (Join-Path $roundTripOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal $comparison.counts.newFailures $roundTrip.counts.newFailures 'serialized ledger comparison matches direct TRX comparison'
    Assert-Equal $comparison.counts.greenToRedShards $roundTrip.counts.greenToRedShards 'shard transitions survive ledger round-trip'

    $invalidLedgerPath = Join-Path $testRoot 'unsupported-ledger.json'
    $invalidLedger = Get-Content -LiteralPath (Join-Path $baselineOutput 'ledger.json') -Raw | ConvertFrom-Json
    $invalidLedger.schemaVersion = 999
    $invalidLedger | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $invalidLedgerPath -Encoding utf8
    $schemaRejected = $false
    $schemaError = ''
    try {
        & $analyzer -CurrentResultsPath $currentRoot -BaselineLedgerPath $invalidLedgerPath `
            -OutputDirectory (Join-Path $testRoot 'unsupported-ledger-output') -CurrentSha 'bbbb'
    }
    catch {
        $schemaRejected = $true
        $schemaError = $_.Exception.Message
    }
    Assert-Equal $true $schemaRejected 'unsupported ledger schema versions are rejected'
    Assert-Matches 'Unsupported baseline ledger schema' $schemaError 'schema rejection identifies the incompatibility'

    $missingOutput = Join-Path $testRoot 'missing-output'
    $missingRoot = Join-Path $testRoot 'missing-current'
    Write-SyntheticShard $missingRoot 'cccc' 'Shard Green' 'GreenTest' 'Passed'
    & $analyzer -CurrentResultsPath $missingRoot -BaselineResultsPath $baselineRoot `
        -OutputDirectory $missingOutput -CurrentSha 'cccc'
    $missing = Get-Content -LiteralPath (Join-Path $missingOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal 1 $missing.counts.missingCurrentShardArtifacts 'an absent red shard artifact is explicitly counted'
    Assert-Equal 1 $missing.counts.currentIncompleteShards 'a missing artifact cannot hide a baseline failure'
    Assert-Equal 0 $missing.counts.fixedFailures 'a missing failure result does not earn fix credit'
    Assert-Equal $false $missing.policyPassed 'the incomplete missing shard still fails the overall policy'

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

    $repository = Join-Path $testRoot 'touched-repository'
    New-Item -Path $repository -ItemType Directory -Force | Out-Null
    Invoke-Git -Repository $repository -Arguments @('init', '--quiet') | Out-Null
    Invoke-Git -Repository $repository -Arguments @('config', 'user.name', 'AiDotNet CI Test') | Out-Null
    Invoke-Git -Repository $repository -Arguments @('config', 'user.email', 'ci-test@aidotnet.invalid') | Out-Null
    $touchedSource = Join-Path $repository 'TouchedRegressionProbe.cs'
    @"
internal static class TouchedRegressionProbe
{
    internal static int TouchedRegressionProbeMethod() => 1;
}
"@ | Set-Content -LiteralPath $touchedSource -Encoding utf8
    Invoke-Git -Repository $repository -Arguments @('add', 'TouchedRegressionProbe.cs') | Out-Null
    Invoke-Git -Repository $repository -Arguments @('commit', '--quiet', '-m', 'baseline') | Out-Null
    $repositoryBaseSha = ([string] (Invoke-Git -Repository $repository -Arguments @('rev-parse', 'HEAD'))).Trim()
    @"
internal static class TouchedRegressionProbe
{
    internal static int TouchedRegressionProbeMethod() => 2;
}
"@ | Set-Content -LiteralPath $touchedSource -Encoding utf8
    Invoke-Git -Repository $repository -Arguments @('add', 'TouchedRegressionProbe.cs') | Out-Null
    Invoke-Git -Repository $repository -Arguments @('commit', '--quiet', '-m', 'current') | Out-Null
    $repositoryHeadSha = ([string] (Invoke-Git -Repository $repository -Arguments @('rev-parse', 'HEAD'))).Trim()

    $touchedBaselineRoot = Join-Path $testRoot 'touched-baseline'
    $touchedCurrentRoot = Join-Path $testRoot 'touched-current'
    $touchedOutput = Join-Path $testRoot 'touched-output'
    Write-SyntheticShard $touchedBaselineRoot 'to11' 'Shard Touched' 'TouchedRegressionProbeMethod' 'Passed' `
        -ClassName 'Synthetic.TouchedRegressionProbeTests'
    Write-SyntheticShard $touchedCurrentRoot 'to12' 'Shard Touched' 'TouchedRegressionProbeMethod' 'Failed' `
        -ClassName 'Synthetic.TouchedRegressionProbeTests'
    & $analyzer -CurrentResultsPath $touchedCurrentRoot -BaselineResultsPath $touchedBaselineRoot `
        -OutputDirectory $touchedOutput -CurrentSha $repositoryHeadSha -BaselineSha $repositoryBaseSha `
        -RepositoryPath $repository
    $touched = Get-Content -LiteralPath (Join-Path $touchedOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal $true $touched.touchedTokenDiscovery.success 'real git history produces a known touched surface'
    Assert-Equal 1 $touched.counts.touchedNewFailures 'a new failure matching a changed method/type is detected'
    Assert-Equal $false $touched.criteria.noTouchedSurfaceRegression 'a touched-surface regression fails its criterion'

    $neutralBaselineRoot = Join-Path $testRoot 'neutral-baseline'
    $neutralCurrentRoot = Join-Path $testRoot 'neutral-current'
    $neutralOutput = Join-Path $testRoot 'neutral-output'
    Write-SyntheticShard $neutralBaselineRoot 'ne11' 'Shard Neutral' 'StableTest' 'Passed'
    Write-SyntheticShard $neutralCurrentRoot 'ne12' 'Shard Neutral' 'StableTest' 'Passed'
    & $analyzer -CurrentResultsPath $neutralCurrentRoot -BaselineResultsPath $neutralBaselineRoot `
        -OutputDirectory $neutralOutput -CurrentSha $repositoryHeadSha -BaselineSha $repositoryBaseSha `
        -RepositoryPath $repository
    $neutral = Get-Content -LiteralPath (Join-Path $neutralOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal $true $neutral.criteria.verifiedFailureBalanceShrank 'a clean neutral comparison does not require an unrelated fixed failure'
    Assert-Equal $true $neutral.policyPassed 'a complete clean neutral comparison passes policy'

    $missingTrxBaselineRoot = Join-Path $testRoot 'missing-trx-baseline'
    $missingTrxCurrentRoot = Join-Path $testRoot 'missing-trx-current'
    $missingTrxOutput = Join-Path $testRoot 'missing-trx-output'
    Write-SyntheticShard $missingTrxBaselineRoot 'mt11' 'Shard Missing TRX' 'StableTest' 'Passed'
    Write-SyntheticMetadataOnlyShard $missingTrxCurrentRoot 'mt12' 'Shard Missing TRX'
    & $analyzer -CurrentResultsPath $missingTrxCurrentRoot -BaselineResultsPath $missingTrxBaselineRoot `
        -OutputDirectory $missingTrxOutput -CurrentSha $repositoryHeadSha -BaselineSha $repositoryBaseSha `
        -RepositoryPath $repository
    $missingTrx = Get-Content -LiteralPath (Join-Path $missingTrxOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal 'Incomplete' $missingTrx.currentIncompleteShards[0].status 'a shard with metadata but no TRX is incomplete'
    Assert-Equal $false $missingTrx.policyPassed 'a missing TRX cannot pass comparison policy'

    $invalidTrxBaselineRoot = Join-Path $testRoot 'invalid-trx-baseline'
    $invalidTrxCurrentRoot = Join-Path $testRoot 'invalid-trx-current'
    $invalidTrxOutput = Join-Path $testRoot 'invalid-trx-output'
    Write-SyntheticShard $invalidTrxBaselineRoot 'it11' 'Shard Invalid TRX' 'StableTest' 'Passed'
    Write-SyntheticShard $invalidTrxCurrentRoot 'it12' 'Shard Invalid TRX' 'StableTest' 'Passed'
    $invalidTrxPath = Get-ChildItem -LiteralPath $invalidTrxCurrentRoot -Recurse -Filter '*.trx' -File | Select-Object -First 1
    '<not-valid-trx' | Set-Content -LiteralPath $invalidTrxPath.FullName -Encoding utf8
    & $analyzer -CurrentResultsPath $invalidTrxCurrentRoot -BaselineResultsPath $invalidTrxBaselineRoot `
        -OutputDirectory $invalidTrxOutput -CurrentSha $repositoryHeadSha -BaselineSha $repositoryBaseSha `
        -RepositoryPath $repository
    $invalidTrx = Get-Content -LiteralPath (Join-Path $invalidTrxOutput 'comparison.json') -Raw | ConvertFrom-Json
    Assert-Equal 'Incomplete' $invalidTrx.currentIncompleteShards[0].status 'an unparseable TRX is incomplete'
    Assert-Equal $false $invalidTrx.policyPassed 'an unparseable TRX cannot pass comparison policy'

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
