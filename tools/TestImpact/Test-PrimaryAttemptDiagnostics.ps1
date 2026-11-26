[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
. "$PSScriptRoot/../PaperFidelity/PaperFidelity.ps1"
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('aidotnet-diagnostic-intake-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null
$sha = 'a' * 40; $sourceId = '8001'; $digest = 'sha256:' + ('2' * 64)
$passed = Get-Content "$PSScriptRoot/fixtures/runner-failover/passed.trx" -Raw
$passed = $passed.Replace('<UnitTestResult executionId=', '<UnitTestResult duration="00:00:01" executionId=')
$failed = $passed.Replace('Core.Example', 'FailedOnly.Original').Replace('outcome="Passed"', 'outcome="Failed"').Replace('passed="1" failed="0"', 'passed="0" failed="1"')
function Require([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Put([string]$Path, [string]$Content) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    [IO.File]::WriteAllText($Path, $Content)
}
function Invoke-Analyzer([string]$Script, [hashtable]$Arguments) {
    # Exercise the public CLI in its own normal PowerShell process, exactly as the workflow does.
    # The fixture's StrictMode must not change an existing script's execution environment.
    $json = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((ConvertTo-Json -InputObject $Arguments -Depth 20)))
    $scriptLiteral = $Script.Replace("'", "''")
    $command = "`$arguments = ConvertFrom-Json -AsHashtable ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$json'))); & '$scriptLiteral' @arguments"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $text = @(& (Get-Process -Id $PID).Path -NoProfile -OutputFormat Text -EncodedCommand $encoded 2>&1)
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Text = ($text -join "`n") }
}
function Write-PaperManifest([string]$Folder, [string[]]$Classes, [int]$Slice, [int]$SliceCount) {
    $selected = @(Select-PaperFidelitySlice -Classes $Classes -Slice $Slice -SliceCount $SliceCount)
    $chunks = @(for ($i = 0; $i -lt $selected.Count; $i++) { [pscustomobject]@{ chunk = ('chunk{0:d3}' -f ($i + 1)); classes = @($selected[$i]); exit = '0'; seconds = 1 } })
    Put (Join-Path $Folder 'slice-summary.json') ([ordered]@{ schemaVersion = 1; slice = $Slice; sliceCount = $SliceCount; classes = $selected.Count; failedChunks = 0; chunks = $chunks } | ConvertTo-Json -Depth 10)
}
try {
    $names = @('Canonical', 'FailedOnly', 'WrongRun', 'Missing', 'CorruptAttempt', 'CorruptCanonical', 'WrongAttempt', 'WrongSource', 'WrongPolicy', 'NestedRetry')
    $matrix = @($names | ForEach-Object { [pscustomobject]@{ name = $_; framework = 'net10.0'; project = 'tests.csproj'; filter = "FullyQualifiedName~$_"; runner = 'ubuntu-latest' } })
    # Producer receipts and workflow intake both consume the serialized matrix, not pipeline-wrapped strings.
    $matrix = @(ConvertFrom-Json (ConvertTo-Json -InputObject $matrix))
    $attemptRoot = Join-Path $scratch 'attempts'; $canonical = Join-Path $scratch 'canonical'
    foreach ($shard in $matrix) {
        if ($shard.name -eq 'Missing') { continue }
        $container = Join-Path $attemptRoot "shard-attempt-primary-$sha-$($shard.name)-1"
        $receipt = [ordered]@{ schemaVersion = 1; shard = $shard.name; definitionHash = Get-FailoverDefinitionHash $shard
            testedSha = $sha; runId = '7001'; runAttempt = '1'; attempt = 'primary'; sourceArtifactId = $sourceId
            sourceArtifactDigest = $digest; coveragePolicyHash = Get-FailoverCoveragePolicyHash; testOutcome = 'failure' }
        switch ($shard.name) {
            WrongRun { $receipt.runId = '7002' }
            WrongAttempt { $receipt.attempt = 'alternate' }
            WrongSource { $receipt.sourceArtifactId = 'other' }
            WrongPolicy { $receipt.coveragePolicyHash = '0' * 64 }
        }
        Put (Join-Path $container 'shard-attempt.json') ($receipt | ConvertTo-Json)
        $text = if ($shard.name -eq 'CorruptAttempt') { '<not-xml' } else { $failed }
        $initialPath = if ($shard.name -eq 'NestedRetry') { "retry/$($shard.name)/test-results.trx" } else { "$($shard.name)/test-results.trx" }
        Put (Join-Path $container $initialPath) $text
        Put (Join-Path $container "$($shard.name)/retry-1.trx") $passed
        if ($shard.name -eq 'FailedOnly') { Put (Join-Path $container "retry/$($shard.name)/test-results.trx") $passed }
    }
    Put (Join-Path $canonical "coverage-$sha-Canonical/test-results.trx") $passed
    Put (Join-Path $canonical "coverage-$sha-CorruptCanonical/test-results.trx") '<not-xml'
    $intakePath = Join-Path $scratch 'intake.json'
    & "$PSScriptRoot/Import-PrimaryAttemptDiagnostics.ps1" -MatrixJson (ConvertTo-Json -InputObject $matrix) `
        -AttemptRoot $attemptRoot -CanonicalRoot $canonical -ArtifactPrefix coverage -TestedSha $sha `
        -RunId 7001 -RunAttempt 1 -SourceArtifactId $sourceId -SourceArtifactDigest $digest -OutputPath $intakePath
    $intake = Get-Content $intakePath -Raw | ConvertFrom-Json
    Require ($intake.imported -eq 1 -and $intake.canonical -eq 2 -and $intake.rejected -eq 7) ("Primary intake inflated canonical/duplicate evidence or accepted an unbound/corrupt attempt: " + ($intake.shards | ConvertTo-Json -Compress))
    Require (@($intake.shards | Where-Object shard -eq 'FailedOnly')[0].excludedTrx -eq 2) 'Ambiguous retry extras were not disclosed.'
    Require (@(Get-ChildItem $canonical -Recurse -Filter '*retry*.trx').Count -eq 0) 'A retry entered the initial diagnostic inventory.'
    Require ((Get-Content (Join-Path $canonical "coverage-$sha-CorruptCanonical/test-results.trx") -Raw) -ceq '<not-xml') 'Corrupt canonical evidence was hidden by replacement.'
    $provenancePath = Join-Path $scratch 'evidence.json'
    Put $provenancePath ([ordered]@{ verified = $false; diagnosticOnly = $true; testedSha = $sha } | ConvertTo-Json)
    $regression = Join-Path $PSScriptRoot '../../.github/scripts/test-regression-analysis.ps1'
    $aggregate = Join-Path $PSScriptRoot '../../.github/scripts/analyze-test-results.ps1'
    $standardReport = Join-Path $scratch 'standard-report'
    $run = Invoke-Analyzer $regression @{ CurrentResultsPath = $canonical; OutputDirectory = $standardReport; CurrentSha = $sha
        ExpectedShardNames = $names; EvidenceProvenancePath = $provenancePath }
    Require ($run.ExitCode -eq 1) ("Missing standard results did not remain incomplete: " + $run.Text)
    $summary = Get-Content (Join-Path $standardReport 'comparison.json') -Raw | ConvertFrom-Json
    Require ($summary.counts.totalTests -eq 2 -and $summary.counts.reportedFailureResults -eq 1 -and
        $summary.missingExpectedShards.Count -eq 7 -and -not $summary.baselinePublishable) 'Actual standard analyzer omitted/inflated the failed primary or certified incomplete evidence.'
    $ledger = Get-Content (Join-Path $standardReport 'ledger.json') -Raw | ConvertFrom-Json
    Require ($ledger.diagnosticOnly -and -not $ledger.evidenceVerified -and -not $ledger.baselinePublishable) 'Ledger JSON lost diagnostic-only status.'
    $run = Invoke-Analyzer $regression @{ CurrentResultsPath = $canonical; OutputDirectory = (Join-Path $scratch 'invalid-baseline')
        CurrentSha = $sha; BaselineLedgerPath = (Join-Path $standardReport 'ledger.json') }
    Require ($run.ExitCode -ne 0 -and $run.Text -match 'diagnostic-only ledger') 'A diagnostic ledger was accepted as a baseline.'

    # A genuine fixed-failure improvement would otherwise pass every policy criterion. Provenance
    # must prevent both its JSON and its human-readable verdict from claiming a certified PASS.
    $validHead = (git -C (Join-Path $PSScriptRoot '../..') rev-parse HEAD).Trim()
    $baseInput = Join-Path $scratch 'valid-baseline-input'; $baseReport = Join-Path $scratch 'valid-baseline-report'
    $comparisonInput = Join-Path $scratch 'comparison-input'; $comparisonReport = Join-Path $scratch 'comparison-report'
    Put (Join-Path $baseInput "coverage-$sha-Canonical/test-results.trx") ($passed.Replace('outcome="Passed"', 'outcome="Failed"').Replace('passed="1" failed="0"', 'passed="0" failed="1"'))
    Put (Join-Path $comparisonInput "coverage-$sha-Canonical/test-results.trx") $passed
    $run = Invoke-Analyzer $regression @{ CurrentResultsPath = $baseInput; OutputDirectory = $baseReport; CurrentSha = $validHead; ExpectedShardNames = @('Canonical') }
    Require ($run.ExitCode -eq 0) ("Valid baseline fixture failed: " + $run.Text)
    $run = Invoke-Analyzer $regression @{ CurrentResultsPath = $comparisonInput; OutputDirectory = $comparisonReport; CurrentSha = $validHead
        ExpectedShardNames = @('Canonical'); BaselineLedgerPath = (Join-Path $baseReport 'ledger.json'); EvidenceProvenancePath = $provenancePath }
    Require ($run.ExitCode -eq 0) ("Diagnostic comparison failed to produce its report: " + $run.Text)
    $comparison = Get-Content (Join-Path $comparisonReport 'comparison.json') -Raw | ConvertFrom-Json
    $markdown = Get-Content (Join-Path $comparisonReport 'summary.md') -Raw
    Require ($comparison.counts.fixedFailures -eq 1 -and $comparison.criteria.verifiedFailureBalanceShrank -and
        -not $comparison.policyPassed -and -not $comparison.baselinePublishable -and
        $markdown -match 'Policy NOT CERTIFIED' -and $markdown -notmatch '## Policy PASSED') 'Unverified comparison claimed a certified PASS in JSON or markdown.'

    # Packed receipts and initial results belong to their exact producer-assigned slot, not a recursive basename.
    $pack = [pscustomobject]@{ name = 'Pack'; framework = 'net10.0'; segments = @($matrix[0], $matrix[1]) }
    $packRoot = Join-Path $scratch 'packed-attempts'; $packDestination = Join-Path $scratch 'packed-results'
    for ($slot = 0; $slot -lt 2; $slot++) {
        $logical = $pack.segments[$slot]; $directory = Join-Path $packRoot "shard-attempt-primary-$sha-Pack-1/slot-$slot"
        $receipt = [ordered]@{ schemaVersion = 1; shard = $logical.name; definitionHash = Get-FailoverDefinitionHash $logical; testedSha = $sha
            runId = '7001'; runAttempt = '1'; attempt = 'primary'; sourceArtifactId = $sourceId; sourceArtifactDigest = $digest
            coveragePolicyHash = Get-FailoverCoveragePolicyHash; testOutcome = 'failure' }
        Put (Join-Path $directory 'shard-attempt.json') ($receipt | ConvertTo-Json)
        $relative = if ($slot -eq 0) { "$($logical.name)/test-results.trx" } else { "retry/$($logical.name)/test-results.trx" }
        Put (Join-Path $directory $relative) $failed
    }
    $packedReport = Join-Path $scratch 'packed-intake.json'
    & "$PSScriptRoot/Import-PrimaryAttemptDiagnostics.ps1" -MatrixJson (ConvertTo-Json -InputObject @($pack) -Depth 20) `
        -AttemptRoot $packRoot -CanonicalRoot $packDestination -ArtifactPrefix coverage -TestedSha $sha `
        -RunId 7001 -RunAttempt 1 -SourceArtifactId $sourceId -SourceArtifactDigest $digest -OutputPath $packedReport
    $packed = Get-Content $packedReport -Raw | ConvertFrom-Json
    Require ($packed.imported -eq 1 -and $packed.rejected -eq 1) 'Packed slot identity accepted a nested retry or lost its valid initial result.'

    $paperRoot = Join-Path $scratch 'paper-artifacts'; $paperInput = Join-Path $scratch 'paper-input'; $paperReport = Join-Path $scratch 'paper-report'
    $classList = @(0..14 | ForEach-Object { 'Fixture.Class{0:d2}' -f $_ }); $classListPath = Join-Path $scratch 'classes.txt'
    Put $classListPath ($classList -join "`n")
    $notExecuted = $passed.Replace('outcome="Passed"', 'outcome="NotExecuted"').Replace('executed="1" passed="1"', 'executed="0" passed="0"').Replace('notExecuted="0"', 'notExecuted="1"')
    for ($slice = 0; $slice -lt 2; $slice++) {
        for ($chunk = 0; $chunk -lt 5; $chunk++) {
            $text = if ($slice -eq 0 -and $chunk -eq 1) { $failed } elseif ($slice -eq 0 -and $chunk -eq 2) { $notExecuted }
                elseif ($slice -eq 1 -and $chunk -eq 4) { '<not-xml' } else { $passed }
            Put (Join-Path $paperRoot ("paper-fidelity-results-pr-$slice/chunk{0:d3}.trx" -f ($chunk + 1))) $text
        }
        Write-PaperManifest (Join-Path $paperRoot "paper-fidelity-results-pr-$slice") $classList $slice 3
    }
    & "$PSScriptRoot/Import-PaperDiagnostics.ps1" -SlicesJson '[0,1,2]' -ArtifactRoot $paperRoot `
        -Destination $paperInput -TestedSha $sha -OutputDirectory $paperReport -ClassListFile $classListPath -ChunkSize 1
    $run = Invoke-Analyzer $regression @{ CurrentResultsPath = $paperInput; OutputDirectory = $paperReport; CurrentSha = $sha
        ExpectedShardNames = @('Paper fidelity - slice 0', 'Paper fidelity - slice 1', 'Paper fidelity - slice 2')
        EvidenceProvenancePath = (Join-Path $paperReport 'evidence-provenance.json') }
    Require ($run.ExitCode -eq 1) ("Missing paper slice did not remain incomplete: " + $run.Text)
    $paper = Get-Content (Join-Path $paperReport 'comparison.json') -Raw | ConvertFrom-Json
    Require ($paper.counts.totalTests -eq 9 -and $paper.counts.executedTests -eq 8 -and $paper.counts.reportedFailureResults -eq 1 -and
        $paper.missingExpectedShards.Count -eq 1 -and -not $paper.baselinePublishable) 'Paper five-chunk counts, failure, corrupt/missing evidence, or certification status were wrong.'
    $jobsPath = Join-Path $scratch 'jobs.json'
    Put $jobsPath (ConvertTo-Json -InputObject @($names | ForEach-Object { [pscustomobject]@{ name = "Tests (net10.0) - $_"; status = 'completed'; conclusion = $(if ($_ -eq 'Canonical') { 'success' } else { 'failure' }); html_url = '' } }))
    $aggregateReport = Join-Path $scratch 'aggregate-report'
    $run = Invoke-Analyzer $aggregate @{ ResultsPath = $canonical; OutputDirectory = $aggregateReport; JobsPath = $jobsPath; CommitSha = $sha
        EvidenceProvenancePath = $provenancePath; PaperResultsPath = (Join-Path $paperReport 'comparison.json') }
    Require ($run.ExitCode -eq 0) ("Aggregate CLI failed: " + $run.Text)
    $combined = Get-Content (Join-Path $aggregateReport 'ci-test-analysis.json') -Raw | ConvertFrom-Json
    Require ($combined.summary.totalTests -eq 2 -and $combined.summary.failedResultRecords -eq 1 -and $combined.summary.missingResultShards -eq 7 -and
        $combined.paperFidelity.available -and $combined.paperFidelity.expectedSlices -eq 3 -and $combined.paperFidelity.counts.totalTests -eq 9 -and
        $combined.paperFidelity.notExecutedTests -eq 1 -and -not $combined.baselinePublishable) 'Aggregate lost failed primary/paper cases, mixed the separate counts, or certified diagnostics.'
    foreach ($mode in @('complete', 'four-of-five', 'retry-extra', 'missing-summary', 'flattened')) {
        # 'flattened': one matching artifact, which actions/download-artifact extracts into the path itself.
        $root = Join-Path $scratch "paper-$mode"
        $artifact = if ($mode -eq 'flattened') { Join-Path $root 'artifacts' } else { Join-Path $root 'artifacts/paper-fidelity-results-pr-0' }
        $output = Join-Path $root 'output'; $input = Join-Path $root 'input'; $plan = Join-Path $root 'classes.txt'
        $classes = @($classList[0..4]); Put $plan ($classes -join "`n")
        for ($chunk = 1; $chunk -le 5; $chunk++) {
            if ($mode -eq 'four-of-five' -and $chunk -eq 5) { continue }
            Put (Join-Path $artifact ('chunk{0:d3}.trx' -f $chunk)) $passed
        }
        if ($mode -ne 'missing-summary') { Write-PaperManifest $artifact $classes 0 1 }
        if ($mode -eq 'retry-extra') { Put (Join-Path $artifact 'retry/chunk001.trx') $passed }
        & "$PSScriptRoot/Import-PaperDiagnostics.ps1" -SlicesJson '[0]' -ArtifactRoot (Join-Path $root 'artifacts') `
            -Destination $input -TestedSha $sha -OutputDirectory $output -ClassListFile $plan -ChunkSize 1
        $run = Invoke-Analyzer $regression @{ CurrentResultsPath = $input; OutputDirectory = $output; CurrentSha = $sha
            ExpectedShardNames = @('Paper fidelity - slice 0'); EvidenceProvenancePath = (Join-Path $output 'evidence-provenance.json') }
        Require ($run.ExitCode -eq 0) ("Paper $mode report failed: " + $run.Text)
        $result = Get-Content (Join-Path $output 'ledger.json') -Raw | ConvertFrom-Json
        $expectedTotal = if ($mode -eq 'four-of-five') { 4 } else { 5 }
        $expectedStatus = if ($mode -in @('complete', 'flattened')) { 'Passed' } else { 'Incomplete' }
        Require ($result.shards[0].total -eq $expectedTotal -and $result.shards[0].status -ceq $expectedStatus -and
            -not $result.baselinePublishable) "Paper $mode lost partial observations, inflated retries, or hid an incomplete chunk inventory."
    }
    # One matching artifact is extracted into the download path itself, with no per-artifact folder.
    # With several planned slices, the flattened artifact binds only to the slice its summary names.
    $root = Join-Path $scratch 'paper-flattened-multi'; $artifact = Join-Path $root 'artifacts'
    $output = Join-Path $root 'output'; $input = Join-Path $root 'input'; $plan = Join-Path $root 'classes.txt'
    Put $plan ($classList -join "`n")
    $selected = @(Select-PaperFidelitySlice -Classes $classList -Slice 1 -SliceCount 3)
    for ($chunk = 1; $chunk -le $selected.Count; $chunk++) { Put (Join-Path $artifact ('chunk{0:d3}.trx' -f $chunk)) $passed }
    Write-PaperManifest $artifact $classList 1 3
    & "$PSScriptRoot/Import-PaperDiagnostics.ps1" -SlicesJson '[0,1,2]' -ArtifactRoot $artifact `
        -Destination $input -TestedSha $sha -OutputDirectory $output -ClassListFile $plan -ChunkSize 1
    $run = Invoke-Analyzer $regression @{ CurrentResultsPath = $input; OutputDirectory = $output; CurrentSha = $sha
        ExpectedShardNames = @('Paper fidelity - slice 0', 'Paper fidelity - slice 1', 'Paper fidelity - slice 2')
        EvidenceProvenancePath = (Join-Path $output 'evidence-provenance.json') }
    Require ($run.ExitCode -eq 1) ("Flattened paper slice hid the two missing slices: " + $run.Text)
    $result = Get-Content (Join-Path $output 'ledger.json') -Raw | ConvertFrom-Json
    $comparison = Get-Content (Join-Path $output 'comparison.json') -Raw | ConvertFrom-Json
    $flatShard = @($result.shards | Where-Object { $_.name -ceq 'Paper fidelity - slice 1' })
    Require ($flatShard.Count -eq 1 -and $flatShard[0].status -ceq 'Passed' -and $flatShard[0].total -eq $selected.Count -and
        @($result.shards).Count -eq 1 -and @($comparison.missingExpectedShards).Count -eq 2) 'Flattened paper artifact was not bound to the slice its summary names.'
    Write-Host 'Primary/paper diagnostics fixtures passed: canonical deduplication, initial failure retained over retry, 4 binding rejections, corrupt/missing evidence, flattened single-artifact downloads, real analyzer counts and baseline refusal.'
}
finally {
    $absolute = [IO.Path]::GetFullPath($scratch)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Refusing fixture cleanup outside the temporary directory.' }
    Remove-Item -LiteralPath $absolute -Recurse -Force
}
