<#
.SYNOPSIS
    Proves runner failover preserves test failures and rejects incomplete or mismatched evidence.
#>
[CmdletBinding()]
param([string] $Policy = (Join-Path $PSScriptRoot 'RunnerFailover.ps1'))

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $Policy

$fixtureDirectory = Join-Path $PSScriptRoot 'fixtures/runner-failover'
$fixtureText = Get-Content -LiteralPath (Join-Path $fixtureDirectory 'evidence.json') -Raw
$passedTrx = Get-Content -LiteralPath (Join-Path $fixtureDirectory 'passed.trx') -Raw
$failures = [Collections.Generic.List[string]]::new()
$caseCount = 0
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('aidotnet-runner-failover-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null
$originalApi = (Get-Item Function:Invoke-FailoverApi).ScriptBlock
$script:apiFixture = $null
Set-Item Function:Invoke-FailoverApi -Value {
    param([string] $Path, [switch] $Paginate, [string] $Token = '')
    if ($null -eq $script:apiFixture) { throw 'Unexpected API call: fixtures must never access the network.' }
    $script:apiFixture.calls.Add($Path)
    if ($Path -ceq 'repos/ooples/AiDotNet/check-runs/99001') { return $script:apiFixture.check }
    if ($Path -ceq 'repos/ooples/AiDotNet/check-runs/99001/annotations') { return $script:apiFixture.annotations }
    throw "Unexpected API fixture route '$Path'."
}

function New-Evidence { return $fixtureText | ConvertFrom-Json }
function Copy-Evidence($Value) { return $Value | ConvertTo-Json -Depth 100 | ConvertFrom-Json }
function Assert-Case([string] $Name, [scriptblock] $Action, [bool] $Reject = $false) {
    $script:caseCount++
    $rejected = $false
    $rejectionMessage = ''
    try { & $Action | Out-Null } catch { $rejected = $true; $rejectionMessage = $_.Exception.Message }
    if ($rejected -ne $Reject) {
        $script:failures.Add("${Name}: expected rejection=$Reject, observed rejection=$rejected; $rejectionMessage")
    }
}
function New-Receipt($Evidence, [string] $Attempt = 'primary') {
    return [pscustomobject]@{
        schemaVersion = 1; shard = $Evidence.shard.name
        definitionHash = Get-FailoverDefinitionHash $Evidence.shard
        testedSha = $Evidence.testedSha; runId = $Evidence.runId; runAttempt = $Evidence.runAttempt
        sourceArtifactId = $Evidence.artifactId; sourceArtifactDigest = $Evidence.artifactDigest
        coveragePolicyHash = Get-FailoverCoveragePolicyHash $Evidence.coveragePolicy
        collectCoverage = $true
        attempt = $Attempt; originalJobId = $(if ($Attempt -eq 'alternate') { [string]$Evidence.primaryJob.id } else { '' })
        runnerOs = 'Linux'; runnerArch = 'X64'; runnerName = 'cpu-linux-1'; testOutcome = 'success'
    }
}
function Assert-Receipt($Receipt, $Evidence, [string] $Attempt = 'primary') {
    Assert-FailoverReceipt $Receipt $Evidence.shard $Evidence.testedSha $Evidence.runId `
        $Evidence.runAttempt $Evidence.artifactId $Evidence.artifactDigest $Attempt (Get-FailoverCoveragePolicyHash $Evidence.coveragePolicy)
}
function New-TrxDirectory([string] $Name, [string] $Text = $passedTrx) {
    $path = Join-Path $scratch $Name
    New-Item -ItemType Directory -Path $path | Out-Null
    if ($Text) { [IO.File]::WriteAllText((Join-Path $path 'initial.trx'), $Text) }
    return $path
}
# -HostedPrimary: the primary ran on ubuntu-latest and (with -Alternate) the alternate on the self-hosted pool,
# which is the direction a lost hosted shard takes. Without it both jobs come from the self-hosted fixture.
# -EnvironmentPolicy: receipts carry the coverage policy read from the environment, as the reconciler script does.
function New-Resolution([string] $Name, [switch] $Alternate, [switch] $HostedPrimary, [switch] $EnvironmentPolicy) {
    $e = New-Evidence
    if ($EnvironmentPolicy) { $e.coveragePolicy = Get-FailoverCoveragePolicy }
    $primary = Copy-Evidence $e.primaryJob
    $alternateJob = $null
    $attempt = 'primary'
    if ($Alternate) {
        $attempt = 'alternate'
        $alternateJob = Copy-Evidence $e.primaryJob
        $alternateJob.id = 9002
        $alternateJob.name = 'Alternate Tests (net10.0) - Core'
        $alternateJob.conclusion = 'success'
        $alternateJob.runner_name = 'hosted-linux-2'
        $alternateJob.steps[1].name = 'Run tests (sharded) with coverage'
        $alternateJob.steps[1].conclusion = 'success'
    } else {
        $primary.conclusion = 'success'
        $primary.steps[1].name = 'Run tests (sharded) with coverage'
        $primary.steps[1].conclusion = 'success'
    }
    if ($HostedPrimary) {
        $primary.labels = @('ubuntu-latest'); $primary.runner_id = 31; $primary.runner_name = 'GitHub Actions 7'
        $e.annotations[0].message = 'The hosted runner lost communication with the server. Anything in your workflow that terminates the runner process, starves it for CPU/Memory, or blocks its network access can cause this error.'
        if ($Alternate) {
            $alternateJob.name = 'Alternate attempt (net10.0) - Core'
            $alternateJob.runner_id = 5001; $alternateJob.runner_name = 'cpu-linux-1'
        }
    }
    $receipt = New-Receipt $e $attempt
    if ($Alternate) { $receipt.runnerName = $alternateJob.runner_name } elseif ($HostedPrimary) { $receipt.runnerName = $primary.runner_name }
    $canonical = [pscustomobject]@{
        id = 10001; name = "test-results-$($e.testedSha)-Core"
        digest = 'sha256:' + ('5' * 64); expired = $false; workflow_run = [pscustomobject]@{ id = 7001 }
    }
    $immutable = Copy-Evidence $canonical
    $immutable.id = 10002
    $immutable.name = "shard-attempt-$attempt-$($e.testedSha)-Core-$($e.runAttempt)"
    $coverageArtifact = Copy-Evidence $canonical
    $coverageArtifact.id = 10003
    $coverageArtifact.name = "coverage-$($e.testedSha)-Core"
    $coverageArtifact.digest = 'sha256:' + ('6' * 64)
    $result = @{
        Shard = $e.shard; PrimaryJob = $primary; PrimaryAnnotations = $e.annotations; AlternateJob = $alternateJob
        Receipt = $receipt; AttemptReceipt = Copy-Evidence $receipt
        CanonicalArtifact = $canonical; AttemptArtifact = $immutable
        CanonicalDirectory = New-TrxDirectory "$Name-canonical"; AttemptDirectory = New-TrxDirectory "$Name-attempt"
        CoverageArtifact = $coverageArtifact; CoverageDirectory = New-TrxDirectory "$Name-coverage"
        ExpectedAttemptArtifactName = $immutable.name
        TestedSha = $e.testedSha; RunId = $e.runId; RunAttempt = $e.runAttempt
        SourceArtifactId = $e.artifactId; SourceArtifactDigest = $e.artifactDigest
        CoveragePolicyHash = Get-FailoverCoveragePolicyHash $e.coveragePolicy
        CoveragePolicy = $e.coveragePolicy
    }
    $receiptJson = ConvertTo-Json -InputObject $receipt -Depth 30
    foreach ($directory in @($result.CanonicalDirectory, $result.AttemptDirectory, $result.CoverageDirectory)) {
        [IO.File]::WriteAllText((Join-Path $directory 'shard-attempt.json'), $receiptJson)
        [IO.File]::WriteAllText((Join-Path $directory 'coverage-invocation.json'), '{"collectCoverage":true}')
    }
    [IO.File]::WriteAllText((Join-Path $result.CoverageDirectory 'coverage.opencover.xml'), '<CoverageSession><Summary numSequencePoints="1" visitedSequencePoints="1" /></CoverageSession>')
    return $result
}
function Initialize-CheckFixture {
    $e = New-Evidence
    $script:apiFixture = [pscustomobject]@{
        calls = [Collections.Generic.List[string]]::new()
        check = [pscustomobject]@{ head_sha = $e.testedSha; name = $e.primaryJob.name; details_url = $e.primaryJob.html_url }
        annotations = $e.annotations
    }
    return $e
}
function New-PublishedEvidence([string] $Name) {
    $e = New-Evidence
    $resolution = New-Resolution $Name
    $resolved = Assert-FailoverResolvedShard @resolution
    return @{
        Certificate = [pscustomobject]@{
            schemaVersion = 4; testedSha = $e.testedSha; runId = 7001; requiresValidation = $true; requiresTests = $true
            runnerFailoverProof = [pscustomobject]@{ artifactId = '11001'; digest = 'sha256:' + ('7' * 64); runAttempt = '1' }
        }
        Proof = [pscustomobject]@{
            schemaVersion = 1; complete = $true; errors = @(); testedSha = $e.testedSha; runId = '7001'; runAttempt = '1'
            requiresValidation = $true; requiresTests = $true
            sourceArtifactId = $e.artifactId; sourceArtifactDigest = $e.artifactDigest
            coveragePolicyHash = $resolution.CoveragePolicyHash
            requiredShards = @($e.shard); shards = @($resolved)
        }
        Artifacts = @($e.sourceArtifact, $resolution.CanonicalArtifact, $resolution.AttemptArtifact, $resolution.CoverageArtifact)
    }
}

try {
    Assert-Case 'annotations use check-run identity rather than Actions job identity' {
        $e = Initialize-CheckFixture
        $found = @(Get-FailoverAnnotations $e.primaryJob 'ooples/AiDotNet' $e.testedSha)
        if ($found.Count -ne 1 -or $found[0] -isnot [pscustomobject] -or $script:apiFixture.calls.Count -ne 2 -or
            $script:apiFixture.calls[0] -cne 'repos/ooples/AiDotNet/check-runs/99001') {
            throw 'Annotations were not retrieved from the independently bound check run.'
        }
    }
    Assert-Case 'multiple API annotations remain separate evidence records' {
        $e = Initialize-CheckFixture
        $e.primaryJob.labels = @('ubuntu-latest')
        $script:apiFixture.annotations += [pscustomobject]@{
            annotation_level = 'failure'; path = 'test.cs'; start_line = 9; message = 'Assert.Equal() Failure'
        }
        $found = @(Get-FailoverAnnotations $e.primaryJob 'ooples/AiDotNet' $e.testedSha)
        if ($found.Count -ne 2 -or $found[0] -isnot [pscustomobject] -or $found[1] -isnot [pscustomobject]) {
            throw 'API annotations were nested or lost.'
        }
        if (Test-FailoverInfrastructureFailure $e.primaryJob $found) { throw 'Assertion evidence was hidden by array nesting.' }
    }
    foreach ($field in @('head_sha', 'name', 'details_url')) {
        Assert-Case "check-run mismatch $field fails closed" {
            $e = Initialize-CheckFixture; $script:apiFixture.check.$field = 'wrong'
            Get-FailoverAnnotations $e.primaryJob 'ooples/AiDotNet' $e.testedSha
        } $true
    }
    foreach ($url in @('https://api.github.com/repos/attacker/Other/check-runs/99001',
        'https://attacker.example/repos/ooples/AiDotNet/check-runs/99001',
        'https://api.github.com/repos/ooples/AiDotNet/check-runs/99001/annotations', '')) {
        Assert-Case "untrusted check-run URL '$url' fails closed" {
            $e = Initialize-CheckFixture; $e.primaryJob.check_run_url = $url
            Get-FailoverAnnotations $e.primaryJob 'ooples/AiDotNet' $e.testedSha
        } $true
    }
    $script:apiFixture = $null
    Assert-Case 'server runner-loss annotation authorizes infrastructure recovery' {
        $e = New-Evidence
        if (-not (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations)) { throw 'Expected runner-loss authorization.' }
    }
    Assert-Case 'server shutdown annotation authorizes infrastructure recovery' {
        $e = New-Evidence
        $e.annotations[0].message = 'The runner has received a shutdown signal. This may happen when the runner service is stopped.'
        if (-not (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations)) { throw 'Expected shutdown authorization.' }
    }
    Assert-Case 'hosted runner-loss annotation in GitHub''s own wording authorizes one alternate' {
        # Verbatim from run 45538563502 (2026-11-25), where seven hosted shards died this way and none was rerun.
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
        $e.annotations[0].message = 'The hosted runner lost communication with the server. Anything in your workflow that terminates the runner process, starves it for CPU/Memory, or blocks its network access can cause this error.'
        if (-not (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations)) { throw 'Expected hosted runner-loss authorization.' }
        $alternate = New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true
        if ((@($alternate.runner) -join '|') -cne 'self-hosted|Linux|X64|aidotnet-cpu') { throw 'Hosted runner loss must rerun on the alternate pool.' }
    }
    Assert-Case 'hosted runner-loss wording cannot hide a failed step' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.primaryJob.steps[1].conclusion = 'failure'
        $e.annotations[0].message = 'The hosted runner lost communication with the server. Anything in your workflow that terminates the runner process, starves it for CPU/Memory, or blocks its network access can cause this error.'
        if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'A failed step was reclassified as runner loss.' }
    }
    # THE GUARD IS AN ALLOWLIST. Each message below is failure-level evidence of something other than a lost
    # runner. None contains a word the old denylist looked for, so each used to authorise a rerun when it sat
    # next to a runner-loss annotation. They are placed where a server annotation sits (.github, line 1) so
    # that only the wording can tell them apart.
    $hostedLoss = 'The hosted runner lost communication with the server. Anything in your workflow that terminates the runner process, starves it for CPU/Memory, or blocks its network access can cause this error.'
    foreach ($message in @('Process completed with exit code 1.', 'System.OutOfMemoryException: Exception of type was thrown',
        '[xUnit.net] Tests.Foo.Bar [FAIL]', 'Test run failed: 3 failed, 120 passed',
        'Error: The process "/usr/bin/dotnet" was killed with signal SIGKILL')) {
        Assert-Case "hosted runner loss cannot carry the unlisted failure '$message'" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
            $e.annotations += [pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = $message }
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'An unlisted failure annotation was rerun as a lost runner.' }
        }
        Assert-Case "the unlisted failure '$message' alone is not a lost runner" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $message
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'An unlisted failure annotation authorised a rerun.' }
        }
    }
    # Near misses of the server's sentence. Only the sentence from its first character counts: a step can
    # print the words, and other services can be "lost" without the runner being lost.
    foreach ($message in @('Step output: The hosted runner lost communication with the server.',
        'The hosted runner lost communication with the cache service.', 'The hosted runner lost communication',
        'the hosted runner lost communication with the server.', 'The hosted runner lost communication with the server',
        ' The hosted runner lost communication with the server.', 'Warning: The runner has received a shutdown signal.',
        'The hosted runner encountered an error running your tests.')) {
        Assert-Case "near-miss wording '$message' is not a lost runner" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $message
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'A near miss of the runner-loss sentence authorised a rerun.' }
        }
        Assert-Case "near-miss wording '$message' next to the real sentence refuses the rerun" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
            $e.annotations += [pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = $message }
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'A second, unlisted failure annotation was ignored.' }
        }
    }
    Assert-Case 'an annotation level that is neither notice nor warning counts as a failure' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
        $e.annotations += [pscustomobject]@{ annotation_level = 'error'; path = '.github'; start_line = 7; message = 'Something else went wrong.' }
        if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'An unknown annotation level was ignored.' }
    }
    Assert-Case 'notices and warnings beside the runner-loss annotation do not refuse the rerun' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
        $e.annotations += [pscustomobject]@{ annotation_level = 'warning'; path = '.github'; start_line = 12; message = 'Node.js 20 actions are deprecated.' }
        $e.annotations += [pscustomobject]@{ annotation_level = 'notice'; path = '.github'; start_line = 3; message = 'Cache restored from key nuget-1.' }
        if (-not (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations)) { throw 'A benign warning blocked a proven runner loss.' }
    }
    Assert-Case 'a warning that mentions the compiler does not veto a proven hosted runner loss' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
        $e.annotations += [pscustomobject]@{ annotation_level = 'warning'; path = '.github'; start_line = 5; message = 'Compiler cache restored.' }
        $e.annotations += [pscustomobject]@{ annotation_level = 'notice'; path = '.github'; start_line = 6; message = 'Analyzer cache restored; timed out waiting for nothing.' }
        if (-not (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations)) { throw 'A non-failure warning vetoed a proven runner loss.' }
    }
    Assert-Case 'a failure-level compiler annotation still vetoes the hosted runner loss' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
        $e.annotations += [pscustomobject]@{ annotation_level = 'failure'; path = 'src/A.cs'; start_line = 5; message = 'Compiler error CS1002' }
        if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'A compiler failure was hidden.' }
    }
    Assert-Case 'a cancellation warning still refuses the rerun' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
        $e.annotations += [pscustomobject]@{ annotation_level = 'warning'; path = '.github'; start_line = 5; message = 'Workflow manually cancelled' }
        if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'A cancellation was hidden.' }
    }
    Assert-Case 'an annotation with no fields is failure-level evidence, not a lost runner' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.annotations[0].message = $hostedLoss
        $e.annotations += [pscustomobject]@{ title = 'unshaped' }
        if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'A malformed annotation was ignored.' }
    }
    foreach ($conclusion in @('cancelled', 'timed_out', 'startup_failure', 'success', 'neutral', 'unknown')) {
        Assert-Case "hosted job conclusion $conclusion never authorizes alternate" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.primaryJob.conclusion = $conclusion
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'Unexpected authorization.' }
        }
    }
    foreach ($message in @('Assert.Equal() Failure', 'Compiler error CS1002', 'Analyzer gate AIDN001 failed',
        'Process terminated: out of memory', 'OOM', 'Test hang timed out', 'Workflow manually cancelled', 'Run superseded')) {
        Assert-Case "hosted runner annotation cannot hide $message" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
            $e.annotations[0].message = 'The hosted runner encountered an error running your job.'
            $e.annotations += [pscustomobject]@{ annotation_level = 'failure'; path = 'test.cs'; start_line = 4; message = $message }
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'Product or cancellation evidence was hidden.' }
        }
    }
    foreach ($stepConclusion in @('failure', 'timed_out')) {
        Assert-Case "hosted failed step $stepConclusion cannot be reclassified as runner loss" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest'); $e.primaryJob.steps[1].conclusion = $stepConclusion
            $e.annotations[0].message = 'The hosted runner encountered an error running your job.'
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'Failed test/compiler/analyzer step was hidden.' }
        }
    }
    Assert-Case 'hosted missing API annotations fail closed' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
        if (Test-FailoverInfrastructureFailure $e.primaryJob @()) { throw 'Missing evidence authorized recovery.' }
    }
    foreach ($field in @('message', 'path', 'start_line', 'annotation_level')) {
        Assert-Case "hosted unproven annotation $field fails closed" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
            $e.annotations[0].$field = switch ($field) {
                message { 'runner offline (unverified local log)' }
                path { 'tests/Example.cs' }
                start_line { 9 }
                annotation_level { 'warning' }
            }
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'Unproven annotation authorized recovery.' }
        }
    }
    foreach ($selfHostedConclusion in @('failure', 'timed_out', 'startup_failure', 'abandoned', 'stale')) {
        Assert-Case "self-hosted $selfHostedConclusion falls back to hosted runner regardless of step or annotation state" {
            $e = New-Evidence
            $e.primaryJob.conclusion = $selfHostedConclusion
            $e.primaryJob.steps[0].conclusion = 'failure'
            if (-not (Test-FailoverInfrastructureFailure $e.primaryJob @())) { throw "Expected self-hosted $selfHostedConclusion fallback." }
            $alt = New-FailoverAlternateShard $e.shard $e.primaryJob @() $true
            if ($alt.runner -cne 'ubuntu-latest') { throw 'Self-hosted fallback must target ubuntu-latest.' }
        }
    }
    foreach ($cancelMessage in @('Workflow manually cancelled', 'Run superseded')) {
        Assert-Case "self-hosted manual cancellation '$cancelMessage' never authorizes fallback" {
            $e = New-Evidence
            $e.annotations = @([pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = $cancelMessage })
            if (Test-FailoverInfrastructureFailure $e.primaryJob $e.annotations) { throw 'Manual cancellation authorized fallback.' }
        }
    }

    Assert-Case 'qualified Linux X64 runner establishes compatible primary pool' {
        $e = New-Evidence; $pool = Get-FailoverJobPool $e.primaryJob $e.runners
        if ($pool.pool -ne 'self-hosted' -or $pool.os -ne 'Linux' -or $pool.arch -ne 'X64') { throw 'Wrong pool.' }
    }
    foreach ($label in @('Linux', 'X64', 'aidotnet-cpu')) {
        Assert-Case "missing runner qualification $label" {
            $e = New-Evidence; $e.runners[0].labels = @($e.runners[0].labels | Where-Object name -ne $label)
            Get-FailoverJobPool $e.primaryJob $e.runners
        } $true
    }
    Assert-Case 'deregistered or unacquired self-hosted runner still resolves pool from job labels for hosted fallback' {
        $e = New-Evidence; $e.primaryJob.runner_id = 0
        $pool = Get-FailoverJobPool $e.primaryJob @()
        if ($pool.pool -ne 'self-hosted' -or $pool.os -ne 'Linux' -or $pool.arch -ne 'X64') { throw 'Expected label-based self-hosted pool.' }
    }
    Assert-Case 'duplicate runner identity' { $e = New-Evidence; Get-FailoverJobPool $e.primaryJob @($e.runners[0], $e.runners[0]) } $true

    Assert-Case 'self-hosted runner loss chooses hosted alternate with unchanged semantics' {
        $e = New-Evidence
        $alternate = New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true
        if ($alternate.runner -cne 'ubuntu-latest' -or $alternate.failoverOriginalJobId -cne '9001' -or
            $alternate.failoverAttempt -cne 'alternate' -or
            (Get-FailoverDefinitionHash $alternate) -cne (Get-FailoverDefinitionHash $e.shard)) { throw 'Alternate changed source shard semantics.' }
    }
    Assert-Case 'hosted runner loss selects certified idle same-repository alternate' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
        $alternate = New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true
        if ((@($alternate.runner) -join '|') -cne 'self-hosted|Linux|X64|aidotnet-cpu') { throw 'Alternate pool lacks explicit OS/architecture qualification.' }
    }
    Assert-Case 'untrusted fork cannot route to self-hosted runner' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
        New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $false
    } $true
    Assert-Case 'second alternate attempt forbidden by explicit retry state' {
        $e = New-Evidence; New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true $true
    } $true
    foreach ($field in @('failoverAttempt', 'failoverOriginalJobId')) {
        Assert-Case "second alternate forbidden by existing $field" {
            $e = New-Evidence; $e.shard | Add-Member $field 'already-attempted'
            New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true
        } $true
    }
    foreach ($field in @('heavy', 'timeout', 'segments')) {
        Assert-Case "hosted fallback cannot use unqualified capability $field" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
            $value = switch ($field) { heavy { $true }; timeout { 120 }; segments { @((Copy-Evidence $e.shard)) } }
            $e.shard | Add-Member $field $value
            New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true
        } $true
    }
    foreach ($unavailable in @('offline', 'busy', 'wrong-os', 'wrong-arch')) {
        Assert-Case "alternate pool $unavailable is ineligible" {
            $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
            switch ($unavailable) {
                offline { $e.runners[0].status = 'offline' }
                busy { $e.runners[0].busy = $true }
                wrong-os { $e.runners[0].labels = @($e.runners[0].labels | Where-Object name -ne 'Linux') }
                wrong-arch { $e.runners[0].labels = @($e.runners[0].labels | Where-Object name -ne 'X64') }
            }
            New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true
        } $true
    }
    # The cap on self-hosted alternates: one per idle runner.
    Assert-Case 'a second lost shard is refused when the one idle runner already holds an alternate' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
        New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true -SelfHostedAuthorized 1
    } $true
    Assert-Case 'two idle runners take two alternates and refuse the third' {
        $e = New-Evidence; $e.primaryJob.labels = @('ubuntu-latest')
        $second = Copy-Evidence $e.runners[0]; $second.id = 5002; $second.name = 'cpu-linux-2'
        $busy = Copy-Evidence $e.runners[0]; $busy.id = 5003; $busy.name = 'cpu-linux-3'; $busy.busy = $true
        $runners = @($e.runners[0], $second, $busy)
        $null = New-FailoverAlternateShard $e.shard $e.primaryJob $runners $true -SelfHostedAuthorized 0
        $null = New-FailoverAlternateShard $e.shard $e.primaryJob $runners $true -SelfHostedAuthorized 1
        $refused = $false
        try { $null = New-FailoverAlternateShard $e.shard $e.primaryJob $runners $true -SelfHostedAuthorized 2 }
        catch { $refused = $_.Exception.Message -match '2 idle Linux/X64 CPU runner\(s\), one alternate each, 2 already authorised' }
        if (-not $refused) { throw 'The cap did not stop at the number of idle runners, or did not say what it is.' }
    }
    Assert-Case 'the cap does not apply to a self-hosted primary falling back to the hosted pool' {
        $e = New-Evidence
        $alternate = New-FailoverAlternateShard $e.shard $e.primaryJob $e.runners $true -SelfHostedAuthorized 50
        if ($alternate.runner -cne 'ubuntu-latest') { throw 'Hosted fallback was capped by self-hosted capacity.' }
    }

    Assert-Case 'exact source artifact binding passes' {
        $e = New-Evidence
        Assert-FailoverSourceArtifact $e.sourceArtifact $e.artifactId $e.artifactDigest $e.testedSha $e.runId
    }
    foreach ($field in @('id', 'digest', 'name', 'expired', 'workflow_run')) {
        Assert-Case "source artifact rejects mismatched $field" {
            $e = New-Evidence
            switch ($field) {
                id { $e.sourceArtifact.id = 8002 }
                digest { $e.sourceArtifact.digest = 'sha256:' + ('3' * 64) }
                name { $e.sourceArtifact.name = 'build-output-' + ('4' * 40) }
                expired { $e.sourceArtifact.expired = $true }
                workflow_run { $e.sourceArtifact.workflow_run.id = 7002 }
            }
            Assert-FailoverSourceArtifact $e.sourceArtifact $e.artifactId $e.artifactDigest $e.testedSha $e.runId
        } $true
    }

    Assert-Case 'exact primary receipt binding passes' { $e = New-Evidence; Assert-Receipt (New-Receipt $e) $e }
    foreach ($field in @('schemaVersion', 'shard', 'definitionHash', 'testedSha', 'runId', 'runAttempt',
        'sourceArtifactId', 'sourceArtifactDigest', 'attempt', 'runnerOs', 'runnerArch', 'testOutcome', 'coveragePolicyHash')) {
        Assert-Case "receipt rejects mismatch $field" {
            $e = New-Evidence; $r = New-Receipt $e; $r.$field = 'wrong'
            Assert-Receipt $r $e
        } $true
    }
    foreach ($field in @('eventName', 'forceCoverage', 'COVERAGE_CARRIED', 'COVERAGE_RUN_WITHOUT_INSTRUMENTATION', 'COVERAGE_ROLLING')) {
        Assert-Case "changed effective coverage policy $field invalidates receipt" {
            $e = New-Evidence; $receipt = New-Receipt $e
            $e.coveragePolicy.$field = if ($field -like 'COVERAGE_*') { @('DifferentShard') } else { 'changed' }
            Assert-Receipt $receipt $e
        } $true
    }
    foreach ($field in @('project', 'filter', 'framework', 'coverageIncludeDirectory')) {
        Assert-Case "receipt rejects changed shard semantic $field" {
            $e = New-Evidence; $r = New-Receipt $e
            $e.shard.$field = 'changed'
            Assert-Receipt $r $e
        } $true
    }
    Assert-Case 'alternate runner routing alone preserves shard definition' {
        $e = New-Evidence; $other = Copy-Evidence $e.shard; $other.runner = 'aidotnet-cpu'
        $other | Add-Member failoverAttempt alternate
        $other | Add-Member failoverOriginalJobId '9001'
        if ((Get-FailoverDefinitionHash $other) -cne (Get-FailoverDefinitionHash $e.shard)) { throw 'Routing changed logical definition.' }
    }
    Assert-Case 'packed logical slices are enumerated independently' {
        $e = New-Evidence; $second = Copy-Evidence $e.shard; $second.name = 'Other'
        $packed = [pscustomobject]@{ name = 'Pack'; framework = 'net10.0'; segments = @($e.shard, $second) }
        $logical = @(Get-FailoverLogicalShards $packed)
        if ($logical.Count -ne 2 -or $logical[0].name -cne 'Core' -or $logical[1].name -cne 'Other') {
            throw 'Packed slice inventory was nested, collapsed or reordered.'
        }
    }
    Assert-Case 'complete packed logical inventory passes' {
        $e = New-Evidence; $second = Copy-Evidence $e.shard; $second.name = 'Other'
        Assert-FailoverLogicalInventory @($e.shard, $second) @([pscustomobject]@{ shard = 'Core' }, [pscustomobject]@{ shard = 'Other' })
    }
    Assert-Case 'packed missing logical slice fails closed' {
        $e = New-Evidence; $second = Copy-Evidence $e.shard; $second.name = 'Other'
        Assert-FailoverLogicalInventory @($e.shard, $second) @([pscustomobject]@{ shard = 'Core' })
    } $true
    Assert-Case 'duplicate required logical slice fails closed' {
        $e = New-Evidence
        Assert-FailoverLogicalInventory @($e.shard, $e.shard) @([pscustomobject]@{ shard = 'Core' })
    } $true
    Assert-Case 'duplicate resolved logical slice fails closed' {
        $e = New-Evidence
        Assert-FailoverLogicalInventory @($e.shard) @([pscustomobject]@{ shard = 'Core' }, [pscustomobject]@{ shard = 'Core' })
    } $true
    Assert-Case 'unexpected resolved logical slice fails closed' {
        $e = New-Evidence
        Assert-FailoverLogicalInventory @($e.shard) @([pscustomobject]@{ shard = 'Other' })
    } $true
    Assert-Case 'empty logical inventory cannot prove success' { Assert-FailoverLogicalInventory @() @() } $true

    Assert-Case 'one complete passing TRX passes' { Assert-FailoverTrx (New-TrxDirectory 'complete') }
    Assert-Case 'missing TRX fails closed' { Assert-FailoverTrx (New-TrxDirectory 'missing' '') } $true
    Assert-Case 'malformed partial TRX fails closed' { Assert-FailoverTrx (New-TrxDirectory 'malformed' '<TestRun>') } $true
    Assert-Case 'missing finish evidence fails closed' {
        Assert-FailoverTrx (New-TrxDirectory 'unfinished' ($passedTrx -replace ' finish="[^"]+"', ''))
    } $true
    Assert-Case 'failed assertion cannot be hidden by passing counter' {
        Assert-FailoverTrx (New-TrxDirectory 'assertion' ($passedTrx.Replace('outcome="Passed"', 'outcome="Failed"')))
    } $true
    Assert-Case 'failed TRX summary cannot be hidden by passing result counters' {
        Assert-FailoverTrx (New-TrxDirectory 'failed-summary' ($passedTrx.Replace('outcome="Completed"', 'outcome="Failed"')))
    } $true
    Assert-Case 'partial result inventory cannot pass' {
        Assert-FailoverTrx (New-TrxDirectory 'partial' ($passedTrx.Replace('total="1"', 'total="2"')))
    } $true
    Assert-Case 'failed initial TRX plus passing retry cannot erase failure' {
        $path = New-TrxDirectory 'duplicate-trx' ($passedTrx.Replace('outcome="Passed"', 'outcome="Failed"').Replace('failed="0"', 'failed="1"'))
        [IO.File]::WriteAllText((Join-Path $path 'retry.trx'), $passedTrx)
        Assert-FailoverTrx $path
    } $true
    Assert-Case 'duplicate execution identities fail closed' {
        $result = '<UnitTestResult executionId="22222222-2222-2222-2222-222222222222" testId="44444444-4444-4444-4444-444444444444" testName="Core.Other" outcome="Passed" />'
        $text = $passedTrx.Replace('</Results>', "$result</Results>").Replace('total="1"', 'total="2"').Replace('executed="1"', 'executed="2"').Replace('passed="1"', 'passed="2"')
        Assert-FailoverTrx (New-TrxDirectory 'duplicate-identities' $text)
    } $true

    Assert-Case 'successful primary independently produces canonical evidence' {
        $resolution = New-Resolution 'primary-success'
        $resolved = Assert-FailoverResolvedShard @resolution
        if ($resolved.attempt -ne 'primary') { throw 'Primary evidence was replaced.' }
    }
    Assert-Case 'proven runner loss permits one independently passing alternate' {
        $resolution = New-Resolution 'alternate-success' -Alternate
        $resolved = Assert-FailoverResolvedShard @resolution
        if ($resolved.attempt -ne 'alternate') { throw 'Alternate was not resolved.' }
    }
    Assert-Case 'hosted primary lost, self-hosted alternate passed: the alternate is the resolved evidence' {
        $resolution = New-Resolution 'hosted-to-self-hosted' -Alternate -HostedPrimary
        $resolved = Assert-FailoverResolvedShard @resolution
        if ($resolved.attempt -ne 'alternate' -or $resolved.chosenJobId -cne '9002') { throw 'Self-hosted alternate was not resolved.' }
    }
    Assert-Case 'hosted primary lost with an unlisted failure annotation stays failed whatever the alternate did' {
        $resolution = New-Resolution 'hosted-loss-plus-fail' -Alternate -HostedPrimary
        $resolution.PrimaryAnnotations += [pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = 'Test run failed: 3 failed, 120 passed' }
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'real hosted primary assertion failure plus passing alternate remains failure' {
        $resolution = New-Resolution 'assertion-plus-alternate' -Alternate
        $resolution.PrimaryJob.labels = @('ubuntu-latest')
        $resolution.PrimaryJob.steps[1].conclusion = 'failure'
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'self-hosted step failure resolves cleanly when ubuntu-latest fallback passes' {
        $resolution = New-Resolution 'self-hosted-step-fail-hosted-pass' -Alternate
        $resolution.PrimaryJob.steps[1].conclusion = 'failure'
        $resolution.PrimaryAnnotations = @()
        $resolved = Assert-FailoverResolvedShard @resolution
        if ($resolved.attempt -ne 'alternate') { throw 'Self-hosted failure did not resolve via ubuntu-latest fallback.' }
    }
    Assert-Case 'successful primary cannot be overwritten by alternate' {
        $resolution = New-Resolution 'overwrite-primary'
        $alternate = Copy-Evidence $resolution.PrimaryJob; $alternate.id = 9002
        $resolution.AlternateJob = $alternate
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'second alternate retry cannot replace failed first alternate' {
        $resolution = New-Resolution 'second-retry' -Alternate
        $second = Copy-Evidence $resolution.AlternateJob; $second.id = 9003
        $resolution.AlternateJob.conclusion = 'failure'
        $resolution.AlternateJob = @($resolution.AlternateJob, $second)
        Assert-FailoverResolvedShard @resolution
    } $true
    foreach ($field in @('originalJobId', 'runnerName', 'testedSha', 'attempt')) {
        Assert-Case "alternate receipt rejects wrong $field even with matching copies" {
            $resolution = New-Resolution "alternate-wrong-$field" -Alternate
            $resolution.Receipt.$field = 'wrong'
            $resolution.AttemptReceipt.$field = 'wrong'
            Assert-FailoverResolvedShard @resolution
        } $true
    }
    Assert-Case 'canonical receipt cannot differ from immutable attempt' {
        $resolution = New-Resolution 'receipt-mismatch'
        $resolution.AttemptReceipt.runnerName = 'another-runner'
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'canonical TRX cannot differ from immutable attempt' {
        $resolution = New-Resolution 'trx-mismatch'
        $file = Join-Path $resolution.CanonicalDirectory 'initial.trx'
        [IO.File]::WriteAllText($file, $passedTrx.Replace('Core.Example', 'Core.DifferentTest'))
        Assert-FailoverResolvedShard @resolution
    } $true
    foreach ($which in @('CanonicalArtifact', 'AttemptArtifact', 'CoverageArtifact')) {
        foreach ($field in @('id', 'digest', 'name', 'expired', 'workflow_run')) {
            Assert-Case "$which rejects invalid $field" {
                $resolution = New-Resolution "$which-$field"
                switch ($field) {
                    id { $resolution[$which].id = 0 }
                    digest { $resolution[$which].digest = 'invalid' }
                    name { $resolution[$which].name = 'wrong-shard-artifact' }
                    expired { $resolution[$which].expired = $true }
                    workflow_run { $resolution[$which].workflow_run.id = 7002 }
                }
                Assert-FailoverResolvedShard @resolution
            } $true
        }
    }
    Assert-Case 'canonical and immutable attempt artifacts cannot share identity' {
        $resolution = New-Resolution 'same-artifact'
        $resolution.AttemptArtifact.id = $resolution.CanonicalArtifact.id
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'caller cannot relabel a different immutable attempt as canonical evidence' {
        $resolution = New-Resolution 'wrong-attempt-binding'
        $resolution.ExpectedAttemptArtifactName = 'shard-attempt-primary-wrongsha-Core-2'
        $resolution.AttemptArtifact.name = $resolution.ExpectedAttemptArtifactName
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'coverage and canonical artifacts cannot share identity' {
        $resolution = New-Resolution 'duplicate-coverage-artifact'
        $resolution.CoverageArtifact.id = $resolution.CanonicalArtifact.id
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'required coverage output cannot be omitted' {
        $resolution = New-Resolution 'missing-coverage'
        Remove-Item -LiteralPath (Join-Path $resolution.CoverageDirectory 'coverage.opencover.xml')
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'missing actual coverage invocation cannot prove instrumentation policy' {
        $resolution = New-Resolution 'missing-invocation'
        Remove-Item -LiteralPath (Join-Path $resolution.CanonicalDirectory 'coverage-invocation.json')
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'receipt cannot falsely claim coverage was omitted' {
        $resolution = New-Resolution 'wrong-coverage-bool'
        $resolution.Receipt.collectCoverage = $false
        $resolution.AttemptReceipt.collectCoverage = $false
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'matching false receipt and invocation cannot omit required coverage' {
        $resolution = New-Resolution 'omitted-required-coverage'
        $resolution.Receipt.collectCoverage = $false
        $resolution.AttemptReceipt.collectCoverage = $false
        foreach ($directory in @($resolution.CanonicalDirectory, $resolution.AttemptDirectory, $resolution.CoverageDirectory)) {
            [IO.File]::WriteAllText((Join-Path $directory 'coverage-invocation.json'), '{"collectCoverage":false}')
            $resolution.Receipt | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $directory 'shard-attempt.json')
        }
        Remove-Item -LiteralPath (Join-Path $resolution.CoverageDirectory 'coverage.opencover.xml')
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'existing heavy-shard policy can prove correctness without instrumentation' {
        $resolution = New-Resolution 'permitted-no-instrumentation'
        $resolution.Shard | Add-Member heavy $true
        $definition = Get-FailoverDefinitionHash $resolution.Shard
        foreach ($receipt in @($resolution.Receipt, $resolution.AttemptReceipt)) {
            $receipt.definitionHash = $definition; $receipt.collectCoverage = $false
        }
        foreach ($directory in @($resolution.CanonicalDirectory, $resolution.AttemptDirectory, $resolution.CoverageDirectory)) {
            [IO.File]::WriteAllText((Join-Path $directory 'coverage-invocation.json'), '{"collectCoverage":false}')
            $resolution.Receipt | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $directory 'shard-attempt.json')
        }
        Remove-Item -LiteralPath (Join-Path $resolution.CoverageDirectory 'coverage.opencover.xml')
        Assert-FailoverResolvedShard @resolution
    }
    Assert-Case 'coverage artifact cannot replace canonical receipt' {
        $resolution = New-Resolution 'coverage-receipt'
        $file = Join-Path $resolution.CoverageDirectory 'shard-attempt.json'
        [IO.File]::WriteAllText($file, ((Get-Content $file -Raw).Replace('cpu-linux-1', 'unknown-runner')))
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'coverage artifact cannot replace canonical test result' {
        $resolution = New-Resolution 'coverage-trx'
        [IO.File]::WriteAllText((Join-Path $resolution.CoverageDirectory 'initial.trx'), $passedTrx.Replace('Core.Example', 'Core.Other'))
        Assert-FailoverResolvedShard @resolution
    } $true
    Assert-Case 'published proof retains all immutable required shard evidence' {
        $published = New-PublishedEvidence 'published-pass'
        Assert-FailoverPublishedProof @published
    }
    foreach ($field in @('testedSha', 'runId', 'runAttempt', 'sourceArtifactId', 'sourceArtifactDigest', 'coveragePolicyHash')) {
        Assert-Case "published proof rejects changed $field" {
            $published = New-PublishedEvidence "published-$field"
            $published.Proof.$field = 'wrong'
            Assert-FailoverPublishedProof @published
        } $true
    }
    Assert-Case 'published proof cannot omit a required logical slice' {
        $published = New-PublishedEvidence 'published-missing-slice'
        $published.Proof.shards = @()
        Assert-FailoverPublishedProof @published
    } $true
    Assert-Case 'published proof cannot replace required logical definition' {
        $published = New-PublishedEvidence 'published-definition'
        $published.Proof.requiredShards[0].filter = 'FullyQualifiedName~Other'
        Assert-FailoverPublishedProof @published
    } $true
    Assert-Case 'published proof cannot repeat a resolved logical slice' {
        $published = New-PublishedEvidence 'published-duplicate'
        $published.Proof.shards = @($published.Proof.shards[0], $published.Proof.shards[0])
        Assert-FailoverPublishedProof @published
    } $true
    Assert-Case 'diagnostics promote resolved jobs only after successful bound proof' {
        $published = New-PublishedEvidence 'diagnostic-verified'
        $resolved = [pscustomobject]@{ name = 'Tests (net10.0) - Core'; conclusion = 'success' }
        $original = [pscustomobject]@{ name = $resolved.name; conclusion = 'failure' }
        $context = Get-FailoverDiagnosticContext 'success' $published.Proof $published.Proof.testedSha '7001' '1' `
            -OriginalJobs @($original) -ResolvedJobs @($resolved) -RequireResolvedJobs
        if (-not $context.verified -or $context.jobs[0].conclusion -cne 'success' -or
            $context.regressionArtifactPrefix -cne 'test-outcome-ledger' -or $context.aggregateArtifactPrefix -cne 'ci-test-analysis') {
            throw 'Verified recovery was not represented by its certified logical result.'
        }
    }
    foreach ($kind in @('missing-proof', 'failed-verifier', 'wrong-sha', 'wrong-run', 'wrong-attempt', 'incomplete', 'untyped-complete', 'proof-errors', 'missing-slice', 'missing-resolved-job')) {
        Assert-Case "diagnostic $kind preserves failure and cannot certify a baseline" {
            $published = New-PublishedEvidence "diagnostic-$kind"
            $sha = $published.Proof.testedSha; $verifier = 'success'
            $resolved = @([pscustomobject]@{ name = 'Tests (net10.0) - Core'; conclusion = 'success' })
            $original = @([pscustomobject]@{ name = 'Tests (net10.0) - Core'; conclusion = 'failure' })
            switch ($kind) {
                missing-proof { $published.Proof = $null }
                failed-verifier { $verifier = 'failure' }
                wrong-sha { $published.Proof.testedSha = 'wrong' }
                wrong-run { $published.Proof.runId = '7002' }
                wrong-attempt { $published.Proof.runAttempt = '2' }
                incomplete { $published.Proof.complete = $false }
                untyped-complete { $published.Proof.complete = 'true' }
                proof-errors { $published.Proof.errors = @('Required receipt rejected.') }
                missing-slice { $published.Proof.shards = @() }
                missing-resolved-job { $resolved = @() }
            }
            $context = Get-FailoverDiagnosticContext $verifier $published.Proof $sha '7001' '1' `
                -OriginalJobs $original -ResolvedJobs $resolved -RequireResolvedJobs
            if ($context.verified -or -not $context.diagnosticOnly -or $context.reasons.Count -eq 0 -or
                $context.jobs[0].conclusion -cne 'failure' -or
                $context.regressionArtifactPrefix -cne 'test-outcome-diagnostics' -or
                $context.aggregateArtifactPrefix -cne 'ci-test-diagnostics') { throw 'Untrusted evidence was promoted or its original failure disappeared.' }
        }
    }
    Assert-Case 'successful diagnostic jobs cannot satisfy the failed validation gate' {
        $shell = (Get-Process -Id $PID).Path
        & $shell -NoProfile -File (Join-Path $PSScriptRoot 'Assert-CiGate.ps1') -Stage Validation -ReuseScope None `
            -SourceResult success -RequiresValidation true -RequiresTests true -SelectResult success `
            -BuildResult success -BuildCompatResult success -TestsResult failure `
            -RegressionAnalysisResult success -VerdictEnforced false -AggregateAnalysisResult success -SizeCheckResult success | Out-Null
        $gateExitCode = $LASTEXITCODE
        # GitHub's pwsh wrapper propagates LASTEXITCODE. This intentional negative child
        # must not make the successful fixture process report a failure to its caller.
        $global:LASTEXITCODE = 0
        if ($gateExitCode -eq 0) { throw 'Diagnostic output bypassed the failed logical shard gate.' }
    }
    Assert-Case 'diagnostic writer persists missing proof and original failure without certification' {
        $directory = Join-Path $scratch 'diagnostic-writer'
        New-Item -ItemType Directory -Path $directory | Out-Null
        $raw = Join-Path $directory 'original.json'; $chosen = Join-Path $directory 'chosen.json'
        ConvertTo-Json -InputObject @([pscustomobject]@{ name = 'Tests (net10.0) - Core'; conclusion = 'failure' }) |
            Set-Content -LiteralPath $raw
        $priorOutput = $env:GITHUB_OUTPUT; $priorSummary = $env:GITHUB_STEP_SUMMARY
        try {
            $env:GITHUB_OUTPUT = Join-Path $directory 'outputs.txt'; $env:GITHUB_STEP_SUMMARY = Join-Path $directory 'summary.md'
            & "$PSScriptRoot/Write-RunnerDiagnosticContext.ps1" -VerifierResult failure -TestedSha ('a' * 40) `
                -RunId 7001 -RunAttempt 1 -ProofDirectory (Join-Path $directory 'missing') `
                -OutputDirectory $directory -OriginalJobsPath $raw -JobsOutputPath $chosen -WarningAction SilentlyContinue
            $metadata = Get-Content (Join-Path $directory 'evidence-provenance.json') -Raw | ConvertFrom-Json
            $outputs = Get-Content $env:GITHUB_OUTPUT -Raw
            $selected = @(Get-Content $chosen -Raw | ConvertFrom-Json)
            if ($metadata.verified -or -not $metadata.diagnosticOnly -or $metadata.reasons.Count -lt 2 -or
                $selected[0].conclusion -cne 'failure' -or $outputs -notmatch '(?m)^verified=false\r?$' -or
                $outputs -notmatch 'aggregate_artifact_prefix=ci-test-diagnostics' -or
                (Get-Content $env:GITHUB_STEP_SUMMARY -Raw) -notmatch 'DIAGNOSTIC ONLY') { throw 'Diagnostic writer omitted the unverified status or promoted raw results.' }
        }
        finally { $env:GITHUB_OUTPUT = $priorOutput; $env:GITHUB_STEP_SUMMARY = $priorSummary }
    }
    Assert-Case 'raw primary intake prevents certification even after a successful verifier' {
        $published = New-PublishedEvidence 'diagnostic-writer-intake'
        $directory = Join-Path $scratch 'diagnostic-writer-intake'
        New-Item -ItemType Directory -Path $directory | Out-Null
        $published.Proof | ConvertTo-Json -Depth 100 | Set-Content (Join-Path $directory 'proof.json')
        [pscustomobject]@{ imported = 1; rejected = 0; canonical = 0 } | ConvertTo-Json |
            Set-Content (Join-Path $directory 'intake.json')
        $priorOutput = $env:GITHUB_OUTPUT; $priorSummary = $env:GITHUB_STEP_SUMMARY
        try {
            $env:GITHUB_OUTPUT = Join-Path $directory 'outputs.txt'; $env:GITHUB_STEP_SUMMARY = ''
            & "$PSScriptRoot/Write-RunnerDiagnosticContext.ps1" -VerifierResult success -TestedSha $published.Proof.testedSha `
                -RunId 7001 -RunAttempt 1 -ProofDirectory $directory -OutputDirectory $directory `
                -PrimaryAttemptImportPath (Join-Path $directory 'intake.json')
            $metadata = Get-Content (Join-Path $directory 'evidence-provenance.json') -Raw | ConvertFrom-Json
            if ($metadata.verified -or -not $metadata.diagnosticOnly -or $metadata.primaryAttemptDiagnostics.imported -ne 1 -or
                $metadata.regressionArtifactPrefix -cne 'test-outcome-diagnostics') { throw 'Substituted raw primary TRX was certified.' }
        }
        finally { $env:GITHUB_OUTPUT = $priorOutput; $env:GITHUB_STEP_SUMMARY = $priorSummary }
    }
    foreach ($kind in @('canonical', 'coverage', 'attempt')) {
        Assert-Case "published $kind artifact replacement invalidates proof" {
            $published = New-PublishedEvidence "published-replaced-$kind"
            $id = $published.Proof.shards[0]."${kind}ArtifactId"
            $artifact = @($published.Artifacts | Where-Object { [string]$_.id -ceq [string]$id })[0]
            $artifact.digest = 'sha256:' + ('8' * 64)
            Assert-FailoverPublishedProof @published
        } $true
    }
    # THE TWO SCRIPTS THE WORKFLOW RUNS, END TO END. New-RunnerFailoverPlan.ps1 decides what is rerun and
    # Resolve-RunnerFailoverResults.ps1 decides what counts; both are run here as files, against a copy of the
    # policy under test, with the GitHub REST layer and the artifact download replaced by fixtures. The policy
    # cases above cannot see what these scripts do with it: the cap is counted by the planner, the previous run
    # is fetched by the planner, and the pool check between a hosted primary and its self-hosted alternate is
    # made by the reconciler.
    $e2eRoot = Join-Path $scratch 'e2e'
    $e2eTools = Join-Path $e2eRoot 'tools/TestImpact'
    New-Item -ItemType Directory -Path $e2eTools, (Join-Path $e2eRoot '.github/scripts') -Force | Out-Null
    Copy-Item -LiteralPath $Policy -Destination (Join-Path $e2eTools 'RunnerFailover.ps1')
    foreach ($file in @('New-RunnerFailoverPlan.ps1', 'Resolve-RunnerFailoverResults.ps1')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination $e2eTools
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../.github/scripts/Get-ShardCoveragePolicy.ps1') -Destination (Join-Path $e2eRoot '.github/scripts')
    # Stands in for the artifact download: copies the fixture directory registered for the artifact ID.
    Set-Content -LiteralPath (Join-Path $e2eTools 'Receive-RequiredArtifact.ps1') -Encoding utf8 -Value (@(
        'param([string] $Repository, [string] $ArtifactId, [string] $ExpectedDigest, [string] $Destination, [string] $RunId, [int] $JobIndex)'
        '$map = Get-Content -LiteralPath $env:FAILOVER_TEST_ARTIFACT_MAP -Raw | ConvertFrom-Json'
        'if (-not $map.PSObject.Properties[$ArtifactId]) { exit 1 }'
        'New-Item -ItemType Directory -Path $Destination -Force | Out-Null'
        'Copy-Item -Path (Join-Path $map.$ArtifactId ''*'') -Destination $Destination -Recurse -Force'
        'exit 0'
    ) -join "`n")

    $e2eRepository = 'ooples/AiDotNet'
    $e2eHistory = "repos/$e2eRepository/actions/workflows/55/runs?event=push&status=completed&branch=master"
    $restRoutes = @{}
    $restCalls = [Collections.Generic.List[string]]::new()
    . "$PSScriptRoot/Invoke-GitHubApiWithRetry.ps1"
    $priorRetryE2e = (Get-Item Function:Invoke-GitHubApiWithRetry).ScriptBlock
    $e2eEnvironment = [ordered]@{
        GH_TOKEN = 'test-token'; GITHUB_EVENT_NAME = 'push'; SHARD_FORCE_COVERAGE = 'false'; COVERAGE_CARRIED = ''
        COVERAGE_RUN_WITHOUT_INSTRUMENTATION = ''; COVERAGE_ROLLING = '["Core"]'
        RUNNER_TEMP = (Join-Path $e2eRoot 'runner-temp'); GITHUB_OUTPUT = (Join-Path $e2eRoot 'github-output.txt')
        GITHUB_STEP_SUMMARY = (Join-Path $e2eRoot 'step-summary.md'); FAILOVER_TEST_ARTIFACT_MAP = (Join-Path $e2eRoot 'artifact-map.json')
    }
    $e2ePriorEnvironment = @{}
    foreach ($key in $e2eEnvironment.Keys) { $e2ePriorEnvironment[$key] = [Environment]::GetEnvironmentVariable($key) }

    function New-E2eScenario([string] $Name, [string[]] $ShardNames = @('Core'), [int] $IdleRunners = 1, [switch] $SelfHostedPrimary) {
        $e = New-Evidence
        $routes = @{}
        $jobs = [Collections.Generic.List[object]]::new()
        $matrix = [Collections.Generic.List[object]]::new()
        for ($index = 0; $index -lt $ShardNames.Count; $index++) {
            $shard = Copy-Evidence $e.shard; $shard.name = $ShardNames[$index]
            $job = Copy-Evidence $e.primaryJob
            $job.id = 9001 + 10 * $index; $job.name = "Tests (net10.0) - $($ShardNames[$index])"
            $check = 99001 + 10 * $index
            $job.check_run_url = "https://api.github.com/repos/$e2eRepository/check-runs/$check"
            $job.html_url = "https://github.com/$e2eRepository/actions/runs/7001/job/$($job.id)"
            $message = 'The self-hosted runner lost communication with the server. Verify the machine is running.'
            if (-not $SelfHostedPrimary) {
                $job.labels = @('ubuntu-latest'); $job.runner_id = 31 + $index; $job.runner_name = "GitHub Actions $(7 + $index)"
                $message = $hostedLoss
            }
            $routes["repos/$e2eRepository/check-runs/$check"] = [pscustomobject]@{ head_sha = $e.testedSha; name = $job.name; details_url = $job.html_url }
            $routes["repos/$e2eRepository/check-runs/$check/annotations"] = @([pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = $message })
            $jobs.Add($job); $matrix.Add($shard)
        }
        $runners = @(for ($index = 0; $index -lt $IdleRunners; $index++) {
            $runner = Copy-Evidence $e.runners[0]; $runner.id = 5001 + $index; $runner.name = "cpu-linux-$(1 + $index)"; $runner
        })
        $routes["repos/$e2eRepository/actions/runs/7001"] = [pscustomobject]@{
            id = 7001; run_attempt = 1; conclusion = $null; head_sha = $e.testedSha; workflow_id = 55; event = 'push'; head_branch = 'master'; run_number = 21
        }
        $routes["repos/$e2eRepository/actions/artifacts/8001"] = $e.sourceArtifact
        $routes["repos/$e2eRepository/actions/runs/7001/attempts/1/jobs"] = [pscustomobject]@{ jobs = $jobs }
        $routes["repos/$e2eRepository/actions/runners"] = [pscustomobject]@{ runners = $runners }
        $routes[$e2eHistory] = [pscustomobject]@{ workflow_runs = @() }
        return @{ Name = $Name; Evidence = $e; Routes = $routes; Jobs = $jobs
            MatrixJson = (ConvertTo-Json -InputObject @($matrix) -Depth 30 -Compress); PlanDirectory = (Join-Path $e2eRoot "plan-$Name") }
    }
    # Records an earlier run (6990) in which the job of that name ended with the given failure annotation. Two
    # newer runs are listed too - one cancelled, one on another branch - and neither has a job route, so
    # reading either as "the previous run" fails the case.
    function Add-E2ePreviousRun($Scenario, [string] $JobName, [string] $Message, [string] $Conclusion = 'failure') {
        $previousSha = '3' * 40
        $Scenario.Routes[$e2eHistory] = [pscustomobject]@{ workflow_runs = @(
            [pscustomobject]@{ id = 6995; run_number = 19; conclusion = 'cancelled'; event = 'push'; head_branch = 'master'; head_sha = ('4' * 40) }
            [pscustomobject]@{ id = 6990; run_number = 18; conclusion = 'failure'; event = 'push'; head_branch = 'master'; head_sha = $previousSha }
            [pscustomobject]@{ id = 6980; run_number = 17; conclusion = 'success'; event = 'push'; head_branch = 'master'; head_sha = ('5' * 40) }
            [pscustomobject]@{ id = 7001; run_number = 21; conclusion = $null; event = 'push'; head_branch = 'master'; head_sha = $Scenario.Evidence.testedSha }
            [pscustomobject]@{ id = 6999; run_number = 20; conclusion = 'failure'; event = 'push'; head_branch = 'other'; head_sha = ('6' * 40) }
        ) }
        $previous = [pscustomobject]@{
            id = 8901; name = $JobName; status = 'completed'; conclusion = $Conclusion; labels = @('ubuntu-latest')
            check_run_url = "https://api.github.com/repos/$e2eRepository/check-runs/98001"
            html_url = "https://github.com/$e2eRepository/actions/runs/6990/job/8901"
            steps = @([pscustomobject]@{ name = 'Run tests (sharded) with coverage'; conclusion = $null })
        }
        $Scenario.Routes["repos/$e2eRepository/actions/runs/6990/jobs?filter=all"] = [pscustomobject]@{ jobs = @($previous) }
        $Scenario.Routes["repos/$e2eRepository/check-runs/98001"] = [pscustomobject]@{ head_sha = $previousSha; name = $JobName; details_url = $previous.html_url }
        $Scenario.Routes["repos/$e2eRepository/check-runs/98001/annotations"] = @([pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = $Message })
    }
    function Use-E2eRoutes($Scenario) {
        $restRoutes.Clear(); $restCalls.Clear()
        foreach ($key in $Scenario.Routes.Keys) { $restRoutes[$key] = $Scenario.Routes[$key] }
        Set-Content -LiteralPath $env:GITHUB_OUTPUT -Value ''
        Set-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Value ''
    }
    function Invoke-E2ePlan($Scenario) {
        Use-E2eRoutes $Scenario
        $e = $Scenario.Evidence
        $log = & (Join-Path $e2eTools 'New-RunnerFailoverPlan.ps1') -MatrixJson $Scenario.MatrixJson -Repository $e2eRepository -RunId '7001' `
            -RunAttempt '1' -TestedSha $e.testedSha -SourceArtifactId $e.artifactId -SourceArtifactDigest $e.artifactDigest `
            -TrustedSameRepository $true -SelfHostedRunnerToken 'runner-token' -OutputDirectory $Scenario.PlanDirectory *>&1 | Out-String
        return [pscustomobject]@{
            Log = $log; Plan = (Get-Content -LiteralPath (Join-Path $Scenario.PlanDirectory 'plan.json') -Raw | ConvertFrom-Json)
            Output = [string](Get-Content -LiteralPath $env:GITHUB_OUTPUT -Raw); Summary = [string](Get-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Raw)
            Calls = @($restCalls)
        }
    }
    # Adds what the alternate attempt leaves behind (its job and the three artifacts) and reconciles.
    function Invoke-E2eResolve($Scenario, $Resolution) {
        $e = $Scenario.Evidence
        $artifacts = @($Resolution.CanonicalArtifact, $Resolution.AttemptArtifact, $Resolution.CoverageArtifact)
        $Scenario.Routes["repos/$e2eRepository/actions/runs/7001/artifacts"] = [pscustomobject]@{ artifacts = $artifacts }
        [ordered]@{ "$($Resolution.CanonicalArtifact.id)" = $Resolution.CanonicalDirectory; "$($Resolution.AttemptArtifact.id)" = $Resolution.AttemptDirectory
            "$($Resolution.CoverageArtifact.id)" = $Resolution.CoverageDirectory } | ConvertTo-Json | Set-Content -LiteralPath $env:FAILOVER_TEST_ARTIFACT_MAP -Encoding utf8
        Use-E2eRoutes $Scenario
        $env:RUNNER_TEMP = Join-Path $e2eRoot "runner-temp-$($Scenario.Name)"
        New-Item -ItemType Directory -Path $env:RUNNER_TEMP -Force | Out-Null
        $proofDirectory = Join-Path $e2eRoot "proof-$($Scenario.Name)"
        $log = & (Join-Path $e2eTools 'Resolve-RunnerFailoverResults.ps1') -MatrixJson $Scenario.MatrixJson -PlanPath (Join-Path $Scenario.PlanDirectory 'plan.json') `
            -Repository $e2eRepository -RunId '7001' -RunAttempt '1' -TestedSha $e.testedSha -SourceArtifactId $e.artifactId `
            -SourceArtifactDigest $e.artifactDigest -TrustedSameRepository $true -SelfHostedRunnerToken 'runner-token' -OutputDirectory $proofDirectory *>&1 | Out-String
        return [pscustomobject]@{
            Log = $log; Proof = (Get-Content -LiteralPath (Join-Path $proofDirectory 'proof.json') -Raw | ConvertFrom-Json)
            Resolved = @(Get-Content -LiteralPath (Join-Path $proofDirectory 'resolved-jobs.json') -Raw | ConvertFrom-Json)
            Summary = [string](Get-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Raw)
        }
    }

    # For a recovery the reconciler must reject: runs it, requires the gate to fail, and returns the reasons it
    # wrote to proof.json, so a case can require the right refusal rather than any refusal.
    function Get-E2eResolveErrors($Scenario, $Resolution) {
        $rejected = $false
        try { $null = Invoke-E2eResolve $Scenario $Resolution } catch { $rejected = $_.Exception.Message -match '^Logical shard evidence gate failed' }
        if (-not $rejected) { throw 'The reconciler accepted the recovery.' }
        $proof = Get-Content -LiteralPath (Join-Path $e2eRoot "proof-$($Scenario.Name)/proof.json") -Raw | ConvertFrom-Json
        if ($proof.complete) { throw 'A failed gate left a complete proof.' }
        return (@($proof.errors) -join "`n")
    }

    try {
        foreach ($key in $e2eEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key, $e2eEnvironment[$key]) }
        New-Item -ItemType Directory -Path $env:RUNNER_TEMP -Force | Out-Null
        Set-Item Function:Invoke-GitHubApiWithRetry -Value {
            param($Headers, [string] $Uri)
            $path = $Uri.Substring('https://api.github.com/'.Length) -replace '[?&]per_page=50&page=1$', ''
            $restCalls.Add($path)
            if (-not $restRoutes.ContainsKey($path)) { throw "Unexpected API fixture route '$path'." }
            return $restRoutes[$path]
        }.GetNewClosure()

        Assert-Case 'planner: one lost hosted shard is rerun once on the idle runner, and says so' {
            $result = Invoke-E2ePlan (New-E2eScenario 'one-lost')
            if (@($result.Plan.alternate).Count -ne 1 -or (@($result.Plan.alternate[0].runner) -join '|') -cne 'self-hosted|Linux|X64|aidotnet-cpu' -or
                -not $result.Plan.attempts[0].recoveryAuthorized -or $result.Output -notmatch 'has_recovery=true') { throw 'The lost shard was not planned for one alternate.' }
            if ($result.Log -notmatch "::warning title=Shard rerun after runner loss::'Core' lost its hosted runner \(job 9001\) and is rerun once on a self-hosted runner\.") { throw 'The rerun is not annotated on the run.' }
            if ($result.Summary -notmatch '\| Core \| 9001 \| rerun once') { throw 'The rerun is not in the step summary.' }
        }
        Assert-Case 'planner: three lost shards and one idle runner give one alternate; the others are refused aloud' {
            $result = Invoke-E2ePlan (New-E2eScenario 'three-lost-one-idle' -ShardNames @('Core', 'Core B', 'Core C'))
            $refused = @($result.Plan.attempts | Where-Object { -not $_.recoveryAuthorized })
            if (@($result.Plan.alternate).Count -ne 1 -or $result.Plan.alternate[0].name -cne 'Core' -or $refused.Count -ne 2) { throw "Expected 1 alternate and 2 refusals; got $(@($result.Plan.alternate).Count) and $($refused.Count)." }
            if (@($refused | Where-Object { $_.reason -notmatch '^Alternate capacity is used up: 1 idle Linux/X64 CPU runner\(s\), one alternate each, 1 already authorised\.$' }).Count -ne 0) { throw 'A refusal does not state the cap.' }
            if (@([regex]::Matches($result.Log, '::error title=Lost shard not rerun::')).Count -ne 2) { throw 'A refused shard is not annotated as an error.' }
            if ($result.Summary -notmatch '\| Core B \| 9011 \| NOT rerun' -or $result.Summary -notmatch 'capped at one per idle runner; 1 authorised') { throw 'The refusals and the cap are not in the step summary.' }
        }
        Assert-Case 'planner: three lost shards and two idle runners give two alternates' {
            $result = Invoke-E2ePlan (New-E2eScenario 'three-lost-two-idle' -ShardNames @('Core', 'Core B', 'Core C') -IdleRunners 2)
            if (@($result.Plan.alternate).Count -ne 2 -or (@($result.Plan.alternate.name) -join '|') -cne 'Core|Core B') { throw 'The cap is not the number of idle runners.' }
        }
        Assert-Case 'planner: a shard lost on the previous run as well is not rerun' {
            $scenario = New-E2eScenario 'lost-twice'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 0 -or $result.Plan.attempts[0].recoveryAuthorized -or $result.Output -notmatch 'has_recovery=false') { throw 'A shard lost on consecutive runs was rerun.' }
            if ($result.Plan.attempts[0].reason -notmatch '^Lost its runner on the previous run \(6990\) as well; not rerun\.') { throw "Wrong reason: $($result.Plan.attempts[0].reason)" }
            if (@($result.Plan.repeatedRunnerLoss).Count -ne 1 -or $result.Plan.repeatedRunnerLoss[0].shard -cne 'Core' -or $result.Plan.repeatedRunnerLoss[0].previousRunId -cne '6990') { throw 'The repeated loss is not recorded in the plan.' }
            if ($result.Log -notmatch "::error title=Lost shard not rerun::'Core': Lost its runner on the previous run") { throw 'The repeated loss is not annotated as an error.' }
        }
        Assert-Case 'planner: a different shard lost on the previous run does not block this one' {
            $scenario = New-E2eScenario 'other-shard-lost'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core B' $hostedLoss
            if (@((Invoke-E2ePlan $scenario).Plan.alternate).Count -ne 1) { throw 'Another shard''s loss refused this rerun.' }
        }
        Assert-Case 'planner: a test failure on the previous run is not a previous runner loss' {
            $scenario = New-E2eScenario 'failed-then-lost'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' 'Process completed with exit code 1.'
            if (@((Invoke-E2ePlan $scenario).Plan.alternate).Count -ne 1) { throw 'A previous test failure was counted as a runner loss.' }
        }
        Assert-Case 'planner: a pass on the previous run leaves the rerun allowed, and cancelled runs are passed over' {
            $scenario = New-E2eScenario 'passed-then-lost'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss 'success'
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 1) { throw 'A previous pass refused the rerun.' }
            if (@($result.Calls | Where-Object { $_ -match 'runs/(6995|6980|6999)/' }).Count -ne 0) { throw 'A cancelled, older or other-branch run was read as the previous run.' }
        }
        Assert-Case 'planner: history that cannot be read refuses the rerun' {
            $scenario = New-E2eScenario 'history-unreadable'
            $scenario.Routes.Remove($e2eHistory)
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 0 -or $result.Plan.attempts[0].reason -notmatch '^The previous run could not be read, so the shard is not rerun: ') { throw 'A rerun was authorised without its history.' }
        }
        Assert-Case 'planner: a lost-runner annotation beside an unlisted failure plans nothing and reads no history' {
            $scenario = New-E2eScenario 'loss-plus-exit-code'
            $scenario.Routes["repos/$e2eRepository/check-runs/99001/annotations"] = @(
                [pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = $hostedLoss }
                [pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = 'Process completed with exit code 1.' })
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 0 -or $result.Log -match '::(warning|error) title=' -or
                @($result.Calls | Where-Object { $_ -match 'workflows/55' }).Count -ne 0) { throw 'A product failure was treated as a lost runner.' }
        }
        Assert-Case 'planner: a self-hosted primary still falls back to the hosted pool after two losses' {
            $scenario = New-E2eScenario 'self-hosted-twice' -SelfHostedPrimary
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 1 -or $result.Plan.alternate[0].runner -cne 'ubuntu-latest' -or
                @($result.Calls | Where-Object { $_ -match 'workflows/55' }).Count -ne 0) { throw 'The consecutive-loss rule reached the opportunistic self-hosted pool.' }
            if ($result.Log -notmatch "'Core' did not finish on its self-hosted runner \(job 9001\) and is rerun once on ubuntu-latest\.") { throw 'The hosted fallback is not annotated.' }
        }
        Assert-Case 'planner: a previous loss of the shard that passed on a rerun attempt does not refuse this rerun' {
            $scenario = New-E2eScenario 'lost-then-recovered'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss
            $route = "repos/$e2eRepository/actions/runs/6990/jobs?filter=all"
            $lost = $scenario.Routes[$route].jobs[0]; $lost | Add-Member -NotePropertyName run_attempt -NotePropertyValue 1
            $passed = Copy-Evidence $lost; $passed.id = 8951; $passed.run_attempt = 2; $passed.conclusion = 'success'
            $passed.check_run_url = "https://api.github.com/repos/$e2eRepository/check-runs/98051"
            $scenario.Routes[$route] = [pscustomobject]@{ jobs = @($lost, $passed) }
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 1 -or @($result.Plan.repeatedRunnerLoss).Count -ne 0) { throw 'A superseded attempt-1 loss refused the rerun.' }
        }
        Assert-Case 'planner: a previous loss stands when a partial rerun of that run did not include the shard' {
            $scenario = New-E2eScenario 'lost-other-rerun'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss
            $route = "repos/$e2eRepository/actions/runs/6990/jobs?filter=all"
            $lost = $scenario.Routes[$route].jobs[0]; $lost | Add-Member -NotePropertyName run_attempt -NotePropertyValue 1
            $other = Copy-Evidence $lost; $other.id = 8952; $other.run_attempt = 2; $other.name = 'Tests (net10.0) - Core B'; $other.conclusion = 'success'
            $scenario.Routes[$route] = [pscustomobject]@{ jobs = @($lost, $other) }
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 0 -or @($result.Plan.repeatedRunnerLoss).Count -ne 1) { throw 'A rerun of another shard erased this shard''s loss.' }
        }
        Assert-Case 'planner: a previous product failure beside a runner-loss annotation is not a previous runner loss' {
            $scenario = New-E2eScenario 'previous-failed-step'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss
            $route = "repos/$e2eRepository/actions/runs/6990/jobs?filter=all"
            $scenario.Routes[$route].jobs[0].steps[0].conclusion = 'failure'
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 1 -or @($result.Plan.repeatedRunnerLoss).Count -ne 0) { throw 'A previous failed step was counted as a previous runner loss.' }
        }
        Assert-Case 'planner: the previous run is found past a first page of cancelled runs' {
            $scenario = New-E2eScenario 'history-page-two'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss
            $older = @($scenario.Routes[$e2eHistory].workflow_runs | Where-Object { $_.run_number -ne 19 })
            $scenario.Routes[$e2eHistory] = [pscustomobject]@{ workflow_runs = @(1..50 | ForEach-Object {
                [pscustomobject]@{ id = 7100 + $_; run_number = 20; conclusion = 'cancelled'; event = 'push'; head_branch = 'master'; head_sha = ('4' * 40) } }) }
            $scenario.Routes["$e2eHistory&per_page=50&page=2"] = [pscustomobject]@{ workflow_runs = $older }
            $result = Invoke-E2ePlan $scenario
            if (@($result.Plan.alternate).Count -ne 0 -or @($result.Plan.repeatedRunnerLoss).Count -ne 1 -or
                $result.Plan.repeatedRunnerLoss[0].previousRunId -cne '6990') { throw 'The previous run on page two was not found.' }
        }
        Assert-Case 'planner: history that cannot be read is looked up once and refuses every hosted shard' {
            $scenario = New-E2eScenario 'history-unreadable-many' -ShardNames @('Core', 'Core B', 'Core C') -IdleRunners 3
            $scenario.Routes.Remove($e2eHistory)
            $result = Invoke-E2ePlan $scenario
            if (@($result.Calls | Where-Object { $_ -match 'workflows/55/runs' }).Count -ne 1) { throw "History was requested $(@($result.Calls | Where-Object { $_ -match 'workflows/55/runs' }).Count) times." }
            if (@($result.Plan.alternate).Count -ne 0 -or @($result.Plan.attempts | Where-Object {
                $_.reason -notmatch '^The previous run could not be read, so the shard is not rerun: ' }).Count -ne 0) { throw 'A shard was not refused with the cached lookup error.' }
        }

        Assert-Case 'reconciler: a hosted primary lost and a self-hosted alternate passed resolve to the alternate, visibly' {
            $scenario = New-E2eScenario 'resolve-hosted-to-self-hosted'
            $null = Invoke-E2ePlan $scenario
            $resolution = New-Resolution 'e2e-hosted-to-self-hosted' -Alternate -HostedPrimary -EnvironmentPolicy
            $scenario.Jobs.Add($resolution.AlternateJob)
            $result = Invoke-E2eResolve $scenario $resolution
            if (-not $result.Proof.complete -or @($result.Proof.shards).Count -ne 1 -or $result.Proof.shards[0].attempt -cne 'alternate' -or
                $result.Proof.shards[0].primaryJobId -cne '9001' -or $result.Proof.shards[0].chosenJobId -cne '9002') { throw 'The alternate attempt was not the resolved evidence.' }
            if ($result.Resolved.Count -ne 1 -or $result.Resolved[0].name -cne 'Tests (net10.0) - Core' -or
                (@($result.Resolved[0].labels) -join '|') -cne 'self-hosted|Linux|X64|aidotnet-cpu') { throw 'The resolved job is not the self-hosted alternate.' }
            if ($result.Log -notmatch "::warning title=Shard recovered by rerun::'Core' did not finish on its first runner \(job 9001\) and passed only on its rerun \(job 9002\)\.") { throw 'The recovered shard is not annotated.' }
            if ($result.Summary -notmatch '1 shard\(s\) passed only after a rerun' -or $result.Summary -notmatch '\| Core \| 9001 \| 9002 \|') { throw 'The recovered shard is not in the step summary.' }
        }
        Assert-Case 'reconciler: the same recovery is rejected when the primary also carries an unlisted failure' {
            $scenario = New-E2eScenario 'resolve-loss-plus-exit-code'
            $null = Invoke-E2ePlan $scenario
            $resolution = New-Resolution 'e2e-loss-plus-exit-code' -Alternate -HostedPrimary -EnvironmentPolicy
            $scenario.Jobs.Add($resolution.AlternateJob)
            $scenario.Routes["repos/$e2eRepository/check-runs/99001/annotations"] = @(
                [pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = $hostedLoss }
                [pscustomobject]@{ annotation_level = 'failure'; path = '.github'; start_line = 1; message = 'System.OutOfMemoryException: Exception of type was thrown' })
            $errors = Get-E2eResolveErrors $scenario $resolution
            if ($errors -notmatch '(?m)^Core: Primary test/compiler/unknown failure cannot be recovered\.$') { throw "Rejected for another reason: $errors" }
        }
        Assert-Case 'reconciler: an alternate that ran in the primary''s own pool is rejected' {
            $scenario = New-E2eScenario 'resolve-same-pool'
            $null = Invoke-E2ePlan $scenario
            $resolution = New-Resolution 'e2e-same-pool' -Alternate -HostedPrimary -EnvironmentPolicy
            $resolution.AlternateJob.labels = @('ubuntu-latest')
            $scenario.Jobs.Add($resolution.AlternateJob)
            $errors = Get-E2eResolveErrors $scenario $resolution
            if ($errors -notmatch '(?m)^Core: Alternate pool/trust/OS/architecture/definition binding mismatch\.$') { throw "Rejected for another reason: $errors" }
        }
        Assert-Case 'reconciler: a shard the planner refused cannot be resolved by an alternate that ran anyway' {
            $scenario = New-E2eScenario 'resolve-refused'
            Add-E2ePreviousRun $scenario 'Tests (net10.0) - Core' $hostedLoss
            $null = Invoke-E2ePlan $scenario
            $resolution = New-Resolution 'e2e-refused' -Alternate -HostedPrimary -EnvironmentPolicy
            $scenario.Jobs.Add($resolution.AlternateJob)
            $errors = Get-E2eResolveErrors $scenario $resolution
            if ($errors -notmatch '(?m)^Core: No single authorized alternate attempt exists\.$') { throw "Rejected for another reason: $errors" }
        }
    }
    finally {
        Set-Item Function:Invoke-GitHubApiWithRetry -Value $priorRetryE2e
        foreach ($key in $e2ePriorEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key, $e2ePriorEnvironment[$key]) }
    }

    Assert-Case 'previous-run search: the result cap without an eligible run is unreadable history, not empty history' {
        $run = [pscustomobject]@{ workflow_id = 55; event = 'push'; head_branch = 'master'; run_number = 5000 }
        $capped = @(1..$script:FailoverRunSearchCap | ForEach-Object {
            [pscustomobject]@{ id = $_; run_number = 4999; conclusion = 'cancelled'; event = 'push'; head_branch = 'master' } })
        $priorApi = (Get-Item Function:Invoke-FailoverApi).ScriptBlock
        try {
            Set-Item Function:Invoke-FailoverApi -Value { param([string] $Path, [switch] $Paginate, [string] $Token = '', [scriptblock] $Until = $null) $capped }.GetNewClosure()
            $threw = $false
            try { $null = Get-FailoverPreviousRun $run 'ooples/AiDotNet' } catch { $threw = $_.Exception.Message -match 'the most GitHub returns' }
            if (-not $threw) { throw 'A capped search was read as an empty history.' }
            $short = @($capped | Select-Object -First 10)
            Set-Item Function:Invoke-FailoverApi -Value { param([string] $Path, [switch] $Paginate, [string] $Token = '', [scriptblock] $Until = $null) $short }.GetNewClosure()
            if ($null -ne (Get-FailoverPreviousRun $run 'ooples/AiDotNet')) { throw 'A verified empty history must be $null.' }
        }
        finally { Set-Item Function:Invoke-FailoverApi -Value $priorApi }
    }
    Assert-Case 'paginated Invoke-FailoverApi stops after the page that satisfies -Until and reads workflow_runs' {
        . "$PSScriptRoot/Invoke-GitHubApiWithRetry.ps1"
        $priorRetry = (Get-Item Function:Invoke-GitHubApiWithRetry).ScriptBlock
        try {
            Set-Item Function:Invoke-GitHubApiWithRetry -Value {
                param([string] $Uri, [hashtable] $Headers)
                if ($Uri -match 'page=1$') { return [pscustomobject]@{ total_count = 120; workflow_runs = @(1..50 | ForEach-Object { [pscustomobject]@{ id = $_ } }) } }
                if ($Uri -match 'page=2$') { return [pscustomobject]@{ total_count = 120; workflow_runs = @(51..100 | ForEach-Object { [pscustomobject]@{ id = $_ } }) } }
                throw "Unexpected pagination URI '$Uri'."
            }
            $paged = @(& $originalApi -Path 'repos/ooples/AiDotNet/actions/workflows/55/runs' -Paginate -Token 'test-token' -Until { param($ReadSoFar) @($ReadSoFar | Where-Object { $_.id -eq 75 }).Count -gt 0 })
            if ($paged.Count -ne 100 -or $paged[0].id -ne 1 -or $paged[99].id -ne 100) { throw "Expected 100 flat runs; got $($paged.Count)." }
        }
        finally { Set-Item Function:Invoke-GitHubApiWithRetry -Value $priorRetry }
    }
    Assert-Case 'paginated Invoke-FailoverApi returns flat items under @(...) across pages' {
        . "$PSScriptRoot/Invoke-GitHubApiWithRetry.ps1"
        $priorRetry = (Get-Item Function:Invoke-GitHubApiWithRetry).ScriptBlock
        try {
            Set-Item Function:Invoke-GitHubApiWithRetry -Value {
                param([string] $Uri, [hashtable] $Headers)
                if ($Uri -match 'page=1$') {
                    return [pscustomobject]@{ jobs = @(1..50 | ForEach-Object { [pscustomobject]@{ id = $_; name = "job-$_"; status = 'completed' } }) }
                }
                if ($Uri -match 'page=2$') {
                    return [pscustomobject]@{ jobs = @(51..62 | ForEach-Object { [pscustomobject]@{ id = $_; name = "job-$_"; status = 'completed' } }) }
                }
                throw "Unexpected pagination URI '$Uri'."
            }
            $paged = @(& $originalApi -Path 'repos/ooples/AiDotNet/actions/runs/1/attempts/1/jobs' -Paginate -Token 'test-token')
            if ($paged.Count -ne 62 -or $paged[0] -is [System.Array] -or $paged[0].name -cne 'job-1' -or $paged[61].name -cne 'job-62') {
                throw "Expected 62 flat job objects; got count=$($paged.Count), firstType=$($paged[0].GetType().FullName)."
            }
        }
        finally {
            Set-Item Function:Invoke-GitHubApiWithRetry -Value $priorRetry
        }
    }
}
finally {
    Set-Item Function:Invoke-FailoverApi -Value $originalApi
    $resolvedScratch = [IO.Path]::GetFullPath($scratch)
    $resolvedTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedScratch.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase)) { throw 'Refusing scratch cleanup outside temporary directory.' }
    Remove-Item -LiteralPath $resolvedScratch -Recurse -Force
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Host "FAIL $failure" }
    throw "$($failures.Count) of $caseCount runner failover cases failed."
}
Write-Host "Runner failover proof passed ($caseCount cases)."
