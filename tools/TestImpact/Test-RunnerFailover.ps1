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
function New-Resolution([string] $Name, [switch] $Alternate) {
    $e = New-Evidence
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
    $receipt = New-Receipt $e $attempt
    if ($Alternate) { $receipt.runnerName = $alternateJob.runner_name }
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
