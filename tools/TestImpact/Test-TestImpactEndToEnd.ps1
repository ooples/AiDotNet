<#
.SYNOPSIS
    Proves a certified map selects a strict subset in a real git diff and fails closed for infra.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

& (Join-Path $PSScriptRoot 'Test-CiImpactWorkflowReview.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Workflow review negative controls failed.' }

$selector = Join-Path $PSScriptRoot 'Select-Shards.ps1'
$coverageSelector = Join-Path $PSScriptRoot 'Select-CoverageShards.ps1'
$missMeasurer = Join-Path $PSScriptRoot 'Measure-SelectionMiss.ps1'
$certificateWriter = Join-Path $PSScriptRoot 'New-ShardMapCertificate.ps1'
$certificateValidator = Join-Path $PSScriptRoot 'Test-CertifiedShardMap.ps1'
$reuseResolver = Join-Path $PSScriptRoot 'Resolve-CiValidationReuse.ps1'
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ("aidotnet-impact-e2e-" + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
$fixtureFailure = $null

function Assert-True {
    param([bool] $Condition, [string] $What)
    if (-not $Condition) { [void] $failures.Add($What) }
}

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments)] [string[]] $Arguments)

    $output = & git @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $details = ($output | Out-String).Trim()
        throw "git $($Arguments -join ' ') failed with exit code $exitCode$(if ($details) { ": $details" })"
    }
    return $output
}

function Assert-CertificateRejected {
    param([string] $CertificateFile, [string] $ExpectedPattern, [string] $What)

    $message = $null
    try {
        & $certificateValidator -MapFile certified/shard-map.json -CertificateFile $CertificateFile `
            -CertificationRunId 102
    }
    catch { $message = [string] $_.Exception.Message }

    if ($null -eq $message) {
        [void] $failures.Add($What)
    }
    elseif ($message -notmatch $ExpectedPattern) {
        [void] $failures.Add("$What (rejected for the wrong reason: $message)")
    }
}

try {
    New-Item -ItemType Directory -Path (Join-Path $fixture 'src') -Force | Out-Null
    Push-Location $fixture
    try {
        Invoke-Git init --quiet
        Invoke-Git config user.email 'ci-impact-fixture@example.invalid'
        Invoke-Git config user.name 'CI impact fixture'
        Invoke-Git config commit.gpgSign false
        $disabledHooks = Join-Path $fixture 'disabled-hooks'
        New-Item -ItemType Directory -Path $disabledHooks -Force | Out-Null
        Invoke-Git config core.hooksPath $disabledHooks
        @('one', 'alpha before', 'three', 'beta unchanged', 'five') |
            Set-Content -LiteralPath src/Feature.cs -Encoding utf8
        Invoke-Git add src/Feature.cs
        Invoke-Git commit --quiet -m baseline
        $baseSha = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()
        if ($baseSha -cnotmatch '^[0-9a-f]{40}$') {
            throw "fixture HEAD is not a full lowercase commit ID: '$baseSha'"
        }

        $map = [ordered]@{
            schemaVersion = 1
            sha = $baseSha
            generatedUtc = '2026-09-09T00:00:00Z'
            knownShards = @('Alpha', 'Beta')
            alwaysRun = @('Always')
            files = [ordered]@{
                'src/Feature.cs' = @(
                    [ordered]@{ s = 0; r = @(2, 2) },
                    [ordered]@{ s = 1; r = @(4, 4) }
                )
            }
        }
        $map | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath shard-map.json -Encoding utf8

        # Exercise the production coverage splitter against a generated repository/map fixture,
        # not a hand-edited workflow result. The map deliberately has one mapped no-coverage
        # shard, one natural producer, and one previous always-run shard so all three typed states
        # must appear in a single output partition.
        & $coverageSelector -PreviousMap shard-map.json -ChangedFiles @() `
            -AllShards @('Alpha', 'Beta', 'Always') -NoCoverageShards @('Alpha', 'Always') `
            -CarryForwardDirectory carried-fixture -OutFile coverage-split.json
        Assert-True ($LASTEXITCODE -eq 0) 'the generated coverage split fixture failed'
        $coverageSplit = Get-Content coverage-split.json -Raw | ConvertFrom-Json
        Assert-True ([int] $coverageSplit.schemaVersion -eq 1) `
            'the coverage split omitted its transport schema'
        Assert-True ((@($coverageSplit.instrument) -join ',') -eq 'Beta') `
            'the natural coverage producer did not receive Instrument'
        Assert-True ((@($coverageSplit.carried) -join ',') -eq 'Alpha') `
            'the byte-identical no-coverage shard did not receive Carried'
        Assert-True ((@($coverageSplit.runWithoutCoverage) -join ',') -eq 'Always') `
            'the previous always-run no-coverage shard did not receive RunWithoutCoverage'
        Assert-True (Test-Path -LiteralPath carried-fixture/Alpha.digest.json) `
            'the Carried disposition did not reconstruct its digest'
        Assert-True (-not (Test-Path -LiteralPath carried-fixture/Always.digest.json)) `
            'RunWithoutCoverage incorrectly synthesized a digest that the shard never produced'

        # Without a previous map there is no evidence authorizing either non-instrumented state.
        # The bootstrap remains fail-closed even for a shard whose NAME resembles the fixture's
        # always-run member.
        & $coverageSelector -PreviousMap absent-map.json -ChangedFiles @() `
            -AllShards @('Alpha', 'Beta', 'Always') -NoCoverageShards @('Alpha', 'Always') `
            -OutFile bootstrap-coverage-split.json
        $bootstrapSplit = Get-Content bootstrap-coverage-split.json -Raw | ConvertFrom-Json
        Assert-True (@($bootstrapSplit.instrument).Count -eq 3 -and
            @($bootstrapSplit.carried).Count -eq 0 -and
            @($bootstrapSplit.runWithoutCoverage).Count -eq 0) `
            'coverage bootstrap did not fail closed to Instrument for every shard'

        # A tests/** change invalidates a mapped carry because test code is absent from coverage
        # digests. It still cannot authorize retrying known-unsafe instrumentation: that shard
        # remains always-run and its complete correctness suite will execute without coverage.
        & $coverageSelector -PreviousMap shard-map.json `
            -ChangedFiles @('tests/AiDotNet.Tests/NewCoverageEdge.cs') `
            -AllShards @('Alpha', 'Beta', 'Always') -NoCoverageShards @('Alpha', 'Always') `
            -OutFile dirty-coverage-split.json
        $dirtyCoverageSplit = Get-Content dirty-coverage-split.json -Raw | ConvertFrom-Json
        Assert-True (@($dirtyCoverageSplit.instrument).Count -eq 2 -and
            $dirtyCoverageSplit.instrument -contains 'Alpha' -and
            $dirtyCoverageSplit.instrument -contains 'Beta') `
            'test-code invalidation did not re-instrument every coverage-capable shard'
        Assert-True ((@($dirtyCoverageSplit.runWithoutCoverage) -join ',') -eq 'Always') `
            'test-code invalidation retried instrumentation already known to exceed the runner envelope'

        @(
            [ordered]@{ shard = 'Alpha'; outcome = 'success' },
            [ordered]@{ shard = 'Beta'; outcome = 'success' },
            [ordered]@{ shard = 'Always'; outcome = 'success' }
        ) | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath outcomes.json -Encoding utf8
        [ordered]@{
            Escalated = $false
            TotalShards = 3
            WouldRun = 2
            WouldSkip = 1
            WouldRunShards = @('Alpha', 'Always')
            WouldSkipShards = @('Beta')
            Failed = 0
            Missed = @()
            MissCount = 0
        } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath audit.json -Encoding utf8

        & $certificateWriter -MapFile shard-map.json -AuditFile audit.json -OutcomesFile outcomes.json `
            -CandidateMapRunId 100 -AuditSourceRunId 101 `
            -AuditSourceSha 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' `
            -CertificationRunId 102 -Basis HistoricalReplay -OutDirectory certified `
            -DecisionFile certified-decision.json
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath certified/certification.json)) {
            throw 'the real certification policy did not produce a certificate for the clean complete audit'
        }

        $global:LASTEXITCODE = 97
        try {
            & $certificateValidator -MapFile certified/shard-map.json `
                -CertificateFile certified/certification.json `
                -CertificationRunId 102
            if ($LASTEXITCODE -ne 0) {
                throw "validator returned exit code $LASTEXITCODE"
            }
        }
        catch {
            [void] $failures.Add("the valid end-to-end certificate was rejected: $($_.Exception.Message)")
        }

        # The generated repository is still at the map's exact commit. Normal PR selection must
        # treat an unexpectedly empty diff as unsafe, while the nightly audit must be able to test
        # that exact tree without pretending it changed. This is the bootstrap case that previously
        # escalated forever and made production certification impossible.
        & $selector -MapFile certified/shard-map.json -ExpectedShards @('Alpha', 'Beta', 'Always') `
            -OutFile unchanged-pr-selection.json
        $unchangedPr = Get-Content unchanged-pr-selection.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $unchangedPr.escalate) `
            'ordinary selection did not fail closed for an unexpectedly empty diff'

        & $missMeasurer -SelectorPath $selector -MapFile certified/shard-map.json `
            -OutcomesFile outcomes.json -OutFile unchanged-audit.json
        Assert-True ($LASTEXITCODE -eq 0) 'the identical-tree selection audit failed'
        $unchangedAudit = Get-Content unchanged-audit.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $unchangedAudit.Escalated) `
            'the identical-tree selection audit escalated instead of exercising reduction'
        Assert-True (($unchangedAudit.WouldRunShards -join ',') -eq 'Always') `
            'the identical-tree audit did not select exactly the always-run shard'
        Assert-True (($unchangedAudit.WouldSkipShards -join ',') -eq 'Alpha,Beta') `
            'the identical-tree audit did not record the expected skipped shards'

        & $certificateWriter -MapFile certified/shard-map.json -AuditFile unchanged-audit.json `
            -OutcomesFile outcomes.json -CandidateMapRunId 105 -AuditSourceRunId 101 `
            -AuditSourceSha $baseSha -CertificationRunId 105 -Basis FreshCoverage `
            -OutDirectory unchanged-certified `
            -DecisionFile unchanged-decision.json
        Assert-True ($LASTEXITCODE -eq 0 -and
            (Test-Path -LiteralPath unchanged-certified/certification.json)) `
            'the policy did not certify the clean identical-tree audit'

        @('one', 'alpha after', 'three', 'beta unchanged', 'five') |
            Set-Content -LiteralPath src/Feature.cs -Encoding utf8
        Invoke-Git add src/Feature.cs
        Invoke-Git commit --quiet -m narrow-change
        & $selector -MapFile certified/shard-map.json -ExpectedShards @('Alpha', 'Beta', 'Always') `
            -OutFile selection.json
        $selection = Get-Content selection.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $selection.escalate) 'a covered one-line edit escalated'
        Assert-True (@($selection.shards).Count -eq 2) 'the covered edit did not select a strict 2/3 subset'
        Assert-True ($selection.shards -contains 'Alpha') 'the covering shard was omitted'
        Assert-True ($selection.shards -contains 'Always') 'the always-run shard was omitted'
        Assert-True (-not ($selection.shards -contains 'Beta')) 'an unaffected shard was selected'

        # Prove the real selector -> miss measurement -> certificate chain in both directions. A
        # failure in Alpha was selected and therefore remains certifiable; the same failure in Beta
        # was skipped and must make Measure-SelectionMiss fail and suppress certificate creation.
        @(
            [ordered]@{ shard = 'Alpha'; outcome = 'failure' },
            [ordered]@{ shard = 'Beta'; outcome = 'success' },
            [ordered]@{ shard = 'Always'; outcome = 'success' }
        ) | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath selected-failure-outcomes.json -Encoding utf8
        & $missMeasurer -SelectorPath $selector -MapFile certified/shard-map.json `
            -OutcomesFile selected-failure-outcomes.json -OutFile selected-failure-audit.json
        Assert-True ($LASTEXITCODE -eq 0) 'a selected shard failure was incorrectly reported as a miss'
        & $certificateWriter -MapFile certified/shard-map.json -AuditFile selected-failure-audit.json `
            -OutcomesFile selected-failure-outcomes.json -CandidateMapRunId 100 `
            -AuditSourceRunId 101 -AuditSourceSha 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' `
            -CertificationRunId 103 -Basis HistoricalReplay -OutDirectory selected-failure-certified `
            -DecisionFile selected-failure-decision.json
        Assert-True ($LASTEXITCODE -eq 0 -and
            (Test-Path -LiteralPath selected-failure-certified/certification.json)) `
            'the end-to-end chain rejected a complete audit whose only failure was selected'
        & $certificateValidator -MapFile selected-failure-certified/shard-map.json `
            -CertificateFile selected-failure-certified/certification.json `
            -CertificationRunId 103
        Assert-True ($LASTEXITCODE -eq 0) `
            'the validator rejected a generated certificate with a selected failure'

        @(
            [ordered]@{ shard = 'Alpha'; outcome = 'success' },
            [ordered]@{ shard = 'Beta'; outcome = 'failure' },
            [ordered]@{ shard = 'Always'; outcome = 'success' }
        ) | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath missed-failure-outcomes.json -Encoding utf8
        $expectedMissOutput = @(& $missMeasurer -SelectorPath $selector `
            -MapFile certified/shard-map.json -OutcomesFile missed-failure-outcomes.json `
            -OutFile missed-failure-audit.json 6>&1)
        $expectedMissExit = $LASTEXITCODE
        foreach ($line in $expectedMissOutput) {
            Write-Host (('[expected miss] ' + [string] $line) -replace '::error::', '')
        }
        Assert-True ($expectedMissExit -eq 1) 'a skipped shard failure did not fail the miss audit'
        & $certificateWriter -MapFile certified/shard-map.json -AuditFile missed-failure-audit.json `
            -OutcomesFile missed-failure-outcomes.json -CandidateMapRunId 100 `
            -AuditSourceRunId 101 -AuditSourceSha 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' `
            -CertificationRunId 104 -Basis HistoricalReplay -OutDirectory missed-failure-certified `
            -DecisionFile missed-failure-decision.json
        Assert-True ($LASTEXITCODE -eq 0 -and
            -not (Test-Path -LiteralPath missed-failure-certified/certification.json)) `
            'the end-to-end chain certified a map after selection skipped a failing shard'

        # Return to the certified map's exact tree so this diff is precisely the three-file #2118
        # counterexample, not accidentally mixed with the covered source edit used above.
        Invoke-Git checkout --quiet --detach $baseSha
        New-Item -ItemType Directory -Path .github/workflows -Force | Out-Null
        'release setup' | Set-Content -LiteralPath .github/AUTOMATED_RELEASE_SETUP.md -Encoding utf8
        'versioning' | Set-Content -LiteralPath .github/VERSIONING.md -Encoding utf8
        'name: release-please' | Set-Content -LiteralPath .github/workflows/release-please.yml -Encoding utf8
        Invoke-Git add .github/AUTOMATED_RELEASE_SETUP.md .github/VERSIONING.md `
            .github/workflows/release-please.yml
        Invoke-Git commit --quiet -m non-runtime-change
        & $selector -ClassifyOnly -BaseSha $baseSha -OutFile non-runtime-classification.json
        Assert-True ($LASTEXITCODE -eq 0) 'the map-independent #2118 classifier failed'
        $nonRuntimeClassification = Get-Content non-runtime-classification.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $nonRuntimeClassification.requiresValidation) `
            'the map-independent #2118 classifier required runtime validation'
        Assert-True (@($nonRuntimeClassification.changedPaths).Count -eq 3) `
            'the map-independent #2118 classifier did not inspect exactly three paths'
        & $selector -MapFile certified/shard-map.json `
            -ExpectedShards @('Alpha', 'Beta', 'Always') -OutFile non-runtime-selection.json
        Assert-True ($LASTEXITCODE -eq 0) 'the exact #2118 path set made the selector fail'
        $nonRuntime = Get-Content non-runtime-selection.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $nonRuntime.escalate) 'the exact #2118 path set escalated'
        Assert-True (-not [bool] $nonRuntime.requiresValidation) `
            'the exact #2118 path set did not suppress runtime validation'
        Assert-True (@($nonRuntime.shards).Count -eq 0) `
            'the exact #2118 path set selected model/test shards'

        # A real validation-control-plane edit mixed into the same otherwise non-runtime change set
        # must reverse that decision and fail closed to full validation.
        'name: changed-main-ci' | Set-Content -LiteralPath .github/workflows/sonarcloud.yml -Encoding utf8
        Invoke-Git add .github/workflows/sonarcloud.yml
        Invoke-Git commit --quiet -m infrastructure-change
        $expectedEscalationOutput = @(& $selector -MapFile certified/shard-map.json `
            -ExpectedShards @('Alpha', 'Beta', 'Always') -OutFile infrastructure-selection.json 6>&1)
        $expectedEscalationExit = $LASTEXITCODE
        foreach ($line in $expectedEscalationOutput) {
            Write-Host (('[expected escalation] ' + [string] $line) -replace '::warning::', '')
        }
        Assert-True ($expectedEscalationExit -eq 0) 'an infrastructure edit made the selector fail'
        $infrastructure = Get-Content infrastructure-selection.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $infrastructure.escalate) `
            'the main validation workflow did not fail closed to the full matrix'
        Assert-True ([bool] $infrastructure.requiresValidation) `
            'the main validation workflow did not require runtime validation'

        & $selector -ClassifyOnly -BaseSha 'ffffffffffffffffffffffffffffffffffffffff' `
            -OutFile invalid-base-classification.json
        $invalidBase = Get-Content invalid-base-classification.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $invalidBase.requiresValidation) `
            'an unresolvable PR base authorized the non-runtime path'

        # --no-renames is a safety boundary: source renamed to a Markdown destination must expose
        # both paths, not classify from only the apparently harmless destination suffix.
        Invoke-Git checkout --quiet --detach $baseSha
        New-Item -ItemType Directory -Path docs -Force | Out-Null
        Invoke-Git mv src/Feature.cs docs/Feature.md
        Invoke-Git commit --quiet -m source-to-markdown-rename
        & $selector -ClassifyOnly -BaseSha $baseSha -OutFile rename-classification.json
        $renameClassification = Get-Content rename-classification.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $renameClassification.requiresValidation) `
            'a source-to-Markdown rename incorrectly suppressed runtime validation'
        Assert-True (@($renameClassification.changedPaths).Count -eq 2) `
            'source-to-Markdown rename classification did not expose both sides'
        & $selector -MapFile certified/shard-map.json `
            -ExpectedShards @('Alpha', 'Beta', 'Always') -OutFile rename-selection.json
        $renameSelection = Get-Content rename-selection.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $renameSelection.escalate) `
            'a mapped source-to-Markdown rename unexpectedly escalated'
        Assert-True ([bool] $renameSelection.requiresValidation) `
            'normal shard selection hid the source side of a source-to-Markdown rename'
        Assert-True (@($renameSelection.shards).Count -eq 3) `
            'normal shard selection did not retain both mapped source shards and always-run validation'

        # ---- A pull request BEHIND master, through a real GitHub-shaped merge commit. -------------
        # Master gains a selector-control commit after the pull request branched from the map's
        # commit. The event's base.sha still names that old commit, which is exactly how #2100 was
        # charged with 16 merged CI-control files and escalated to all 116 shards.
        Invoke-Git checkout --quiet --detach $baseSha
        New-Item -ItemType Directory -Path tools/TestImpact -Force | Out-Null
        'merged selector fix' | Set-Content -LiteralPath tools/TestImpact/Merged.ps1 -Encoding utf8
        Invoke-Git add tools/TestImpact/Merged.ps1
        Invoke-Git commit --quiet -m master-moves-on
        $masterSha = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()

        Invoke-Git checkout --quiet --detach $baseSha
        @('one', 'alpha behind', 'three', 'beta unchanged', 'five') |
            Set-Content -LiteralPath src/Feature.cs -Encoding utf8
        New-Item -ItemType Directory -Path tests/Proj/UnitTests/Alpha -Force | Out-Null
        @(
            'using Xunit;',
            'namespace Proj.UnitTests.Alpha;',
            'public class NewAlphaTests',
            '{',
            '    [Fact]',
            '    public void Works() { }',
            '}'
        ) | Set-Content -LiteralPath tests/Proj/UnitTests/Alpha/NewAlphaTests.cs -Encoding utf8
        Invoke-Git add src/Feature.cs tests/Proj/UnitTests/Alpha/NewAlphaTests.cs
        Invoke-Git commit --quiet -m pull-request-change
        $prHeadSha = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()

        # refs/pull/N/merge: base-branch tip first, pull request head second.
        Invoke-Git checkout --quiet --detach $masterSha
        Invoke-Git merge --quiet --no-ff --no-edit $prHeadSha
        $testedMergeSha = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()
        @(
            [ordered]@{ name = 'Alpha'; project = 'tests/Proj/Proj.csproj'; filter = 'FullyQualifiedName~UnitTests.Alpha' },
            [ordered]@{ name = 'Beta'; project = 'tests/Proj/Proj.csproj'; filter = 'FullyQualifiedName~UnitTests.Beta' },
            [ordered]@{ name = 'Always'; project = 'tests/Proj/Proj.csproj'; filter = 'Category=Heavy' }
        ) | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath shard-manifest.json -Encoding utf8
        $shardNames = @('Alpha', 'Beta', 'Always')

        # Reproduction first: the stale base, exactly as the workflow used to pass it.
        $staleOutput = @(& $selector -MapFile certified/shard-map.json -BaseSha $baseSha `
            -ExpectedShards $shardNames -OutFile behind-stale-selection.json 6>&1)
        foreach ($line in $staleOutput) {
            Write-Host (('[expected escalation, stale base] ' + [string] $line) -replace '::warning::', '')
        }
        $stale = Get-Content behind-stale-selection.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $stale.escalate -and (@($stale.reasons) -join ';') -like '*tools/TestImpact/Merged.ps1*') `
            'reproduction failed: the stale base did not charge master''s merged control file to the pull request'
        & $selector -ClassifyOnly -BaseSha $baseSha -OutFile behind-stale-classification.json
        $staleClassification = Get-Content behind-stale-classification.json -Raw | ConvertFrom-Json
        Assert-True (@($staleClassification.changedPaths).Count -eq 3) `
            'reproduction failed: the stale base did not report master''s file as a pull-request path'

        # Fixed: the merge commit's first parent.
        & $selector -ClassifyOnly -PullRequestHeadSha $prHeadSha -OutFile behind-classification.json
        $behindClassification = Get-Content behind-classification.json -Raw | ConvertFrom-Json
        Assert-True ((@($behindClassification.changedPaths | Sort-Object) -join ',') -eq
            'src/Feature.cs,tests/Proj/UnitTests/Alpha/NewAlphaTests.cs') `
            'a behind pull request was classified on paths other than its own'
        Assert-True ([string] $behindClassification.baseSha -eq $masterSha) `
            'classification did not use the merge commit''s first parent as the base'
        & $selector -MapFile certified/shard-map.json -PullRequestHeadSha $prHeadSha `
            -ShardManifestFile shard-manifest.json -ExpectedShards $shardNames -OutFile behind-selection.json
        Assert-True ($LASTEXITCODE -eq 0) 'selection for a behind pull request failed'
        $behind = Get-Content behind-selection.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $behind.escalate) `
            "a behind pull request still escalated: $(@($behind.reasons) -join '; ')"
        Assert-True ((@($behind.shards) -join ',') -eq 'Alpha,Always') `
            "a behind pull request did not select exactly its covering/new-test shard and always-run (a strict 2/3): $(@($behind.shards) -join ',')"
        Assert-True (@($behind.routes | Where-Object { $_ -like 'Alpha <= runs tests affected by tests/Proj/UnitTests/Alpha/NewAlphaTests.cs*' }).Count -eq 1) `
            'the new test file was not routed to Alpha through its filter'

        # The head must be the merge commit's second parent, and the checkout must be a merge.
        & $selector -MapFile certified/shard-map.json -PullRequestHeadSha $masterSha `
            -ShardManifestFile shard-manifest.json -ExpectedShards $shardNames -OutFile wrong-head-selection.json 6>$null
        $wrongHead = Get-Content wrong-head-selection.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $wrongHead.escalate) 'a head that is not the merge commit''s second parent was trusted'
        & $selector -ClassifyOnly -PullRequestHeadSha $masterSha -OutFile wrong-head-classification.json 6>$null
        Assert-True ([bool] (Get-Content wrong-head-classification.json -Raw | ConvertFrom-Json).requiresValidation) `
            'classification trusted a head that is not the merge commit''s second parent'
        Invoke-Git checkout --quiet --detach $prHeadSha
        & $selector -MapFile certified/shard-map.json -PullRequestHeadSha $prHeadSha `
            -ShardManifestFile shard-manifest.json -ExpectedShards $shardNames -OutFile non-merge-selection.json 6>$null
        Assert-True ([bool] (Get-Content non-merge-selection.json -Raw | ConvertFrom-Json).escalate) `
            'a checkout that is not a merge commit was trusted as a pull-request merge'

        # A pull request that DELETES a test file: no file at HEAD names its types, so the reference
        # search's git grep exits 1. The selector must still exit 0 - callers read a nonzero exit as
        # a selector failure and throw away the (valid) selection for the full matrix.
        # Its own small repository, because the file must exist in the MAP's commit for its deletion
        # to be a map-coordinate change at all.
        $deletionRepo = Join-Path $fixture 'deletion-repo'
        New-Item -ItemType Directory -Path (Join-Path $deletionRepo 'tests/Proj/UnitTests/Alpha') -Force | Out-Null
        Push-Location $deletionRepo
        try {
            Invoke-Git init --quiet
            Invoke-Git config user.email 'ci-impact-fixture@example.invalid'
            Invoke-Git config user.name 'CI impact fixture'
            Invoke-Git config commit.gpgSign false
            Invoke-Git config core.hooksPath $disabledHooks
            @('namespace Proj.UnitTests.Alpha;', 'public class DoomedTests', '{', '    [Fact]',
              '    public void Works() { }', '}') |
                Set-Content -LiteralPath tests/Proj/UnitTests/Alpha/DoomedTests.cs -Encoding utf8
            Invoke-Git add .
            Invoke-Git commit --quiet -m map-commit
            $deletionBase = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()
            Invoke-Git rm --quiet tests/Proj/UnitTests/Alpha/DoomedTests.cs
            Invoke-Git commit --quiet -m delete-a-test-file
            $deletingHeadSha = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()
            Invoke-Git checkout --quiet --detach $deletionBase
            Invoke-Git commit --quiet --allow-empty -m base-moves-on
            Invoke-Git merge --quiet --no-ff --no-edit $deletingHeadSha
            [ordered]@{
                schemaVersion = 1; sha = $deletionBase; knownShards = @('Alpha', 'Beta'); alwaysRun = @('Always')
                files = [ordered]@{ 'src/Placeholder.cs' = @([ordered]@{ s = 1; r = @(1, 1) }) }
            } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath deletion-map.json -Encoding utf8
            $global:LASTEXITCODE = 99
            & $selector -MapFile deletion-map.json -PullRequestHeadSha $deletingHeadSha `
                -ShardManifestFile ../shard-manifest.json -ExpectedShards $shardNames `
                -OutFile deleted-test-selection.json 6>$null
            $deletedExit = $LASTEXITCODE
            $deletedTest = Get-Content deleted-test-selection.json -Raw | ConvertFrom-Json
        }
        finally { Pop-Location }
        Assert-True (-not [bool] $deletedTest.escalate -and $deletedTest.shards -contains 'Alpha') `
            "deleting a test file was not routed to its shard: $(@($deletedTest.reasons) -join '; ')"
        Assert-True ($deletedExit -eq 0) `
            "a valid selection exited $deletedExit, which the workflow reads as a selector failure"

        # Every result shape - escalations included - carries routes; consumers read it under StrictMode.
        & $selector -MapFile absent-map.json -ExpectedShards $shardNames -OutFile escalated-selection.json 6>$null
        $escalated = Get-Content escalated-selection.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $escalated.escalate -and $null -ne $escalated.PSObject.Properties['routes']) `
            'an escalated selection result has no routes property'

        # ---- Post-merge DELTA reuse, from the same pull request. --------------------------------
        # The pull request was validated as $testedMergeSha and its run executed Alpha and Always.
        # Before it lands, master gains more commits. Exact-tree reuse can never match now; delta
        # reuse must rebuild the validated tree from its parents and decide from what master added.
        $testedTree = ((Invoke-Git rev-parse "$testedMergeSha^{tree}") | Out-String).Trim()
        $pullRequestShards = '["Alpha","Always"]'
        function Invoke-DeltaPlanFixture([string] $Name, [string] $Tree = $testedTree) {
            & $reuseResolver -PlanDelta -TestedBaseSha $masterSha -TestedHeadSha $prHeadSha -TestedTree $Tree `
                -PullRequestShardsJson $pullRequestShards -MapFile certified/shard-map.json `
                -ShardManifestFile shard-manifest.json -SelectorPath $selector -OutFile "$Name.json" 6>$null
            return Get-Content "$Name.json" -Raw | ConvertFrom-Json
        }

        # (a) Master edits a line only Beta executes. Δ selects Beta + Always; the pull request ran
        #     Alpha + Always, so only Always is re-run and Alpha's results are imported.
        Invoke-Git checkout --quiet --detach $masterSha
        @('one', 'alpha before', 'three', 'beta changed on master', 'five') |
            Set-Content -LiteralPath src/Feature.cs -Encoding utf8
        Invoke-Git commit --quiet -am master-runtime-change
        Invoke-Git merge --quiet --no-ff --no-edit $prHeadSha
        $plan = Invoke-DeltaPlanFixture 'delta-partial'
        Assert-True ($plan.mode -ceq 'Partial') "an overlapping master delta did not plan a partial re-run: $($plan.mode) - $($plan.why)"
        Assert-True ((@($plan.rerun) -join ',') -ceq 'Always') `
            "the partial re-run was not exactly the overlap: rerun=$(@($plan.rerun) -join ','); routes=$(@($plan.routes) -join ' | ')"
        Assert-True ((@($plan.import) -join ',') -ceq 'Alpha') `
            "the partial re-run did not import the untouched pull-request shard: import=$(@($plan.import) -join ',')"
        Assert-True ($plan.tree -ceq $testedTree) 'the validated tree was not rebuilt exactly from its parents'

        # (b) Master adds only documentation: nothing the pull request certified can have changed.
        Invoke-Git checkout --quiet --detach $masterSha
        New-Item -ItemType Directory -Path docs -Force | Out-Null
        'release notes' | Set-Content -LiteralPath docs/notes.md -Encoding utf8
        Invoke-Git add docs/notes.md
        Invoke-Git commit --quiet -m master-docs-change
        Invoke-Git merge --quiet --no-ff --no-edit $prHeadSha
        $plan = Invoke-DeltaPlanFixture 'delta-reuse'
        Assert-True ($plan.mode -ceq 'Reuse') "a documentation-only master delta was not reused outright: $($plan.mode) - $($plan.why)"

        # (c) Master changes CI control: the full matrix, whatever the pull request ran.
        Invoke-Git checkout --quiet --detach $masterSha
        'later selector change' | Set-Content -LiteralPath tools/TestImpact/Later.ps1 -Encoding utf8
        Invoke-Git add tools/TestImpact/Later.ps1
        Invoke-Git commit --quiet -m master-control-change
        Invoke-Git merge --quiet --no-ff --no-edit $prHeadSha
        $plan = Invoke-DeltaPlanFixture 'delta-control'
        Assert-True ($plan.mode -ceq 'None') 'a master delta containing a CI-control edit was reused'

        # (d) A tree that the parents do not rebuild to is never trusted.
        $plan = Invoke-DeltaPlanFixture 'delta-wrong-tree' (((Invoke-Git rev-parse "$baseSha^{tree}") | Out-String).Trim())
        Assert-True ($plan.mode -ceq 'None' -and $plan.why -like '*could not be rebuilt*') `
            'a validated tree that its parents do not rebuild to was trusted'

        $badCertificate = Get-Content certified/certification.json -Raw | ConvertFrom-Json
        $badCertificate.missCount = 1
        $badCertificate | ConvertTo-Json -Depth 5 | Set-Content bad-certification.json -Encoding utf8
        Assert-CertificateRejected -CertificateFile bad-certification.json `
            -ExpectedPattern 'selection miss' `
            -What 'a certificate recording a selection miss was accepted'

        $badCertificate = Get-Content certified/certification.json -Raw | ConvertFrom-Json
        $badCertificate.failedShards = -1
        $badCertificate | ConvertTo-Json -Depth 5 | Set-Content negative-failure-certification.json -Encoding utf8
        Assert-CertificateRejected -CertificateFile negative-failure-certification.json `
            -ExpectedPattern 'failedShards must be at least 0' `
            -What 'a certificate with a negative failure count was accepted'
    }
    finally {
        Pop-Location
    }
}
catch {
    $fixtureFailure = $_
    throw
}
finally {
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    if ($resolvedFixture.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedFixture).StartsWith('aidotnet-impact-e2e-', [StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        $message = "refusing to remove unexpected fixture path '$resolvedFixture'"
        if ($null -ne $fixtureFailure) { Write-Warning "$message; preserving the original failure" }
        else { throw $message }
    }
}

if ($failures.Count -gt 0) {
    Write-Host 'Test-impact end-to-end proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host 'Test-impact end-to-end proof passed: covered edit selected 2/3; #2118 selected none; CI control edit escalated; behind-master PR with a new test selected 2/3 instead of escalating.'
exit 0
