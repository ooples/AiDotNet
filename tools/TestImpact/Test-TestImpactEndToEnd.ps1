<#
.SYNOPSIS
    Proves a certified map selects a strict subset in a real git diff and fails closed for infra.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$selector = Join-Path $PSScriptRoot 'Select-Shards.ps1'
$coverageSelector = Join-Path $PSScriptRoot 'Select-CoverageShards.ps1'
$missMeasurer = Join-Path $PSScriptRoot 'Measure-SelectionMiss.ps1'
$certificateWriter = Join-Path $PSScriptRoot 'New-ShardMapCertificate.ps1'
$certificateValidator = Join-Path $PSScriptRoot 'Test-CertifiedShardMap.ps1'
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
            -CertificationRunId 102 -OutDirectory certified
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
            -OutcomesFile outcomes.json -CandidateMapRunId 100 -AuditSourceRunId 101 `
            -AuditSourceSha $baseSha -CertificationRunId 105 -OutDirectory unchanged-certified
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
            -CertificationRunId 103 -OutDirectory selected-failure-certified
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
            -CertificationRunId 104 -OutDirectory missed-failure-certified
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

Write-Host 'Test-impact end-to-end proof passed: covered edit selected 2/3; #2118 selected none; CI control edit escalated.'
exit 0
