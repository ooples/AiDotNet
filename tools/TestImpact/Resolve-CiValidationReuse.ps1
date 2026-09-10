<#
.SYNOPSIS
    Resolves exact-tree CI evidence for a landed pull request.

.DESCRIPTION
    A Validation certificate suppresses build/test/model validation but deliberately reruns
    CodeQL and Sonar. A Complete certificate suppresses both stages. Every runtime certificate
    additionally requires the immutable analysis, outcome-ledger, and coverage artifacts that the
    landed run must promote. Missing, malformed, stale, or tree-mismatched evidence fails closed.
#>
[CmdletBinding(DefaultParameterSetName = 'Resolve')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $Repository,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $CommitSha,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $EventName,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $ExpectedBaseBranch,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $GitHubOutput,
    [Parameter(ParameterSetName = 'Resolve')] [ValidateRange(0, 120)] [int] $WaitMinutes = 40,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum CiValidationReuseScope {
    None
    Validation
    Complete
}

function ConvertTo-RequiredBooleanProperty {
    param($Object, [string] $Name)
    if (-not $Object.PSObject.Properties[$Name] -or $Object.$Name -isnot [bool]) {
        throw "certificate property '$Name' must be a JSON boolean"
    }
    return [bool] $Object.$Name
}

function Test-JsonInteger {
    param($Value)
    return $Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [short] -or $Value -is [ushort] -or
        $Value -is [int] -or $Value -is [uint] -or
        $Value -is [long] -or $Value -is [ulong]
}

function ConvertTo-CertificateEvidence {
    param(
        [Parameter(Mandatory)] $Certificate,
        [Parameter(Mandatory)] [long] $ExpectedRunId
    )

    foreach ($name in 'schemaVersion', 'runId', 'testedSha', 'event') {
        if (-not $Certificate.PSObject.Properties[$name]) { throw "certificate is missing $name" }
    }
    $schema = 0
    if (-not [int]::TryParse([string] $Certificate.schemaVersion, [ref] $schema)) {
        throw 'certificate schemaVersion must be an integer'
    }
    $runId = 0L
    if (-not [long]::TryParse([string] $Certificate.runId, [ref] $runId) -or
        $runId -ne $ExpectedRunId) {
        throw 'certificate runId does not match its workflow run'
    }
    $testedSha = [string] $Certificate.testedSha
    if ($testedSha -cnotmatch '^[0-9a-f]{40}$') {
        throw 'certificate testedSha must be a lowercase 40-character Git SHA'
    }
    $event = [string] $Certificate.event
    if ($event -cne 'pull_request' -and $event -cne 'merge_group') {
        throw "certificate has unsupported event '$event'"
    }
    $requiresValidation = $true

    $scope = [CiValidationReuseScope]::None
    $testedTree = ''
    switch ($schema) {
        1 {
            # Legacy v1 did not contain requiresValidation, but if one remains in artifact
            # retention it is conservatively runtime-scoped and complete only.
            $requiresValidation = $true
            if (-not $Certificate.PSObject.Properties['ciGateConclusion'] -or
                [string] $Certificate.ciGateConclusion -cne 'success') {
                throw 'legacy certificate did not pass CI Gate'
            }
            $scope = [CiValidationReuseScope]::Complete
        }
        2 {
            $requiresValidation = ConvertTo-RequiredBooleanProperty $Certificate 'requiresValidation'
            if (-not $Certificate.PSObject.Properties['ciGateConclusion'] -or
                [string] $Certificate.ciGateConclusion -cne 'success') {
                throw 'legacy certificate did not pass CI Gate'
            }
            $scope = [CiValidationReuseScope]::Complete
        }
        3 {
            $requiresValidation = ConvertTo-RequiredBooleanProperty $Certificate 'requiresValidation'
            if (-not (Test-JsonInteger $Certificate.schemaVersion) -or
                -not (Test-JsonInteger $Certificate.runId)) {
                throw 'schema-v3 certificate schemaVersion and runId must be JSON integers'
            }
            foreach ($name in 'scope', 'testedTree', 'gateConclusion') {
                if (-not $Certificate.PSObject.Properties[$name]) {
                    throw "schema-v3 certificate is missing $name"
                }
            }
            if ([string] $Certificate.gateConclusion -cne 'success') {
                throw 'schema-v3 certificate did not pass its gate'
            }
            $scopeText = [string] $Certificate.scope
            if (-not [Enum]::TryParse[CiValidationReuseScope]($scopeText, $false, [ref] $scope) -or
                $scope -eq [CiValidationReuseScope]::None) {
                throw "schema-v3 certificate has unsupported scope '$scopeText'"
            }
            $testedTree = [string] $Certificate.testedTree
            if ($testedTree -cnotmatch '^[0-9a-f]{40}$') {
                throw 'schema-v3 certificate testedTree must be a lowercase 40-character Git SHA'
            }
        }
        default { throw "unsupported certificate schemaVersion $schema" }
    }

    if (-not $testedTree -and $Certificate.PSObject.Properties['testedTree']) {
        $testedTree = [string] $Certificate.testedTree
        if ($testedTree -and $testedTree -cnotmatch '^[0-9a-f]{40}$') {
            throw 'legacy certificate testedTree is malformed'
        }
    }

    return [pscustomobject]@{
        Scope = $scope
        RunId = $runId
        TestedSha = $testedSha
        TestedTree = $testedTree
        RequiresValidation = $requiresValidation
    }
}

function Select-BestCiEvidence {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Evidence)

    $eligible = @($Evidence | Where-Object {
        $_.TreeMatches -and
        (-not $_.RequiresValidation -or ($_.HasAnalysis -and $_.HasCoverage -and $_.HasLedger))
    })
    if ($eligible.Count -eq 0) { return $null }
    return @($eligible | Sort-Object `
        @{ Expression = { [int] $_.Scope }; Descending = $true },
        @{ Expression = { [DateTimeOffset] $_.CreatedAt }; Descending = $true },
        @{ Expression = { [long] $_.RunId }; Descending = $true } |
        Select-Object -First 1)[0]
}

function Write-ReuseDecision {
    param(
        [Parameter(Mandatory)] [CiValidationReuseScope] $DecisionScope,
        [long] $RunId = 0,
        [long] $PrNumber = 0,
        [string] $TestedSha = '',
        [bool] $RequiresValidation = $true,
        [string] $Summary = ''
    )

    $reuse = $DecisionScope -ne [CiValidationReuseScope]::None
    $reuseQuality = $DecisionScope -eq [CiValidationReuseScope]::Complete
    $lines = @(
        "run_id=$(if ($RunId -gt 0) { $RunId } else { '' })",
        "pr_number=$(if ($PrNumber -gt 0) { $PrNumber } else { '' })",
        "tested_sha=$TestedSha",
        "reuse_scope=$($DecisionScope.ToString())",
        "reuse=$($reuse.ToString().ToLowerInvariant())",
        "reuse_quality=$($reuseQuality.ToString().ToLowerInvariant())",
        "reused_requires_validation=$($RequiresValidation.ToString().ToLowerInvariant())",
        "execute_validation=$(((-not $reuse)).ToString().ToLowerInvariant())",
        "execute_quality=$(((-not $reuseQuality)).ToString().ToLowerInvariant())"
    )
    $lines | Out-File -LiteralPath $GitHubOutput -Append -Encoding utf8
    if ($Summary -and $env:GITHUB_STEP_SUMMARY) {
        $Summary | Out-File -LiteralPath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8
    }
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $Message)
        if (-not $Condition) { [void] $failures.Add($Message) }
    }
    function Assert-Rejected { param([scriptblock] $Action, [string] $Message)
        $rejected = $false
        try { & $Action | Out-Null } catch { $rejected = $true }
        Assert-True $rejected $Message
    }

    $sha = '0123456789abcdef0123456789abcdef01234567'
    $tree = '89abcdef0123456789abcdef0123456789abcdef'
    $partialJson = [pscustomobject]@{
        schemaVersion = 3; scope = 'Validation'; runId = 10; testedSha = $sha
        testedTree = $tree; event = 'pull_request'; requiresValidation = $true
        gateConclusion = 'success'
    }
    $partial = ConvertTo-CertificateEvidence $partialJson 10
    Assert-True ($partial.Scope -eq [CiValidationReuseScope]::Validation) `
        'validation evidence did not retain its typed scope'

    $completeJson = $partialJson.PSObject.Copy()
    $completeJson.scope = 'Complete'
    $complete = ConvertTo-CertificateEvidence $completeJson 10
    Assert-True ($complete.Scope -eq [CiValidationReuseScope]::Complete) `
        'complete evidence did not retain its typed scope'

    $legacyJson = [pscustomobject]@{
        schemaVersion = 2; runId = '11'; testedSha = $sha; testedTree = $tree
        event = 'merge_group'; requiresValidation = $false; ciGateConclusion = 'success'
    }
    $legacy = ConvertTo-CertificateEvidence $legacyJson 11
    Assert-True ($legacy.Scope -eq [CiValidationReuseScope]::Complete) `
        'legacy full-gate evidence was not treated as complete'

    $legacyV1Json = [pscustomobject]@{
        schemaVersion = 1; runId = '12'; testedSha = $sha
        event = 'pull_request'; ciGateConclusion = 'success'
    }
    $legacyV1 = ConvertTo-CertificateEvidence $legacyV1Json 12
    Assert-True ($legacyV1.Scope -eq [CiValidationReuseScope]::Complete -and
        $legacyV1.RequiresValidation) `
        'legacy schema-v1 evidence did not retain conservative runtime semantics'

    $badBoolean = $partialJson.PSObject.Copy(); $badBoolean.requiresValidation = 'true'
    Assert-Rejected { ConvertTo-CertificateEvidence $badBoolean 10 } `
        'a string requiresValidation value was accepted'
    $badScope = $partialJson.PSObject.Copy(); $badScope.scope = 'AlmostComplete'
    Assert-Rejected { ConvertTo-CertificateEvidence $badScope 10 } `
        'an unknown certificate scope was accepted'
    Assert-Rejected { ConvertTo-CertificateEvidence $partialJson 99 } `
        'a certificate bound to another run was accepted'
    $stringRunId = $partialJson.PSObject.Copy(); $stringRunId.runId = '10'
    Assert-Rejected { ConvertTo-CertificateEvidence $stringRunId 10 } `
        'a schema-v3 certificate with a string runId was accepted'

    $partialCandidate = [pscustomobject]@{
        Scope = [CiValidationReuseScope]::Validation; RunId = 10; CreatedAt = '2026-01-01T00:00:00Z'
        TreeMatches = $true; RequiresValidation = $true; HasAnalysis = $true
        HasCoverage = $true; HasLedger = $true
    }
    $completeCandidate = $partialCandidate.PSObject.Copy()
    $completeCandidate.Scope = [CiValidationReuseScope]::Complete
    $completeCandidate.RunId = 9
    $completeCandidate.CreatedAt = '2025-12-31T00:00:00Z'
    $best = Select-BestCiEvidence @($partialCandidate, $completeCandidate)
    Assert-True ($best.Scope -eq [CiValidationReuseScope]::Complete) `
        'newer partial evidence displaced exact-tree complete evidence'

    $missingCoverage = $partialCandidate.PSObject.Copy(); $missingCoverage.HasCoverage = $false
    Assert-True ($null -eq (Select-BestCiEvidence @($missingCoverage))) `
        'runtime evidence without coverage was reusable'
    $missingLedger = $partialCandidate.PSObject.Copy(); $missingLedger.HasLedger = $false
    Assert-True ($null -eq (Select-BestCiEvidence @($missingLedger))) `
        'runtime evidence without an outcome ledger was reusable'
    $wrongTree = $partialCandidate.PSObject.Copy(); $wrongTree.TreeMatches = $false
    Assert-True ($null -eq (Select-BestCiEvidence @($wrongTree))) `
        'tree-mismatched evidence was reusable'
    $nonRuntime = $partialCandidate.PSObject.Copy()
    $nonRuntime.RequiresValidation = $false
    $nonRuntime.HasAnalysis = $false; $nonRuntime.HasCoverage = $false; $nonRuntime.HasLedger = $false
    Assert-True ($null -ne (Select-BestCiEvidence @($nonRuntime))) `
        'non-runtime evidence incorrectly required test artifacts'

    if ($failures.Count -gt 0) {
        Write-Host 'Resolve-CiValidationReuse self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'Resolve-CiValidationReuse self-test passed.'
    exit 0
}

if ($CommitSha -cnotmatch '^[0-9a-f]{40}$') { throw 'CommitSha must be a lowercase 40-character Git SHA' }
if (Test-Path -LiteralPath $GitHubOutput) {
    # GitHub owns this file and may already contain defaults from a previous step. Appending is the
    # workflow protocol; refusing to overwrite applies to our artifacts, not GITHUB_OUTPUT.
    Write-Verbose "appending reuse decision to $GitHubOutput"
}

if ($EventName -cne 'push') {
    Write-ReuseDecision -DecisionScope None -Summary "Event '$EventName' requires current-run validation."
    exit 0
}

function Invoke-GhJson {
    param([Parameter(Mandatory)] [string[]] $Arguments)
    $json = @(& gh @Arguments 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "gh failed: gh $($Arguments -join ' ')" }
    return (($json -join "`n") | ConvertFrom-Json)
}

function Download-RunArtifact {
    param([long] $RunId, [string] $Name, [string] $Directory)
    if (Test-Path -LiteralPath $Directory) {
        throw "refusing to overwrite existing artifact directory '$Directory'"
    }
    & gh run download $RunId --repo $Repository --name $Name --dir $Directory *> $null
    return $LASTEXITCODE -eq 0
}

$pulls = @(Invoke-GhJson @('api', '-H', 'Accept: application/vnd.github+json',
    "repos/$Repository/commits/$CommitSha/pulls"))
$mergedPulls = @($pulls | Where-Object {
    $_.merged_at -and $_.base -and [string] $_.base.ref -ceq $ExpectedBaseBranch
} | Sort-Object @{ Expression = { [DateTimeOffset] $_.merged_at }; Descending = $true })
if ($mergedPulls.Count -eq 0) {
    Write-ReuseDecision -DecisionScope None -Summary 'No merged PR for this target branch is associated with the push; current-run validation is required.'
    exit 0
}
$exactMerge = @($mergedPulls | Where-Object { [string] $_.merge_commit_sha -ceq $CommitSha } | Select-Object -First 1)
$pull = if ($exactMerge.Count -gt 0) { $exactMerge[0] } else { $mergedPulls[0] }
$prNumber = [long] $pull.number
$headSha = [string] $pull.head.sha
if ($prNumber -lt 1 -or $headSha -cnotmatch '^[0-9a-f]{40}$') {
    throw 'associated PR metadata is incomplete'
}
$masterTree = [string] (& gh api "repos/$Repository/git/commits/$CommitSha" --jq '.tree.sha' 2>$null)
if ($LASTEXITCODE -ne 0 -or $masterTree -cnotmatch '^[0-9a-f]{40}$') {
    throw 'could not resolve the landed Git tree'
}

$deadline = [DateTimeOffset]::UtcNow.AddMinutes($WaitMinutes)
do {
    $prRuns = Invoke-GhJson @('api', '--method', 'GET',
        "repos/$Repository/actions/workflows/sonarcloud.yml/runs",
        '-f', 'event=pull_request', '-f', "head_sha=$headSha", '-f', 'per_page=20')
    $mergeRuns = Invoke-GhJson @('api', '--method', 'GET',
        "repos/$Repository/actions/workflows/sonarcloud.yml/runs",
        '-f', 'event=merge_group', '-f', 'per_page=20')
    $runs = @($mergeRuns.workflow_runs) + @($prRuns.workflow_runs)
    $evidence = [System.Collections.Generic.List[object]]::new()

    foreach ($run in $runs) {
        $runId = [long] $run.id
        $createdAt = [DateTimeOffset] $run.created_at
        # A normal full-matrix run produces more than 100 artifacts. Reading only the first page
        # makes reuse depend on API ordering and can hide either the certificate or coverage.
        $artifactPages = @(Invoke-GhJson @('api', '--paginate', '--slurp',
            "repos/$Repository/actions/runs/$runId/artifacts?per_page=100"))
        $artifacts = @($artifactPages | ForEach-Object { $_.artifacts } |
            Where-Object { -not [bool] $_.expired })
        $certificateArtifacts = @($artifacts | Where-Object {
            ([string] $_.name).StartsWith('ci-expensive-validation-certificate-', [StringComparison]::Ordinal) -or
            ([string] $_.name).StartsWith('ci-validation-certificate-', [StringComparison]::Ordinal)
        } | Sort-Object @{ Expression = {
            if (([string] $_.name).StartsWith('ci-validation-certificate-', [StringComparison]::Ordinal)) { 1 } else { 0 }
        }; Descending = $true })
        if ($certificateArtifacts.Count -eq 0) { continue }

        foreach ($artifact in $certificateArtifacts) {
            $artifactName = [string] $artifact.name
            $artifactDirectory = Join-Path $env:RUNNER_TEMP "validation-certificate-$runId-$([guid]::NewGuid().ToString('N'))"
            if (-not (Download-RunArtifact $runId $artifactName $artifactDirectory)) { continue }
            $certificateFile = @(Get-ChildItem -LiteralPath $artifactDirectory -Filter validation-certificate.json -File -Recurse | Select-Object -First 1)
            if ($certificateFile.Count -eq 0) { continue }
            try {
                $certificate = Get-Content -LiteralPath $certificateFile[0].FullName -Raw | ConvertFrom-Json
                $parsed = ConvertTo-CertificateEvidence $certificate $runId
            }
            catch {
                Write-Host "::warning::rejecting malformed validation certificate from run ${runId}: $($_.Exception.Message)"
                continue
            }
            $expectedArtifactPrefix = if ($parsed.Scope -eq [CiValidationReuseScope]::Complete) {
                'ci-validation-certificate-'
            } else { 'ci-expensive-validation-certificate-' }
            if (-not $artifactName.StartsWith($expectedArtifactPrefix, [StringComparison]::Ordinal)) {
                Write-Host "::warning::certificate scope and artifact name disagree in run $runId"
                continue
            }

            $testedTree = [string] (& gh api "repos/$Repository/git/commits/$($parsed.TestedSha)" --jq '.tree.sha' 2>$null)
            if ($LASTEXITCODE -ne 0 -or $testedTree -cnotmatch '^[0-9a-f]{40}$') { continue }
            $treeMatches = $testedTree -ceq $masterTree -and
                (-not $parsed.TestedTree -or $parsed.TestedTree -ceq $testedTree)

            $analysisArtifacts = @($artifacts | Where-Object {
                ([string] $_.name).StartsWith('ci-test-analysis-', [StringComparison]::Ordinal)
            } | Sort-Object created_at -Descending)
            $coverageCount = @($artifacts | Where-Object {
                ([string] $_.name).StartsWith('coverage-', [StringComparison]::Ordinal)
            }).Count
            $ledgerCount = @($artifacts | Where-Object {
                ([string] $_.name).StartsWith('test-outcome-ledger-', [StringComparison]::Ordinal)
            }).Count
            $hasAnalysis = $false
            if ($analysisArtifacts.Count -gt 0) {
                $analysisDirectory = Join-Path $env:RUNNER_TEMP "ci-test-analysis-$runId-$([guid]::NewGuid().ToString('N'))"
                if (Download-RunArtifact $runId ([string] $analysisArtifacts[0].name) $analysisDirectory) {
                    $analysisFile = @(Get-ChildItem -LiteralPath $analysisDirectory -Filter ci-test-analysis.json -File -Recurse | Select-Object -First 1)
                    if ($analysisFile.Count -gt 0) {
                        try {
                            $analysis = Get-Content -LiteralPath $analysisFile[0].FullName -Raw | ConvertFrom-Json
                            $hasAnalysis = $analysis.source -and [string] $analysis.source.commitSha -ceq $parsed.TestedSha
                        }
                        catch { $hasAnalysis = $false }
                    }
                }
            }

            [void] $evidence.Add([pscustomobject]@{
                Scope = $parsed.Scope
                RunId = $runId
                PrNumber = $prNumber
                CreatedAt = $createdAt
                TestedSha = $parsed.TestedSha
                TreeMatches = $treeMatches
                RequiresValidation = $parsed.RequiresValidation
                HasAnalysis = $hasAnalysis
                HasCoverage = $coverageCount -gt 0
                HasLedger = $ledgerCount -gt 0
            })
        }
    }

    $best = Select-BestCiEvidence @($evidence)
    if ($null -ne $best) {
        $summary = "Reusing $($best.Scope) evidence from PR #$prNumber run $($best.RunId); its tested tree exactly matches the landed tree."
        Write-ReuseDecision -DecisionScope $best.Scope -RunId $best.RunId -PrNumber $prNumber `
            -TestedSha $best.TestedSha -RequiresValidation $best.RequiresValidation -Summary $summary
        exit 0
    }

    $inFlight = @($runs | Where-Object { [string] $_.status -cne 'completed' }).Count -gt 0
    if (-not $inFlight -or [DateTimeOffset]::UtcNow -ge $deadline) { break }
    Write-Host "PR #$prNumber validation is still in flight; waiting for reusable evidence."
    Start-Sleep -Seconds 60
} while ($true)

Write-ReuseDecision -DecisionScope None -PrNumber $prNumber `
    -Summary "No exact-tree validation evidence is reusable for PR #$prNumber; current-run validation is required."
exit 0
