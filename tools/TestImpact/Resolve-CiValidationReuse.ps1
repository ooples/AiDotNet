<#
.SYNOPSIS
    Resolves CI evidence for a landed pull request: exact-tree reuse, or delta reuse when the pull
    request was behind its base branch.

.DESCRIPTION
    A Validation certificate suppresses build/test/model validation but deliberately reruns
    CodeQL and Sonar. A Complete certificate suppresses both stages. Every runtime certificate
    additionally requires the immutable analysis, outcome-ledger, and coverage artifacts that the
    landed run must promote. Missing, malformed, stale, or tree-mismatched evidence fails closed.

    DELTA REUSE. A pull request merged while behind its base branch lands a tree that differs from
    the one its run validated - by exactly the commits the base branch gained in between (Δ).
    Exact-tree matching can never reuse that, which is why nearly every merge re-ran the full
    matrix. With a certified map, Δ is selected like any other change:

      * Δ needs the full matrix (a CI-control, build or unmappable edit)  -> run everything.
      * Δ touches no shard the pull request ran                          -> reuse its results.
      * Δ touches some of the pull request's shards                      -> re-run only those,
        and import the pull request's artifacts for the rest, so the landed commit's ledger,
        analysis and coverage still describe every shard the pull request validated.

    A shard Δ affects but the pull request did not run was validated by the commits that make up Δ,
    each on its own run, and this pull request's change does not reach it; that is the same premise
    pull-request selection itself rests on, and the nightly miss audit measures it.

    Delta reuse is at most Validation-scoped: CodeQL and Sonar analysed a different tree, so they
    always run again on the landed one.
#>
[CmdletBinding(DefaultParameterSetName = 'Resolve')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $Repository,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $CommitSha,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $EventName,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $ExpectedBaseBranch,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $GitHubOutput,
    [Parameter(ParameterSetName = 'Resolve')] [ValidateRange(0, 120)] [int] $WaitMinutes = 40,
    # Delta reuse inputs. Without a certified map there is nothing to select with, and an
    # exact-tree mismatch keeps failing closed exactly as before.
    [Parameter(ParameterSetName = 'Resolve')]
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [string] $MapFile,
    [Parameter(ParameterSetName = 'Resolve')]
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [string] $ShardManifestFile,
    [Parameter(ParameterSetName = 'Resolve')]
    [Parameter(ParameterSetName = 'PlanDelta')] [string] $SelectorPath = "$PSScriptRoot/Select-Shards.ps1",
    # Offline delta planning against the checked-out landed commit, for tests and diagnosis:
    # the tested merge commit's parents and tree, and the shards its run executed.
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [switch] $PlanDelta,
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [string] $TestedBaseSha,
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [string] $TestedHeadSha,
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [string] $TestedTree,
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [AllowEmptyString()] [string] $PullRequestShardsJson,
    [Parameter(Mandatory, ParameterSetName = 'PlanDelta')] [string] $OutFile,
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

function Get-DeltaReuseDecision {
    <#
        Pure: the landed commit's validation plan from the selector's verdict on Δ and the shards
        the pull request's run executed. Mode is Reuse (nothing to re-run), Partial (re-run Rerun,
        import Import from the pull request run) or None (run the full matrix).
    #>
    param(
        [Parameter(Mandatory)] [bool] $SelectionEscalated,
        [Parameter(Mandatory)] [bool] $SelectionRequiresValidation,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $DeltaShards,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $PullRequestShards
    )

    $pullRequest = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
    foreach ($shard in $PullRequestShards) { if ($shard) { [void] $pullRequest.Add([string] $shard) } }

    if ($SelectionEscalated) {
        return [pscustomobject]@{ Mode = 'None'; Rerun = @(); Import = @()
            Why = 'the change since the validated tree needs the full matrix' }
    }
    if (-not $SelectionRequiresValidation) {
        return [pscustomobject]@{ Mode = 'Reuse'; Rerun = @(); Import = @($pullRequest)
            Why = 'the change since the validated tree is non-runtime' }
    }

    $delta = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($shard in $DeltaShards) { if ($shard) { [void] $delta.Add([string] $shard) } }
    # An empty selection for a runtime delta is a selector anomaly, never permission to skip.
    if ($delta.Count -eq 0) {
        return [pscustomobject]@{ Mode = 'None'; Rerun = @(); Import = @()
            Why = 'the selector returned no shards for a runtime change' }
    }

    $rerun = @($pullRequest | Where-Object { $delta.Contains($_) })
    $import = @($pullRequest | Where-Object { -not $delta.Contains($_) })
    if ($rerun.Count -eq 0) {
        return [pscustomobject]@{ Mode = 'Reuse'; Rerun = @(); Import = $import
            Why = 'the change since the validated tree reaches none of the shards the pull request ran' }
    }
    return [pscustomobject]@{ Mode = 'Partial'; Rerun = $rerun; Import = $import
        Why = "the change since the validated tree reaches $($rerun.Count) of the $($pullRequest.Count) shard(s) the pull request ran" }
}

function Resolve-ValidatedTree {
    <#
        Rebuilds the tree a pull-request run validated from its merge commit's two parents, and
        returns it only if it is byte-identical to the tree GitHub reports for that commit.

        The tested merge commit itself is usually unreachable once the pull request merges (its
        refs/pull/N/merge ref moves or is removed), so it cannot be relied on to be fetchable. Its
        parents - the base-branch commit and the pull request head - remain reachable. A
        conflicting rebuild, or any tree difference, returns $null: delta reuse then fails closed.
    #>
    param(
        [Parameter(Mandatory)] [string] $BaseSha,
        [Parameter(Mandatory)] [string] $HeadSha,
        [Parameter(Mandatory)] [string] $ExpectedTree
    )

    $output = @(& git merge-tree --write-tree --no-messages $BaseSha $HeadSha 2>$null)
    if ($LASTEXITCODE -ne 0 -or $output.Count -eq 0) { return $null }
    $tree = ([string] $output[0]).Trim()
    if ($tree -cne $ExpectedTree) { return $null }
    return $tree
}

function Invoke-DeltaPlan {
    <#
        Everything delta reuse decides from Git alone: rebuild the validated tree from the tested
        merge commit's parents, select over what the checked-out landed commit adds to it, and
        intersect with the shards the pull request ran. Returns a failure string, or the plan.
    #>
    param(
        [Parameter(Mandatory)] [string] $BaseSha,
        [Parameter(Mandatory)] [string] $HeadSha,
        [Parameter(Mandatory)] [string] $ExpectedTree,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $PullRequestShards,
        [Parameter(Mandatory)] [string] $Map,
        [Parameter(Mandatory)] [string] $Manifest,
        [Parameter(Mandatory)] [string] $Selector
    )

    $tree = Resolve-ValidatedTree -BaseSha $BaseSha -HeadSha $HeadSha -ExpectedTree $ExpectedTree
    if ($null -eq $tree) { return "the validated tree $ExpectedTree could not be rebuilt exactly from its parents" }

    $expectedShards = @(Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json | ForEach-Object { [string] $_.name })
    $selectionFile = Join-Path ([System.IO.Path]::GetTempPath()) "delta-selection-$PID-$([guid]::NewGuid().ToString('N')).json"
    try {
        & $Selector -MapFile $Map -DeltaFromTree $tree -ShardManifestFile $Manifest `
            -ExpectedShards $expectedShards -OutFile $selectionFile | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $selectionFile)) { return 'the delta selector failed' }
        $selection = Get-Content -LiteralPath $selectionFile -Raw | ConvertFrom-Json
    }
    finally { Remove-Item -LiteralPath $selectionFile -ErrorAction SilentlyContinue }

    $decision = Get-DeltaReuseDecision -SelectionEscalated ([bool] $selection.escalate) `
        -SelectionRequiresValidation ([bool] $selection.requiresValidation) `
        -DeltaShards @($selection.shards) -PullRequestShards $PullRequestShards
    return [pscustomobject]@{ Decision = $decision; Tree = $tree; Selection = $selection }
}

function Get-MissingImportArtifacts {
    <#
        Pure: which per-shard artifacts a partial re-run would import but the pull request run does
        not have. Every consumer imports by exact name (Import-PullRequestShardArtifacts.ps1), and a
        missing one fails that consumer - so a partial plan that relies on it must be declined up front
        and the landed commit validated in full instead.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $ArtifactNames,
        [Parameter(Mandatory)] [string] $TestedSha,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Shards
    )
    $present = [System.Collections.Generic.HashSet[string]]::new([string[]] $ArtifactNames, [StringComparer]::Ordinal)
    $missing = [System.Collections.Generic.List[string]]::new()
    foreach ($shard in $Shards) {
        $slug = $shard -replace '[\\/:*?"<>|\s-]+', '_'
        foreach ($prefix in @('coverage', 'test-results')) {
            $name = "$prefix-$TestedSha-$slug"
            if (-not $present.Contains($name)) { [void] $missing.Add($name) }
        }
    }
    return $missing.ToArray()
}

function Get-OptionalArray {
    <# A property that may be absent (older selector output) read as an array, under StrictMode. #>
    param($Object, [string] $Name)
    if ($null -eq $Object) { return }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) { return }
    return @($property.Value)
}

function Get-ShardFromJobName {
    <# 'Tests (net10.0) - Unit - 01 Activation' -> 'Unit - 01 Activation'; anything else -> $null. #>
    param([string] $JobName)
    if (-not $JobName.StartsWith('Tests (', [StringComparison]::Ordinal)) { return $null }
    $separator = $JobName.IndexOf(') - ', [StringComparison]::Ordinal)
    if ($separator -lt 0) { return $null }
    return $JobName.Substring($separator + 4)
}

function Write-ReuseDecision {
    param(
        [Parameter(Mandatory)] [CiValidationReuseScope] $DecisionScope,
        [long] $RunId = 0,
        [long] $PrNumber = 0,
        [string] $TestedSha = '',
        [bool] $RequiresValidation = $true,
        [string] $Summary = '',
        [string] $DeltaMode = '',
        [string[]] $PartialShards = @(),
        [long] $ImportRunId = 0,
        [string] $ImportSha = '',
        [string[]] $ImportShards = @()
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
        "execute_quality=$(((-not $reuseQuality)).ToString().ToLowerInvariant())",
        "delta_mode=$DeltaMode",
        "partial_shards=$(ConvertTo-Json -InputObject @($PartialShards) -Compress)",
        "import_run_id=$(if ($ImportRunId -gt 0) { $ImportRunId } else { '' })",
        "import_sha=$ImportSha",
        "import_shards=$(ConvertTo-Json -InputObject @($ImportShards) -Compress)"
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

    # ---- Delta reuse decisions. ----------------------------------------------------------------
    $pr = @('Integration D', 'Integration E-G', 'Unit - 10 RL')
    $d = Get-DeltaReuseDecision -SelectionEscalated $true -SelectionRequiresValidation $true -DeltaShards @() -PullRequestShards $pr
    Assert-True ($d.Mode -ceq 'None') 'a delta that needs the full matrix was reused'
    $d = Get-DeltaReuseDecision -SelectionEscalated $false -SelectionRequiresValidation $false -DeltaShards @() -PullRequestShards $pr
    Assert-True ($d.Mode -ceq 'Reuse' -and @($d.Rerun).Count -eq 0) 'a non-runtime delta was not reused outright'
    $d = Get-DeltaReuseDecision -SelectionEscalated $false -SelectionRequiresValidation $true `
        -DeltaShards @('Unit - 02 Data', 'ModelFamily - Audio') -PullRequestShards $pr
    Assert-True ($d.Mode -ceq 'Reuse') 'a delta reaching none of the pull request''s shards was not reused'
    Assert-True (((@($d.Import) | Sort-Object) -join ',') -eq ((@($pr) | Sort-Object) -join ',')) `
        'a full delta reuse did not account for every pull-request shard'
    $d = Get-DeltaReuseDecision -SelectionEscalated $false -SelectionRequiresValidation $true `
        -DeltaShards @('Integration D', 'Unit - 02 Data') -PullRequestShards $pr
    Assert-True ($d.Mode -ceq 'Partial') 'an overlapping delta did not produce a partial re-run'
    Assert-True ((@($d.Rerun) -join ',') -ceq 'Integration D') 'a partial re-run included shards outside the overlap'
    Assert-True ((@($d.Import) -join ',') -ceq 'Integration E-G,Unit - 10 RL') `
        'a partial re-run did not import exactly the pull request''s other shards'
    Assert-True (-not (@($d.Rerun) -contains 'Unit - 02 Data')) `
        'a shard only the base branch''s own commits reach was re-run for this pull request'
    $d = Get-DeltaReuseDecision -SelectionEscalated $false -SelectionRequiresValidation $true -DeltaShards @() -PullRequestShards $pr
    Assert-True ($d.Mode -ceq 'None') 'an empty selection for a runtime delta was treated as permission to skip'
    $d = Get-DeltaReuseDecision -SelectionEscalated $false -SelectionRequiresValidation $true `
        -DeltaShards @('integration d') -PullRequestShards $pr
    Assert-True ($d.Mode -ceq 'Reuse') 'shard names were matched ignoring case'

    Assert-True ((Get-ShardFromJobName 'Tests (net10.0) - Unit - 01 Activation/Attention') -ceq 'Unit - 01 Activation/Attention') `
        'a test job name did not yield its shard'
    Assert-True ($null -eq (Get-ShardFromJobName 'Build')) 'a non-test job was read as a shard'
    Assert-True ($null -eq (Get-ShardFromJobName 'Tests')) 'a malformed test job name was read as a shard'

    # A partial re-run imports both artifacts of every imported shard by exact name.
    $tested = '0123456789abcdef0123456789abcdef01234567'
    $names = @("coverage-$tested-Integration_E_G", "test-results-$tested-Integration_E_G",
        "coverage-$tested-ModelFamily_Generated_Layers_F")
    $missing = @(Get-MissingImportArtifacts -ArtifactNames $names -TestedSha $tested `
        -Shards @('Integration E-G', 'ModelFamily - Generated Layers F'))
    Assert-True (($missing -join ',') -ceq "test-results-$tested-ModelFamily_Generated_Layers_F") `
        'a missing per-shard import artifact was not reported'
    Assert-True (@(Get-MissingImportArtifacts -ArtifactNames $names -TestedSha $tested -Shards @('Integration E-G')).Count -eq 0) `
        'a shard with both artifacts was reported missing'
    Assert-True (@(Get-MissingImportArtifacts -ArtifactNames @() -TestedSha $tested -Shards @()).Count -eq 0) `
        'an empty import list reported missing artifacts'

    # Selector output read under StrictMode must tolerate an absent property.
    Assert-True (@(Get-OptionalArray ([pscustomobject]@{ escalate = $true }) 'routes').Count -eq 0) `
        'an absent routes property was not read as empty'
    Assert-True ((@(Get-OptionalArray ([pscustomobject]@{ routes = @('a', 'b') }) 'routes') -join ',') -ceq 'a,b') `
        'a present routes property was not read back'

    if ($failures.Count -gt 0) {
        Write-Host 'Resolve-CiValidationReuse self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'Resolve-CiValidationReuse self-test passed.'
    exit 0
}

if ($PlanDelta) {
    $shards = @()
    if (-not [string]::IsNullOrWhiteSpace($PullRequestShardsJson)) {
        $shards = @($PullRequestShardsJson | ConvertFrom-Json | ForEach-Object { [string] $_ } | Where-Object { $_ })
    }
    $plan = Invoke-DeltaPlan -BaseSha $TestedBaseSha -HeadSha $TestedHeadSha -ExpectedTree $TestedTree `
        -PullRequestShards $shards -Map $MapFile -Manifest $ShardManifestFile -Selector $SelectorPath
    $result = if ($plan -is [string]) {
        [pscustomobject]@{ mode = 'None'; why = $plan; rerun = @(); import = @(); tree = ''; routes = @() }
    }
    else {
        [pscustomobject]@{
            mode = $plan.Decision.Mode; why = $plan.Decision.Why
            rerun = @($plan.Decision.Rerun); import = @($plan.Decision.Import)
            tree = $plan.Tree; routes = @(Get-OptionalArray $plan.Selection 'routes')
        }
    }
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutFile -Encoding utf8
    Write-Host "delta plan: $($result.mode) - $($result.why)"
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
                TestedTree = $testedTree
                ArtifactNames = @($artifacts | ForEach-Object { [string] $_.name })
                Event = [string] $run.event
                HeadSha = [string] $run.head_sha
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

# ---------------------------------------------------------------- delta reuse
# No exact-tree evidence. The pull request was validated on a base branch that has since moved on;
# decide from the certified map whether that movement can have changed any result it certified.

function Get-DeltaPlan {
    param([Parameter(Mandatory)] $Candidate)

    $commit = Invoke-GhJson @('api', "repos/$Repository/git/commits/$($Candidate.TestedSha)")
    $parents = @($commit.parents | ForEach-Object { [string] $_.sha })
    if ($parents.Count -ne 2) { return "the validated commit is not a two-parent pull-request merge" }
    if ($parents[1] -cne $Candidate.HeadSha) { return "the validated commit's second parent is not the pull request head" }

    foreach ($sha in $parents) {
        & git cat-file -e "$sha^{commit}" 2>$null
        if ($LASTEXITCODE -eq 0) { continue }
        & git fetch --quiet --no-tags origin $sha 2>$null
        if ($sha -ceq $parents[1]) { & git fetch --quiet --no-tags origin "refs/pull/$prNumber/head" 2>$null }
        & git cat-file -e "$sha^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) { return "commit $sha is not available to rebuild the validated tree" }
    }
    $pullRequestShards = @()
    if ($Candidate.RequiresValidation) {
        $jobPages = @(Invoke-GhJson @('api', '--paginate', '--slurp',
            "repos/$Repository/actions/runs/$($Candidate.RunId)/jobs?per_page=100"))
        $pullRequestShards = @($jobPages | ForEach-Object { $_.jobs } |
            Where-Object { [string] $_.conclusion -in @('success', 'failure') } |
            ForEach-Object { Get-ShardFromJobName ([string] $_.name) } |
            Where-Object { $_ } | Sort-Object -Unique)
        # A runtime certificate with no executed shard is not evidence of anything to reuse.
        if ($pullRequestShards.Count -eq 0) { return 'the validated run executed no test shards' }
    }

    return Invoke-DeltaPlan -BaseSha $parents[0] -HeadSha $parents[1] -ExpectedTree $Candidate.TestedTree `
        -PullRequestShards $pullRequestShards -Map $MapFile -Manifest $ShardManifestFile -Selector $SelectorPath
}

$deltaCandidates = @($evidence | Where-Object {
    $_.Event -ceq 'pull_request' -and -not $_.TreeMatches -and
    (-not $_.RequiresValidation -or ($_.HasAnalysis -and $_.HasCoverage -and $_.HasLedger))
} | Sort-Object @{ Expression = { [DateTimeOffset] $_.CreatedAt }; Descending = $true },
                @{ Expression = { [long] $_.RunId }; Descending = $true })

if ($deltaCandidates.Count -gt 0 -and $MapFile -and $ShardManifestFile -and
    (Test-Path -LiteralPath $MapFile) -and (Test-Path -LiteralPath $ShardManifestFile)) {
    # The newest validated state is the closest to what landed, so it leaves the smallest delta.
    $candidate = $deltaCandidates[0]
    $plan = $null
    try { $plan = Get-DeltaPlan -Candidate $candidate }
    catch { $plan = "delta planning failed: $($_.Exception.Message)" }

    if ($plan -is [string]) {
        Write-Host "::notice::delta reuse unavailable for PR #$prNumber run $($candidate.RunId): $plan"
    }
    else {
        $decision = $plan.Decision
        Write-Host "delta from PR #$prNumber run $($candidate.RunId) (tree $($plan.Tree)): $($decision.Mode) - $($decision.Why)"
        foreach ($route in @(Get-OptionalArray $plan.Selection 'routes')) { Write-Host "  delta route: $route" }
        foreach ($reason in @(Get-OptionalArray $plan.Selection 'reasons')) { Write-Host "  delta reason: $reason" }

        if ($decision.Mode -ceq 'Reuse') {
            # Never Complete: CodeQL and Sonar analysed a different tree.
            $summary = "Reusing Validation evidence from PR #$prNumber run $($candidate.RunId) across the base branch's later commits: $($decision.Why)."
            Write-ReuseDecision -DecisionScope Validation -RunId $candidate.RunId -PrNumber $prNumber `
                -TestedSha $candidate.TestedSha -RequiresValidation $candidate.RequiresValidation `
                -DeltaMode 'Reuse' -ImportShards @($decision.Import) -Summary $summary
            exit 0
        }
        $missingImports = @()
        if ($decision.Mode -ceq 'Partial') {
            $missingImports = @(Get-MissingImportArtifacts -ArtifactNames @($candidate.ArtifactNames) `
                -TestedSha $candidate.TestedSha -Shards @($decision.Import))
        }
        if ($decision.Mode -ceq 'Partial' -and $missingImports.Count -gt 0) {
            # Every consumer imports these by exact name and fails on a missing one, so relying on
            # them would turn a reuse decision into a red landed commit. Validate in full instead.
            Write-Host "::notice::delta reuse declined: PR #$prNumber run $($candidate.RunId) lacks $($missingImports.Count) artifact(s) a partial re-run would import: $($missingImports -join ', ')"
        }
        elseif ($decision.Mode -ceq 'Partial') {
            $summary = "Re-running $($decision.Rerun.Count) shard(s) and importing $($decision.Import.Count) from PR #$prNumber run $($candidate.RunId): $($decision.Why).`n`nRe-run: $($decision.Rerun -join ', ')"
            Write-ReuseDecision -DecisionScope None -PrNumber $prNumber -DeltaMode 'Partial' `
                -PartialShards @($decision.Rerun) -ImportRunId $candidate.RunId `
                -ImportSha $candidate.TestedSha -ImportShards @($decision.Import) -Summary $summary
            exit 0
        }
        if ($decision.Mode -ceq 'None') {
            Write-Host "::notice::delta reuse declined: $($decision.Why)"
        }
    }
}

Write-ReuseDecision -DecisionScope None -PrNumber $prNumber `
    -Summary "No exact-tree validation evidence is reusable for PR #$prNumber; current-run validation is required."
exit 0
