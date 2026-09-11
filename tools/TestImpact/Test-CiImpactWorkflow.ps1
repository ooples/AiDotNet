<#
.SYNOPSIS
    Verifies the two CI time-saving contracts at the workflow wiring boundary.

.DESCRIPTION
    Script-level selector tests cannot prove that a GitHub Actions job actually consumes the
    selector output, and the exact-tree resolver cannot save work if a newly added job forgets to
    honor its decision. This test deliberately inspects the workflow graph as checked in.

    It fails closed when:
      * an expensive job can start without the certified-validation decision;
      * shadow mode can silently replace a reduced matrix with the full matrix;
      * the map lifecycle does not distinguish an unaudited candidate from a certified map; or
      * a no-checkout dispatch job relies on an ambient git repository.
#>
[CmdletBinding()]
param(
    [string] $ValidationWorkflow = '.github/workflows/sonarcloud.yml',
    [string] $MapWorkflow = '.github/workflows/test-impact-map.yml',
    [string] $Selector = 'tools/TestImpact/Select-Shards.ps1',
    [string] $CertifiedMapResolver = 'tools/TestImpact/Resolve-CertifiedShardMap.ps1',
    [string] $RequiredArtifactReceiver = 'tools/TestImpact/Receive-RequiredArtifact.ps1',
    [string] $ValidationReuseResolver = 'tools/TestImpact/Resolve-CiValidationReuse.ps1',
    [string] $ValidationCertificateWriter = 'tools/TestImpact/New-CiValidationCertificate.ps1',
    [string] $MapCertificateWriter = 'tools/TestImpact/New-ShardMapCertificate.ps1',
    [string] $CiGatePolicy = 'tools/TestImpact/Assert-CiGate.ps1',
    [string] $ShardManifest = '.github/test-shards.yml'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$failures = [System.Collections.Generic.List[string]]::new()

function Assert-Contract {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { [void] $failures.Add($Message) }
}

function Get-JobBlock {
    param([string] $WorkflowText, [string] $Job)

    $escaped = [Regex]::Escape($Job)
    $match = [Regex]::Match(
        $WorkflowText,
        "(?ms)^  ${escaped}:\s*\r?\n(?<body>.*?)(?=^  [A-Za-z0-9_-]+:\s*\r?\n|\z)")
    if (-not $match.Success) {
        [void] $failures.Add("workflow job '$Job' is absent")
        return ''
    }
    return $match.Value
}

function Get-StepBlock {
    param([string] $JobBlock, [string] $Step)

    $escaped = [Regex]::Escape($Step)
    return [Regex]::Match(
        $JobBlock,
        "(?ms)^      - name: ${escaped}\s*\r?\n.*?(?=^      - [A-Za-z0-9_-]+:|\z)").Value
}

function Get-JobHeader {
    param([string] $JobBlock)

    $steps = [Regex]::Match($JobBlock, '(?m)^    steps:\s*$')
    if (-not $steps.Success) { return '' }
    return $JobBlock.Substring(0, $steps.Index)
}

function Test-JobDependency {
    param([string] $JobHeader, [string] $Dependency)

    $match = [Regex]::Match($JobHeader, '(?m)^    needs:[ \t]*(?<inline>[^\r\n]*)[ \t]*\r?$')
    if (-not $match.Success) { return $false }
    $inline = $match.Groups['inline'].Value.Trim()
    if ($inline.StartsWith('[', [StringComparison]::Ordinal) -and
        $inline.EndsWith(']', [StringComparison]::Ordinal)) {
        $values = @($inline.Substring(1, $inline.Length - 2).Split(',') |
            ForEach-Object { $_.Trim().Trim("'`"") })
        return $values -ccontains $Dependency
    }
    if ($inline) { return $inline.Trim("'`"") -ceq $Dependency }

    $list = [Regex]::Match($JobHeader, '(?ms)^    needs:[ \t]*\r?\n(?<items>(?:      - [^\r\n]+\r?\n?)+)')
    if (-not $list.Success) { return $false }
    $values = @([Regex]::Matches($list.Groups['items'].Value, '(?m)^      -[ \t]*(?<value>[^\r\n]+?)[ \t]*\r?$') |
        ForEach-Object { $_.Groups['value'].Value.Trim().Trim("'`"") })
    return $values -ccontains $Dependency
}

function Get-ContinuedShellCommand {
    param([string] $Text, [string] $CommandPattern)

    $lines = [Regex]::Split($Text, '\r?\n')
    $commands = [System.Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -notmatch $CommandPattern) { continue }
        $parts = [System.Collections.Generic.List[string]]::new()
        do {
            $line = $lines[$i]
            [void] $parts.Add($line.Trim())
            $continued = $line.TrimEnd().EndsWith('\', [StringComparison]::Ordinal)
            if ($continued) { $i++ }
        } while ($continued -and $i -lt $lines.Count)
        [void] $commands.Add($parts -join ' ')
    }
    return @($commands)
}

function Test-MapSelectorPullRequestScope {
    param([string] $Step)

    # Parse individual PowerShell invocations rather than searching the whole step:
    # the classifier's argument, a comment, or a different variable proves nothing
    # about the map-backed command that actually emits the selected shard matrix.
    $lines = [regex]::Split($Step, '\r?\n')
    $mapCommands = [System.Collections.Generic.List[System.Management.Automation.Language.CommandAst]]::new()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -notmatch '^\s*&\s+\./tools/TestImpact/Select-Shards\.ps1\b') { continue }
        $parts = [System.Collections.Generic.List[string]]::new()
        do {
            $line = $lines[$i]
            [void] $parts.Add($line)
            $continued = $line.TrimEnd().EndsWith('`', [StringComparison]::Ordinal)
            if ($continued) { $i++ }
        } while ($continued -and $i -lt $lines.Count)
        $tokens = $null; $parseErrors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseInput(
            ($parts -join "`n"), [ref] $tokens, [ref] $parseErrors)
        if ($parseErrors.Count -gt 0) { return $false }
        foreach ($command in $ast.FindAll({ param($node)
            $node -is [System.Management.Automation.Language.CommandAst]
        }, $true)) {
            if ($command.GetCommandName() -cne './tools/TestImpact/Select-Shards.ps1') { continue }
            if (@($command.CommandElements | Where-Object {
                $_ -is [System.Management.Automation.Language.CommandParameterAst] -and $_.ParameterName -ceq 'MapFile'
            }).Count -gt 0) { [void] $mapCommands.Add($command) }
        }
    }
    if ($mapCommands.Count -ne 1) { return $false }
    $elements = $mapCommands[0].CommandElements
    $arguments = @{}
    for ($i = 1; $i -lt $elements.Count; $i++) {
        $element = $elements[$i]
        if ($element -isnot [System.Management.Automation.Language.CommandParameterAst] -or
            $element.ParameterName -cnotin @('MapFile', 'PullRequestHeadSha')) { continue }
        if ($arguments.ContainsKey($element.ParameterName)) { return $false }
        $value = $element.Argument
        if ($null -eq $value -and $i + 1 -lt $elements.Count) { $value = $elements[$i + 1] }
        $arguments[$element.ParameterName] = $value
    }
    return $arguments.ContainsKey('MapFile') -and $arguments.ContainsKey('PullRequestHeadSha') -and
        $arguments.MapFile -is [System.Management.Automation.Language.StringConstantExpressionAst] -and
        $arguments.MapFile.Value -ceq 'map/shard-map.json' -and
        $arguments.PullRequestHeadSha -is [System.Management.Automation.Language.VariableExpressionAst] -and
        $arguments.PullRequestHeadSha.VariablePath.UserPath -ceq 'env:PR_HEAD_SHA'
}

$validation = Get-Content -LiteralPath $ValidationWorkflow -Raw
$map = Get-Content -LiteralPath $MapWorkflow -Raw
$selectorText = Get-Content -LiteralPath $Selector -Raw
$certifiedMapResolverText = Get-Content -LiteralPath $CertifiedMapResolver -Raw
$requiredArtifactReceiverText = Get-Content -LiteralPath $RequiredArtifactReceiver -Raw
$validationReuseResolverText = Get-Content -LiteralPath $ValidationReuseResolver -Raw
$validationCertificateWriterText = Get-Content -LiteralPath $ValidationCertificateWriter -Raw
$mapCertificateText = Get-Content -LiteralPath $MapCertificateWriter -Raw
$ciGatePolicyText = Get-Content -LiteralPath $CiGatePolicy -Raw

# Every job in this list consumes meaningful runner time. Dependency-based incidental skipping is
# not enough: each job must explicitly consume both independent typed decisions -- whether exact
# PR validation is reusable, and whether this change can affect runtime/model/test behavior.
$validationJobs = @(
    'build',
    'build-compat',
    'test-net10-sharded',
    'test-regression-analysis',
    'size-check',
    'ci-test-analysis'
)
$qualityJobs = @('codeql', 'sonarcloud')
$expensiveJobs = @($validationJobs) + @($qualityJobs)

foreach ($job in $expensiveJobs) {
    $block = Get-JobBlock -WorkflowText $validation -Job $job
    if (-not $block) { continue }
    $header = Get-JobHeader -JobBlock $block
    Assert-Contract ([bool] $header) `
        "expensive job '$job' has no job-level header before its steps"
    Assert-Contract (Test-JobDependency -JobHeader $header -Dependency 'validation-source') `
        "expensive job '$job' does not explicitly depend on validation-source"
    Assert-Contract (Test-JobDependency -JobHeader $header -Dependency 'select-shards') `
        "expensive job '$job' does not explicitly depend on select-shards"
    $jobIf = [Regex]::Match($header, '(?m)^    if:\s*(?<value>[^\r\n]+)\s*$')
    Assert-Contract $jobIf.Success `
        "expensive job '$job' has no job-level execution condition"
    $executionOutput = if ($qualityJobs -ccontains $job) { 'execute_quality' } else { 'execute_validation' }
    Assert-Contract ($jobIf.Success -and $jobIf.Groups['value'].Value.Contains(
        "fromJSON(needs.validation-source.outputs.$executionOutput)")) `
        "expensive job '$job' does not consume the typed $executionOutput decision"
    if ($validationJobs -ccontains $job) {
        Assert-Contract ($jobIf.Success -and
            $jobIf.Groups['value'].Value.Contains('fromJSON(needs.select-shards.outputs.requires_validation')) `
            "expensive job '$job' does not consume the typed requires_validation decision"
    }
}

# The active repository ruleset requires a CodeQL result for the candidate commit, even when no
# runtime files changed. Keep it as one mandatory job while suppressing every model/test matrix.
$codeqlJob = Get-JobBlock -WorkflowText $validation -Job 'codeql'
$codeqlHeader = Get-JobHeader -JobBlock $codeqlJob
Assert-Contract (-not $codeqlHeader.Contains(
        'fromJSON(needs.select-shards.outputs.requires_validation')) `
    'CodeQL can be skipped even though the Main ruleset requires its result'
Assert-Contract ($codeqlHeader.Contains('always()') -and $codeqlHeader.Contains('!cancelled()')) `
    'CodeQL cannot publish its required result after a selector failure'

# Every Sonar step parses this value with fromJSON. It must therefore be defined on the Sonar job,
# not on a neighboring job where it is invisible and becomes a null template value at runtime.
$sonarJob = Get-JobBlock -WorkflowText $validation -Job 'sonarcloud'
$sonarHeader = Get-JobHeader -JobBlock $sonarJob
$effectiveValidationEnvironment = 'EFFECTIVE_REQUIRES_VALIDATION: ${{ (needs.validation-source.outputs.reuse == ''true'' && needs.validation-source.outputs.reused_requires_validation) || needs.select-shards.outputs.requires_validation || ''true'' }}'
Assert-Contract ($sonarHeader.Contains($effectiveValidationEnvironment)) `
    'Sonar steps parse EFFECTIVE_REQUIRES_VALIDATION but the Sonar job does not define it'
Assert-Contract (([Regex]::Matches($validation, '(?m)^      EFFECTIVE_REQUIRES_VALIDATION:')).Count -eq 1) `
    'EFFECTIVE_REQUIRES_VALIDATION must be defined exactly once on the Sonar job'

foreach ($job in @(
    'test-regression-analysis', 'sonarcloud', 'ci-test-analysis',
    'validation-gate', 'certify-expensive-validation', 'promote-ci-test-analysis',
    'ci-gate', 'certify-validation'
)) {
    $header = Get-JobHeader -JobBlock (Get-JobBlock -WorkflowText $validation -Job $job)
    Assert-Contract ($header.Contains('always()') -and $header.Contains('!cancelled()')) `
        "job '$job' can keep an explicitly canceled run alive"
}

$resolver = Get-JobBlock -WorkflowText $validation -Job 'validation-source'
foreach ($output in 'execute_validation', 'execute_quality') {
    Assert-Contract ($resolver.Contains("steps.resolve.outputs.$output || steps.defaults.outputs.$output")) `
        "validation-source does not publish $output"
    Assert-Contract ($resolver.Contains("echo '$output=true'")) `
        "the resolver lacks a fail-closed $output default"
}
Assert-Contract ($resolver.Contains('steps.resolve.outputs.reuse_scope || steps.defaults.outputs.reuse_scope')) `
    'validation-source does not publish the typed reuse scope'
$resolveStep = Get-StepBlock -JobBlock $resolver -Step 'Resolve exact-tree PR run'
Assert-Contract ([bool] $resolveStep) `
    'the exact-tree resolver step is absent'
$resolverCheckoutStep = Get-StepBlock -JobBlock $resolver -Step 'Checkout validation reuse policy'
Assert-Contract ([bool] $resolverCheckoutStep) `
    'validation-source invokes a repository script without checking out the tested tree'
Assert-Contract ($resolverCheckoutStep.Contains('uses: actions/checkout@')) `
    'validation-source checkout step does not use actions/checkout'
Assert-Contract ($resolverCheckoutStep.Contains('persist-credentials: false')) `
    'validation-source persists checkout credentials although it only reads Git history'
Assert-Contract ([bool] $resolverCheckoutStep -and [bool] $resolveStep -and
    $resolver.IndexOf($resolverCheckoutStep, [StringComparison]::Ordinal) -lt
    $resolver.IndexOf($resolveStep, [StringComparison]::Ordinal)) `
    'validation-source invokes the exact-tree resolver before checking out its script'
Assert-Contract ($resolveStep.Contains('continue-on-error: true')) `
    'an unexpected resolver failure blocks dependents instead of retaining fail-closed defaults'
Assert-Contract ($resolveStep.Contains('./tools/TestImpact/Resolve-CiValidationReuse.ps1')) `
    'the workflow bypasses the executable typed validation resolver'
Assert-Contract ($validation.Contains("- 'ci-proof/**'")) `
    'the real post-merge push path cannot be exercised before this change reaches master'
Assert-Contract ($resolveStep.Contains("-ExpectedBaseBranch '`${{ github.ref_name }}'")) `
    'the resolver does not bind associated PR evidence to the pushed target branch'
Assert-Contract ($validationReuseResolverText.Contains('enum CiValidationReuseScope')) `
    'validation reuse scope is represented by strings instead of a closed enum'
Assert-Contract ($validationReuseResolverText.Contains("'ci-expensive-validation-certificate-'")) `
    'the resolver cannot consume validation-only evidence'
Assert-Contract ($validationReuseResolverText.Contains("'ci-validation-certificate-'")) `
    'the resolver cannot consume complete evidence'
Assert-Contract ($validationReuseResolverText.Contains('HasAnalysis -and $_.HasCoverage -and $_.HasLedger')) `
    'runtime reuse does not require analysis, coverage, and outcome-ledger artifacts'
Assert-Contract ($validationReuseResolverText.Contains("'--paginate', '--slurp'")) `
    'validation reuse examines only the first 100 artifacts of a sharded run'

$selectorJob = Get-JobBlock -WorkflowText $validation -Job 'select-shards'
$selectorHeader = Get-JobHeader -JobBlock $selectorJob
Assert-Contract (Test-JobDependency -JobHeader $selectorHeader -Dependency 'validation-source') `
    'select-shards does not depend on validation-source'
Assert-Contract ($selectorHeader.Contains('fromJSON(needs.validation-source.outputs.execute_validation)')) `
    'select-shards does not consume the typed execute_validation decision'
Assert-Contract ($selectorHeader.Contains('requires_validation: ${{ steps.select.outputs.requires_validation }}')) `
    'select-shards does not publish its typed runtime-validation decision'
$selfTestStep = Get-StepBlock -JobBlock $selectorJob -Step 'Verify the impact tooling'
Assert-Contract ([bool] $selfTestStep) `
    'select-shards no longer runs the impact tooling self-tests'
Assert-Contract (-not $selfTestStep.Contains('continue-on-error: true')) `
    'checked-in impact tooling can fail its self-tests while CI remains green'
Assert-Contract ($selfTestStep.Contains('./tools/TestImpact/Receive-RequiredArtifact.ps1 -SelfTest')) `
    'the required-artifact transport policy is not executed against its self-tests'
Assert-Contract ($selfTestStep.Contains('./tools/TestImpact/Invoke-GitHubApiWithRetry.ps1 -SelfTest')) `
    'the aggregate GitHub API retry policy is not executed against its self-tests'
Assert-Contract ($selfTestStep.Contains('./tools/TestImpact/Test-ValidationReuseModes.ps1')) `
    'the post-merge certificate modes are not exercised by primary CI'
$selectStep = Get-StepBlock -JobBlock $selectorJob -Step 'Select'
Assert-Contract (-not $selectStep.Contains('-AuditUnchangedMap')) `
    'ordinary PR selection was given the audit-only unchanged-map capability'
Assert-Contract ($selectStep.Contains('-ClassifyOnly')) `
    'non-runtime classification still depends on a coverage map being available'
Assert-Contract ($selectStep.Contains('-ClassifyOnly `') -and
        $selectStep.Contains('-PullRequestHeadSha $env:PR_HEAD_SHA -OutFile path-classification.json')) `
    'non-runtime classification does not use the exact PR base-to-head path set'
Assert-Contract ($selectStep.Contains('PR_HEAD_SHA: ${{ github.event.pull_request.head.sha }}')) `
    'the selector is not given the pull request head it must verify as the merge commit''s second parent'
# The event base.sha is the base branch as it was when the pull request was opened. For a pull
# request behind master it made every commit master gained since look like part of the pull
# request: on #2100 it attributed 16 merged CI-control files to a 4-file change, escalating 116 shards.
Assert-Contract (-not $selectStep.Contains('pull_request.base.sha') -and -not $selectStep.Contains('-BaseSha')) `
    'pull-request selection reads the stale event base.sha instead of the merge commit''s first parent'
Assert-Contract (Test-MapSelectorPullRequestScope -Step $selectStep) `
    'the map-backed selector does not scope its change to the pull request head'
Assert-Contract ($selectorText.Contains('function Resolve-PullRequestBase') -and
        $selectorText.Contains('$parents.Count -ne 3') -and
        $selectorText.Contains('$parents[2].Equals($PullRequestHeadSha')) `
    'the selector does not verify the checkout is the two-parent merge of the pull request head'
Assert-Contract ($selectStep.Contains('-ShardManifestFile shard-manifest.json') -and
        $selectStep.Contains('Set-Content -LiteralPath shard-manifest.json')) `
    'test sources are not routed through the shard manifest, so every test edit escalates'
Assert-Contract ($selectStep.Contains('$pathRequiresValidation = Read-RequiredJsonBoolean')) `
    'the selector does not validate and consume the path classifier boolean'
Assert-Contract ($selectStep.Contains('$selectionEscalated = Read-RequiredJsonBoolean')) `
    'the selector does not validate the map escalation decision'
Assert-Contract ($selectStep.Contains('$selectionRequiresValidation = Read-RequiredJsonBoolean')) `
    'the selector does not validate the map runtime-validation decision'
Assert-Contract ($selectStep.Contains("throw 'shard selector contradicted the exact-path runtime decision'")) `
    'the map selector can suppress validation after the exact-path classifier required it'
Assert-Contract ($selectStep.Contains("selector output property 'shards' must be a JSON string array")) `
    'the selector does not validate the selected shard payload type'
Assert-Contract ($selectStep.Contains('requires_validation=$($requiresValidation.ToString().ToLowerInvariant())')) `
    'the selector does not emit the runtime-validation decision as a JSON boolean'
Assert-Contract ($selectStep.Contains('if ($requiresValidation -and $matrixShards.Count -eq 0)')) `
    'the empty-matrix fallback cannot distinguish an authorized non-runtime result from uncertainty'
Assert-Contract ($selectorText.Contains('enum ChangedPathImpact')) `
    'changed-path control flow is not represented by a closed enum'
Assert-Contract ($selectorText.Contains('SelectionControl')) `
    'selector-control changes are not represented separately from runtime/build invalidation'
Assert-Contract ($selectorText.Contains('diff --no-renames --name-only $BaseSha HEAD')) `
    'path classification can hide the source side of a rename'
Assert-Contract ($selectorText.Contains('diff --no-renames --name-only $MapSha HEAD')) `
    'normal shard selection can hide the source path of a rename'
Assert-Contract ($selectorText.Contains('diff --no-renames --no-ext-diff -U0 $MapSha HEAD')) `
    'normal shard range selection can omit the deleted side of a rename'
Assert-Contract ($selectorHeader.Contains(
        'coverage_run_without_instrumentation: ${{ steps.select.outputs.coverage_run_without_instrumentation }}')) `
    'select-shards does not publish the known always-run coverage disposition'
Assert-Contract ($selectStep.Contains('-NoCoverageShards $nn')) `
    'coverage splitting is not restricted to the workflow''s explicit heavy/timing boundary'
Assert-Contract ($selectStep.Contains('$runWithoutInstrumentation = @($coverageSplit.runWithoutCoverage)')) `
    'the selector drops the RunWithoutCoverage disposition before matrix execution'
Assert-Contract ($selectStep.Contains('runWithoutInstrumentationShardNames = @($runWithoutInstrumentation)')) `
    'coverage provenance does not record which shards deliberately ran without instrumentation'
Assert-Contract ($selectStep.Contains('coverage_run_without_instrumentation=$(ConvertTo-Json')) `
    'the typed coverage decision is not transported to the shard jobs'

# ---- Post-merge delta reuse. A pull request merged while behind master lands a tree its run never
# validated, so exact-tree reuse never matched and nearly every merge re-ran all 116 shards.
$sourceHeader = Get-JobHeader -JobBlock $resolver
Assert-Contract ($resolver.Contains("fetch-depth: `${{ github.event_name == 'push' && '0' || '1' }}") -and
        -not $resolver.Contains("== 'push' && 0 ||")) `
    'validation-source lacks the history to rebuild the validated tree and diff it in map coordinates'
$deltaMapStep = Get-StepBlock -JobBlock $resolver -Step 'Resolve certified shard map for delta reuse'
Assert-Contract ([regex]::Matches($deltaMapStep, '(?m)^        id: delta-map[ \t]*(?:#[^\r\n]*)?\r?$').Count -eq 1 -and
        [regex]::Matches($deltaMapStep, '(?m)^        id:').Count -eq 1) `
    'the delta-map step lacks the exact unique identity consumed by the reuse resolver'
Assert-Contract ([bool] $deltaMapStep -and $deltaMapStep.Contains('continue-on-error: true') -and
        $deltaMapStep.Contains('Resolve-CertifiedShardMap.ps1') -and $deltaMapStep.Contains('Test-CertifiedShardMap.ps1')) `
    'delta reuse does not select with an audited map, or a map failure can fail the push instead of disabling delta reuse'
$mapTimeout = [regex]::Match($deltaMapStep, '(?m)^        timeout-minutes: (?<minutes>[1-9][0-9]*)[ \t]*\r?$')
$jobTimeout = [regex]::Match($sourceHeader, '(?m)^    timeout-minutes: (?<minutes>[1-9][0-9]*)[ \t]*\r?$')
$evidenceWait = [regex]::Match($resolveStep, '(?m)^            -WaitMinutes (?<minutes>[0-9]+)[ \t]*\r?$')
Assert-Contract ($mapTimeout.Success -and $jobTimeout.Success -and $evidenceWait.Success -and
        [int] $mapTimeout.Groups['minutes'].Value + [int] $evidenceWait.Groups['minutes'].Value + 5 -le
            [int] $jobTimeout.Groups['minutes'].Value) `
    'delta-map resolution is not bounded within the job budget with resolver-wait and setup headroom'
Assert-Contract ($resolveStep.Contains("-MapFile '`${{ steps.delta-map.outputs.map_file }}'") -and
        $resolveStep.Contains("-ShardManifestFile '`${{ steps.delta-map.outputs.manifest_file }}'")) `
    'the reuse resolver is not given the delta-reuse map and manifest'
foreach ($output in @('delta_mode', 'partial_shards', 'import_run_id', 'import_sha', 'import_shards')) {
    Assert-Contract ($sourceHeader.Contains("${output}: `${{ steps.resolve.outputs.$output || steps.defaults.outputs.$output }}")) `
        "validation-source does not publish the fail-closed delta output '$output'"
}
Assert-Contract ($validationReuseResolverText.Contains('function Resolve-ValidatedTree') -and
        $validationReuseResolverText.Contains('$tree -cne $ExpectedTree')) `
    'delta reuse does not require the rebuilt tree to equal the validated tree exactly'
Assert-Contract ($validationReuseResolverText.Contains("-DecisionScope Validation -RunId `$candidate.RunId")) `
    'delta reuse can claim Complete scope although CodeQL and Sonar analysed a different tree'
Assert-Contract ($selectStep.Contains('$deltaPartial') -and $selectStep.Contains('PARTIAL_SHARDS: ${{ needs.validation-source.outputs.partial_shards }}')) `
    'a partial post-merge re-run does not reduce the shard matrix'
$mapDownload = Get-StepBlock -JobBlock $selectorJob -Step 'Download the shard map'
Assert-Contract ($mapDownload.Contains("MAP_BRANCH: `${{ startsWith(github.base_ref, 'ci-proof/') && 'master' || github.base_ref || 'master' }}")) `
    'a canary into a ci-proof/** branch looks for maps built on that branch, finds none, and cannot prove selection'
Assert-Contract (-not $selectStep.Contains('-DeltaFromTree')) `
    'the select step computes its own master delta - a second, unaudited reuse mechanism'
Assert-Contract ($selectStep.Contains('foreach ($name in $importShards) { [void] $running.Add($name) }')) `
    'imported shards are reported to regression analysis as deliberately skipped'
foreach ($consumer in @(
        @{ Job = 'test-regression-analysis'; Prefix = 'coverage'; Download = 'Download pull-request shard results for delta import'; Import = 'Import pull-request shard results' },
        @{ Job = 'ci-test-analysis'; Prefix = 'test-results'; Download = 'Download pull-request diagnostics for delta import'; Import = 'Import pull-request diagnostics' },
        @{ Job = 'sonarcloud'; Prefix = 'coverage'; Download = 'Download pull-request coverage for delta import'; Import = 'Import pull-request coverage' })) {
    $consumerJob = Get-JobBlock -WorkflowText $validation -Job $consumer.Job
    $downloadStep = Get-StepBlock -JobBlock $consumerJob -Step $consumer.Download
    $importStep = Get-StepBlock -JobBlock $consumerJob -Step $consumer.Import
    $downloadInputs = [regex]::Match($downloadStep,
        '(?m)^        with:\r?\n(?<fields>(?:^          [^\r\n]*(?:\r?\n|\z))*)').Groups['fields'].Value
    $expectedRunId = [regex]::Escape('          run-id: ${{ needs.validation-source.outputs.import_run_id }}')
    $expectedPattern = [regex]::Escape("          pattern: $($consumer.Prefix)-`${{ needs.validation-source.outputs.import_sha }}-*")
    Assert-Contract ($downloadStep -match '(?m)^        uses: actions/download-artifact@[^\s#]+(?:[ \t]+#[^\r\n]*)?[ \t]*\r?$' -and
            $downloadInputs -match ('(?m)^' + $expectedRunId + '[ \t]*\r?$') -and
            $downloadInputs -match ('(?m)^' + $expectedPattern + '[ \t]*\r?$') -and
            $downloadInputs -match '(?m)^          path: delta-import-staging[ \t]*\r?$') `
        "$($consumer.Job) delta download is not bound to the matching PR artifacts"
    $expectedInvocation = '(?m)^        run: \|\r?\n' +
        '          \./tools/TestImpact/Import-PullRequestShardArtifacts\.ps1 -StagingDirectory delta-import-staging `\r?\n' +
        '            -DestinationDirectory [^\s]+ -ArtifactPrefix ' + [regex]::Escape($consumer.Prefix) + ' `\r?\n'
    Assert-Contract ($importStep -match $expectedInvocation) `
        "$($consumer.Job) delta importer is not bound to its matching download"
    foreach ($step in @($downloadStep, $importStep)) {
        $condition = [regex]::Match($step, '(?m)^        if: (?<expression>[^\r\n]+)').Groups['expression'].Value
        $expectedCondition = "needs.validation-source.outputs.import_run_id != '' && needs.select-shards.outputs.escalated == 'false'"
        if ($consumer.Job -ceq 'sonarcloud') { $expectedCondition = "`${{ fromJSON(env.EFFECTIVE_REQUIRES_VALIDATION) && $expectedCondition }}" }
        Assert-Contract ($condition.Trim() -ceq $expectedCondition) `
            "$($consumer.Job) delta import is not disabled after selector escalation"
    }
}

# A 100+ shard fan-out must not resolve the same build artifact by name in every job. That path calls
# ListArtifacts concurrently and GitHub responds with a secondary-rate-limit 403. The build publishes
# the immutable ID and digest once; both consumer matrices use the tested direct receiver, retain hard
# failure on exhaustion, and suppress output uploads when their prerequisite never arrived.
$buildJob = Get-JobBlock -WorkflowText $validation -Job 'build'
$buildHeader = Get-JobHeader -JobBlock $buildJob
$buildUpload = Get-StepBlock -JobBlock $buildJob -Step 'Upload build artifacts'
Assert-Contract ($buildHeader.Contains('artifact_id: ${{ steps.upload-build-artifact.outputs.artifact-id }}')) `
    'build does not expose the immutable artifact ID to its consumers'
Assert-Contract ($buildHeader.Contains("artifact_digest: `${{ format('sha256:{0}', steps.upload-build-artifact.outputs.artifact-digest) }}")) `
    'build does not expose the upload digest in the receiver''s canonical SHA-256 form'
Assert-Contract ($buildUpload.Contains('id: upload-build-artifact')) `
    'the build artifact upload has no stable step identity for its outputs'

Assert-Contract ($requiredArtifactReceiverText.Contains('enum ArtifactRequestDisposition')) `
    'artifact retry state is represented by strings instead of a closed type'
Assert-Contract ($requiredArtifactReceiverText.Contains('actions/artifacts/$ArtifactId/zip')) `
    'required artifact transport does not use the immutable-ID archive endpoint'
Assert-Contract ($requiredArtifactReceiverText.Contains("--proto-redir '=https'")) `
    'required artifact transport permits a redirect to downgrade from HTTPS'
Assert-Contract (-not $requiredArtifactReceiverText.Contains('ArtifactService/ListArtifacts')) `
    'required artifact transport still performs the rate-limited artifact-list lookup'
Assert-Contract ($requiredArtifactReceiverText.Contains('Test-ArtifactDigest')) `
    'direct artifact transport does not validate the upload digest before extraction'
Assert-Contract ($requiredArtifactReceiverText.Contains('secondary rate limit')) `
    'artifact transport does not distinguish transient throttling from a permission denial'
Assert-Contract ($requiredArtifactReceiverText.Contains('Start-Sleep -Seconds $delay')) `
    'artifact transport retries immediately instead of applying its tested backoff policy'
Assert-Contract ($requiredArtifactReceiverText.Contains('$PSNativeCommandUseErrorActionPreference = $false')) `
    'native-command error handling can bypass the typed artifact retry policy'

$githubApiRetryPath = Join-Path $PSScriptRoot 'Invoke-GitHubApiWithRetry.ps1'
$githubApiRetryText = Get-Content -LiteralPath $githubApiRetryPath -Raw
Assert-Contract ($githubApiRetryText.Contains('enum GitHubApiRequestDisposition')) `
    'GitHub API retry state is represented by strings instead of a closed type'
Assert-Contract ($githubApiRetryText.Contains('Get-GitHubApiRequestDisposition')) `
    'GitHub API requests do not distinguish transient and permanent failures'
Assert-Contract ($githubApiRetryText.Contains('Start-Sleep -Seconds $delay')) `
    'GitHub API transient failures retry immediately instead of applying bounded backoff'

$analysisJob = Get-JobBlock -WorkflowText $validation -Job 'ci-test-analysis'
$collectStateStep = Get-StepBlock -JobBlock $analysisJob `
    -Step 'Collect shard states and latest merged baseline'
Assert-Contract ($collectStateStep.Contains('. ./tools/TestImpact/Invoke-GitHubApiWithRetry.ps1')) `
    'aggregate test analysis does not load the tested GitHub API retry policy'
Assert-Contract (-not $collectStateStep.Contains('Invoke-RestMethod')) `
    'aggregate test analysis bypasses retry policy for a REST read'
Assert-Contract (-not $collectStateStep.Contains('Invoke-WebRequest')) `
    'aggregate test analysis bypasses retry policy for an artifact download'
Assert-Contract (([Regex]::Matches($collectStateStep, 'Invoke-GitHubApiWithRetry')).Count -eq 5) `
    'not every aggregate GitHub API read is routed through retry policy'

$artifactConsumers = @('test-net10-sharded')
foreach ($job in $artifactConsumers) {
    $consumer = Get-JobBlock -WorkflowText $validation -Job $job
    $header = Get-JobHeader -JobBlock $consumer
    $download = Get-StepBlock -JobBlock $consumer -Step 'Download required build artifact'
    Assert-Contract ([bool] $download) `
        "artifact consumer '$job' does not use the required-artifact transport"
    Assert-Contract ($download.Contains('./tools/TestImpact/Receive-RequiredArtifact.ps1')) `
        "artifact consumer '$job' bypasses the tested receiver"
    Assert-Contract ($download.Contains("-ArtifactId '`${{ needs.build.outputs.artifact_id }}'")) `
        "artifact consumer '$job' does not bind the immutable build artifact ID"
    Assert-Contract ($download.Contains("-ExpectedDigest '`${{ needs.build.outputs.artifact_digest }}'")) `
        "artifact consumer '$job' does not bind the build artifact digest"
    Assert-Contract (-not $download.Contains('continue-on-error: true')) `
        "artifact consumer '$job' can pass without its required build output"
    Assert-Contract ($download.Contains('timeout-minutes: 25')) `
        "artifact consumer '$job' does not reserve the bounded retry and extraction budget"
    Assert-Contract (-not $consumer.Contains('Retry build artifact download')) `
        "artifact consumer '$job' retains the immediate retry that caused repeated 403 responses"
    Assert-Contract ($header -match '(?m)^      actions: read\s*$') `
        "artifact consumer '$job' lacks actions:read for the archive endpoint"
}

$testConsumer = Get-JobBlock -WorkflowText $validation -Job 'test-net10-sharded'
Assert-Contract ($testConsumer.Contains('enum CoverageDisposition')) `
    'coverage execution state is represented by string comparisons instead of a closed enum'
Assert-Contract ($testConsumer.Contains(
        'COVERAGE_RUN_WITHOUT_INSTRUMENTATION: ${{ needs.select-shards.outputs.coverage_run_without_instrumentation }}')) `
    'shard jobs do not consume the selected RunWithoutCoverage set'
Assert-Contract ($testConsumer.Contains(
        '$coverageDisposition = [CoverageDisposition]::RunWithoutCoverage')) `
    'shard execution cannot select the explicit RunWithoutCoverage state'
Assert-Contract ($testConsumer.Contains(
        '$coverageDisposition -eq [CoverageDisposition]::Instrument')) `
    'forced coverage is not authorized exclusively by the Instrument state'
Assert-Contract ($testConsumer.Contains(
        'coverage disposition for ''$shardName'' is both Carried and RunWithoutCoverage')) `
    'overlapping coverage dispositions do not fail closed'
Assert-Contract ($testConsumer.Contains(
        'coverage disposition ''$coverageDisposition'' is invalid for coverage-producing shard ''$shardName''')) `
    'non-instrumented coverage dispositions are not bounded to heavy/timing shards'

# Parse the exact embedded pwsh program after replacing GitHub's expression tokens. Text searches
# prove the wiring is present; the language parser proves a later YAML edit cannot leave that
# critical disposition code syntactically dead while the contract still sees its strings.
$testRunStep = Get-StepBlock -JobBlock $testConsumer -Step 'Run tests (sharded) with coverage'
Assert-Contract ([bool] $testRunStep) `
    'the sharded test execution step is absent'
$testStepLines = [Regex]::Split($testRunStep, '\r?\n')
$testRunLine = [Array]::IndexOf($testStepLines, '        run: |')
Assert-Contract ($testRunLine -ge 0) `
    'the sharded test step has no literal PowerShell run block to validate'
if ($testRunLine -ge 0) {
    $testRunBody = [System.Collections.Generic.List[string]]::new()
    for ($i = $testRunLine + 1; $i -lt $testStepLines.Count; $i++) {
        $line = $testStepLines[$i]
        if ($line -and -not $line.StartsWith('          ', [StringComparison]::Ordinal)) { break }
        [void] $testRunBody.Add($(if ($line.Length -ge 10) { $line.Substring(10) } else { '' }))
    }
    $testRunScript = $testRunBody -join "`n"
    $testRunScript = [Regex]::Replace($testRunScript, '\$\{\{[^\r\n]*?\}\}', 'placeholder')
    $parseTokens = $null
    $parseErrors = $null
    [void] [System.Management.Automation.Language.Parser]::ParseInput(
        $testRunScript, [ref] $parseTokens, [ref] $parseErrors)
    $parseErrorMessages = @($parseErrors | ForEach-Object { $_.Message }) -join '; '
    Assert-Contract (@($parseErrors).Count -eq 0) `
        "the embedded sharded-test PowerShell has syntax errors: $parseErrorMessages"

    # Execute the exact checked-in decision block for the failure that motivated this state. This
    # is intentionally extracted from the workflow rather than restating its algorithm in the
    # test: Integration D must be recognized as heavy, consume the selected disposition, and leave
    # XPlat instrumentation disabled. The dotnet invocation begins immediately after the slice.
    $decisionStart = $testRunScript.IndexOf('$heavyShards = @(', [StringComparison]::Ordinal)
    $decisionEnd = $testRunScript.IndexOf('$dotnetArgs = @(', [StringComparison]::Ordinal)
    Assert-Contract ($decisionStart -ge 0 -and $decisionEnd -gt $decisionStart) `
        'the executable coverage-decision block could not be isolated from the workflow'
    if ($decisionStart -ge 0 -and $decisionEnd -gt $decisionStart) {
        $decisionBody = $testRunScript.Substring($decisionStart, $decisionEnd - $decisionStart)
        $decisionBody = $decisionBody.Replace("'placeholder' -eq 'true'", "'true' -eq 'true'")
        $decisionProof = @"
`$shardName = 'Integration D'
`$env:COVERAGE_CARRIED = '[]'
`$env:COVERAGE_RUN_WITHOUT_INSTRUMENTATION = '["Integration D"]'
$decisionBody
[pscustomobject]@{
    Heavy = `$heavyShard
    CollectCoverage = `$collectCoverage
    Disposition = [string] `$coverageDisposition
}
"@
        $decisionResult = @(& ([scriptblock]::Create($decisionProof))) | Select-Object -Last 1
        Assert-Contract ([bool] $decisionResult.Heavy) `
            'the production workflow no longer recognizes Integration D as a heavy shard'
        Assert-Contract (-not [bool] $decisionResult.CollectCoverage) `
            'RunWithoutCoverage still adds instrumentation to Integration D'
        Assert-Contract ($decisionResult.Disposition -ceq 'RunWithoutCoverage') `
            'Integration D did not consume the typed RunWithoutCoverage disposition'
    }
}
foreach ($stepName in @(
    'Upload test-impact digest',
    'Upload coverage + test results',
    'Upload crash evidence',
    'Upload compact test diagnostics')) {
    $upload = Get-StepBlock -JobBlock $testConsumer -Step $stepName
    Assert-Contract ($upload.Contains("if: always() && steps.download-build-artifacts.outcome == 'success'")) `
        "'$stepName' can amplify an artifact-service failure after the required download failed"
}
# THE MODEL-INVENTORY SWEEPS AND CONFORMANCE WINDOWS ARE SHARDS. As two bespoke jobs they ran all
# 45 workloads on every validating run whatever the change touched, and nothing about them could
# be selected, reused or imported. Keeping them in .github/test-shards.yml is what lets the map,
# the delta planner and the audit treat them like every other shard, so their return as separate
# jobs - or an entry losing the configuration that bounds it - is a regression.
foreach ($retired in @('parameter-enumeration-sweep', 'model-shape-conformance-windows')) {
    Assert-Contract (-not [Regex]::IsMatch($validation, "(?m)^  $([Regex]::Escape($retired)):\s*\r?$")) `
        "'$retired' is back as a job that runs outside shard selection"
}
$shardList = Get-Content -LiteralPath $ShardManifest -Raw
$shardEntries = [Regex]::Split($shardList, '(?m)^  - name: ') | Select-Object -Skip 1
function Get-ShardEntry([string] $Name) {
    $shardEntries | Where-Object { $_.StartsWith($Name + "`n") -or $_.StartsWith($Name + "`r`n") } | Select-Object -First 1
}
$expectedInventoryShards = [System.Collections.Generic.List[object]]::new()
foreach ($k in 0..7) {
    [void] $expectedInventoryShards.Add(@{ Name = "Sweep - ParameterCountContractTests $k/8"; Filter = 'FullyQualifiedName~ParameterCountContractTests'; Env = @("AIDOTNET_PARAMETER_COUNT_SHARD: '$k'"); MustCover = 'src/NeuralNetworks/Layers/*' })
}
[void] $expectedInventoryShards.Add(@{ Name = 'Sweep - ParameterChunkParityTests'; Filter = 'FullyQualifiedName~ParameterChunkParityTests'; Env = @(); MustCover = 'src/NeuralNetworks/Layers/*' })
[void] $expectedInventoryShards.Add(@{ Name = 'Sweep - ParameterEnumerationParityTests'; Filter = 'FullyQualifiedName~ParameterEnumerationParityTests'; Env = @() })
foreach ($offset in (0..34 | ForEach-Object { $_ * 5 })) {
    [void] $expectedInventoryShards.Add(@{
        Name = "Conformance - VisionLanguage offset $offset"; Filter = 'FullyQualifiedName~ModelContractConformanceTests'
        MustCover = '*/VisionLanguage/*'
        Env = @("ADNSHAPE_CONF_NAMESPACE: 'VisionLanguage'", "ADNSHAPE_CONF_OFFSET: '$offset'", "ADNSHAPE_CONF_BUDGET: '5'") })
}
foreach ($expected in $expectedInventoryShards) {
    $entry = Get-ShardEntry $expected.Name
    Assert-Contract ([bool] $entry) "inventory shard '$($expected.Name)' is missing from test-shards.yml"
    if (-not $entry) { continue }
    Assert-Contract ($entry.Contains("filter: '$($expected.Filter)'")) "inventory shard '$($expected.Name)' has the wrong filter"
    Assert-Contract ($entry -match '(?m)^    heavy: true\s*$') "inventory shard '$($expected.Name)' is not on the heavy path"
    $hangLimit = [regex]::Match($entry, '(?m)^    hangTimeout: (?<minutes>[1-9][0-9]*)min[ \t]*\r?$')
    Assert-Contract ($hangLimit.Success -and [int] $hangLimit.Groups['minutes'].Value -gt 5) `
        "inventory shard '$($expected.Name)' must declare a hang timeout greater than the 5-minute default"
    if ($expected.ContainsKey('MustCover')) {
        Assert-Contract ($entry.Contains("    mustCover: ['$($expected.MustCover)']")) `
            "inventory shard '$($expected.Name)' can become selectable without its coverage reaching the model code it tests"
        Assert-Contract ($entry.Contains('    coverageIncludeDirectory: tests/AiDotNet.ParameterSweepWorker/bin/Release/net10.0')) `
            "inventory shard '$($expected.Name)' does not instrument the worker its models run in"
    }
    foreach ($pair in $expected.Env) {
        Assert-Contract ($entry.Contains("      $pair")) "inventory shard '$($expected.Name)' lost its env '$pair'"
    }
}

Assert-Contract ($map.Contains('-CoverageRequirementsFile $requirements')) `
    'the map build ignores mustCover, so a shard whose subject coverage never reached can become selectable'

# The four Sweep-category surveys no job ran (every shard excludes Category=Sweep; the old sweep job named
# only its own three classes). They run in one gating heavy shard now; losing any of them silently
# returns it to never running.
$surveys = Get-ShardEntry 'Sweep - Layer and Model Contract Surveys'
Assert-Contract ([bool] $surveys) 'the contract-survey sweeps have no shard and would never run'
Assert-Contract ($surveys -match '(?m)^    heavy: true\s*$') 'the contract-survey shard is not on the heavy path'
Assert-Contract ($surveys -match '(?m)^    hangTimeout: 35min\s*$') 'the contract-survey shard lost its 35-minute hang timeout'
foreach ($survey in @('LayerOverrideRedundancyTests', 'LayerParameterSurfaceTests', 'ContractShadowSweepTests', 'ModelBaseCoverageTests')) {
    Assert-Contract ($surveys -and $surveys.Contains("FullyQualifiedName~$survey")) "the survey sweep '$survey' is no longer selected by any shard"
}

$shardRun = Get-StepBlock -JobBlock $testConsumer -Step 'Run tests (sharded) with coverage'
Assert-Contract ($shardRun.Contains('SHARD_ENV: ${{ toJSON(matrix.shard.env) }}')) `
    'the shard step does not receive the entry env as data'
Assert-Contract ($shardRun.Contains('& ./.github/scripts/Set-ShardEnvironment.ps1 -Json $env:SHARD_ENV')) `
    'the shard step does not apply the entry env through the validated helper'
Assert-Contract ($shardRun.Contains("(`$heavyShards -contains `$shardName) -or (`$env:SHARD_HEAVY -eq 'true')")) `
    'a shard declaring heavy: true does not get the heavy path'
Assert-Contract ($shardRun.Contains('SHARD_COVERAGE_INCLUDE: ${{ matrix.shard.coverageIncludeDirectory }}') -and
        $shardRun.Contains('& ./.github/scripts/New-CoverageRunSettings.ps1 -Base coverlet.runsettings')) `
    'a shard whose models run in the worker never instruments the worker'
Assert-Contract ($shardRun.Contains("'--blame-hang-timeout', `$hangTimeout,")) `
    'a shard hangTimeout is not passed to the blame-hang collector'
$shardRetry = Get-StepBlock -JobBlock $testConsumer -Step 'Rerun PR-new failures once'
Assert-Contract ($shardRetry.Contains('& ./.github/scripts/Set-ShardEnvironment.ps1 -Json $env:SHARD_ENV')) `
    'a targeted retry runs without the shard env and would walk the whole inventory'
Assert-Contract ($selectStep.Contains("`$declared.PSObject.Properties['heavy'] -and `$declared.heavy -eq `$true")) `
    'a map-feeding run does not treat heavy: true entries as no-coverage shards'

$promotion = Get-JobBlock -WorkflowText $validation -Job 'promote-ci-test-analysis'
Assert-Contract ($promotion.Contains("needs.validation-source.outputs.reuse == 'true'")) `
    'certified artifact promotion is not restricted to exact-tree reuse'
Assert-Contract ($promotion.Contains('fromJSON(needs.validation-source.outputs.reused_requires_validation)')) `
    'non-runtime reuse can still attempt to promote test artifacts that do not exist'

$validationGate = Get-JobBlock -WorkflowText $validation -Job 'validation-gate'
Assert-Contract ($validationGate.Contains('./tools/TestImpact/Assert-CiGate.ps1')) `
    'the validation-only boundary bypasses the typed gate policy'
Assert-Contract ($validationGate.Contains('-Stage Validation')) `
    'the validation-only boundary invokes the complete gate by mistake'
foreach ($job in @('select-shards') + $validationJobs) {
    Assert-Contract ($validationGate -match "(?m)^\s+- $([Regex]::Escape($job))\s*$") `
        "Validation Gate does not depend on validation job '$job'"
}

$partialCertificate = Get-JobBlock -WorkflowText $validation -Job 'certify-expensive-validation'
Assert-Contract ($partialCertificate.Contains("needs.validation-gate.result == 'success'")) `
    'validation-only evidence can be published before Validation Gate succeeds'
Assert-Contract ($partialCertificate.Contains('-Scope Validation')) `
    'validation-only evidence is minted with the wrong typed scope'
Assert-Contract ($partialCertificate.Contains('ci-expensive-validation-certificate-')) `
    'validation-only evidence is not published as a distinct artifact'

$validationCertificate = Get-JobBlock -WorkflowText $validation -Job 'certify-validation'
Assert-Contract ($validationCertificate.Contains("needs.ci-gate.result == 'success'")) `
    'complete validation evidence can be published before CI Gate succeeds'
Assert-Contract ($validationCertificate.Contains('-Scope Complete')) `
    'complete validation evidence is minted with the wrong typed scope'
Assert-Contract ($validationCertificate.Contains('ci-validation-certificate-')) `
    'complete validation evidence is not published as a distinct artifact'
Assert-Contract ($validationCertificateWriterText.Contains('enum CiValidationCertificateScope')) `
    'certificate scope is represented by strings instead of a closed enum'
Assert-Contract ($validationCertificateWriterText.Contains('requiresValidation = $CertificateRequiresValidation')) `
    'the certificate omits the runtime-validation decision'
Assert-Contract ($resolver.Contains('reused_requires_validation: ${{ steps.resolve.outputs.reused_requires_validation || steps.defaults.outputs.reused_requires_validation }}')) `
    'exact-tree resolver does not publish the reused validation scope'
Assert-Contract ($validationReuseResolverText.Contains('$requiresValidation = $true')) `
    'legacy certificates are not conservatively treated as full runtime validation'

$gate = Get-JobBlock -WorkflowText $validation -Job 'ci-gate'
foreach ($job in @('validation-gate', 'select-shards') + $expensiveJobs) {
    Assert-Contract ($gate -match "(?m)^\s+- $([Regex]::Escape($job))\s*$") `
        "CI Gate does not depend on expensive job '$job', so its failure cannot block validation"
}
Assert-Contract ($gate.Contains('./tools/TestImpact/Assert-CiGate.ps1')) `
    'the complete CI boundary bypasses the typed gate policy'
Assert-Contract ($gate.Contains('-ReuseScope ''${{ needs.validation-source.outputs.reuse_scope }}''')) `
    'CI Gate does not consume the typed reuse scope'
Assert-Contract ($ciGatePolicyText.Contains('enum CiValidationReuseScope')) `
    'CI Gate reuse control flow is represented by strings instead of a closed enum'
Assert-Contract ($ciGatePolicyText.Contains('([CiValidationReuseScope]::Validation)')) `
    'CI Gate has no validation-only reuse mode'
Assert-Contract ($ciGatePolicyText.Contains("Add-RequiredSuccess `$requirements 'codeql' `$codeql")) `
    'validation-only reuse does not rerun CodeQL'
Assert-Contract ($ciGatePolicyText.Contains("Add-RequiredSuccess `$requirements 'sonarcloud' `$sonar")) `
    'validation-only reuse does not rerun SonarCloud'

# SonarCloud Analysis is still a required repository status. Its job must succeed cheaply for a
# non-runtime PR, but no setup, cache, download, restore, scanner, or build step may execute there.
$sonarJob = Get-JobBlock -WorkflowText $validation -Job 'sonarcloud'
Assert-Contract ($sonarJob.Contains('- name: Report non-runtime validation')) `
    'the required SonarCloud status has no lightweight non-runtime success path'
foreach ($stepName in @(
    'Set up JDK 17', 'Checkout code', 'Setup .NET 10.0', 'Cache NuGet packages',
    'Cache SonarCloud packages', 'Cache SonarCloud scanner', 'Install SonarCloud scanner',
    'Download current-run coverage artifacts', 'Restore dependencies', 'Begin SonarCloud analysis',
    'Build source generator first', 'Build (Release)', 'End SonarCloud analysis'
)) {
    $step = Get-StepBlock -JobBlock $sonarJob -Step $stepName
    Assert-Contract ($step.Contains('fromJSON(env.EFFECTIVE_REQUIRES_VALIDATION')) `
        "SonarCloud step '$stepName' can run for a non-runtime change"
}
$certifiedCoverage = Get-StepBlock -JobBlock $sonarJob -Step 'Download certified coverage artifacts'
Assert-Contract ($certifiedCoverage.Contains("needs.validation-source.outputs.reuse == 'true'")) `
    'a quality-only landed run cannot consume the certified PR coverage'
Assert-Contract ($certifiedCoverage.Contains('run-id: ${{ needs.validation-source.outputs.run_id }}')) `
    'certified coverage download is not bound to the evidence source run'

# A repository variable left the shipped feature permanently in shadow mode. Certification is the
# authorization boundary now; no second switch may silently restore the full matrix.
Assert-Contract (-not $validation.Contains('TEST_IMPACT_MODE')) `
    'TEST_IMPACT_MODE shadow/enforce switching is still present'
Assert-Contract ($certifiedMapResolverText.Contains('--name certified-shard-map')) `
    'PR selection does not consume a certified map artifact'
Assert-Contract ($validation.Contains('candidate-shard-map')) `
    'coverage audit runs do not consume an explicit candidate map artifact'
Assert-Contract (
    $validation -match '(?s)if \[ "\$FORCE_COVERAGE" = ''true'' \].*?--name candidate-shard-map.*?else' -and
    $certifiedMapResolverText.Contains('--name certified-shard-map')) `
    'candidate and certified artifacts are not separated at the execution boundary'

# Only an explicit audited miss or an unclassified failure revokes older positive evidence.
# Cancellation and a typed zero-miss/no-reduction result are neutral, so queue cleanup cannot
# silently disable selective CI across the repository.
Assert-Contract ($mapDownload.Contains('./tools/TestImpact/Resolve-CertifiedShardMap.ps1')) `
    'PR selection bypasses the executable certified-map resolver'
Assert-Contract ($mapDownload.Contains('-ResultFile map-resolution.json')) `
    'PR selection does not consume the resolver decision explicitly'
Assert-Contract ($mapDownload.Contains('-MapBranch "$MAP_BRANCH"')) `
    'proof-branch PRs cannot resolve a certified map from their own target branch'
Assert-Contract ($validation.Contains('./tools/TestImpact/Resolve-CertifiedShardMap.ps1 -SelfTest')) `
    'the certified-map revocation policy is not executed against its adversarial fixtures'
Assert-Contract ($certifiedMapResolverText.Contains('enum ShardMapAuditDisposition')) `
    'map audit disposition is represented by strings instead of a closed enum'
Assert-Contract ($certifiedMapResolverText.Contains('([ShardMapAuditDisposition]::RevokedMiss)')) `
    'an explicit audited selection miss does not revoke older evidence'
Assert-Contract ($certifiedMapResolverText.Contains("if (`$candidate.Conclusion -ceq 'cancelled') { continue }")) `
    'a cancelled maintenance run can still revoke older positive evidence'
Assert-Contract ($certifiedMapResolverText.Contains('UnclassifiedFailure')) `
    'an old failed audit without typed evidence does not fail closed'

# The map builder publishes a candidate separately from its certificate. Historical replay is
# preferred; typed FreshCoverage is limited to bootstrap/no-reduction re-seeding and cannot hide a
# historical miss. Reusing one artifact name for both states recreates the unproven-rollout bug.
Assert-Contract ($map.Contains('name: candidate-shard-map')) `
    'map workflow does not publish candidate-shard-map'
Assert-Contract ($map.Contains('name: certified-shard-map')) `
    'map workflow does not publish certified-shard-map'
Assert-Contract ($map.Contains('name: shard-map-audit-decision')) `
    'map workflow does not publish the typed audit disposition'
Assert-Contract ($map.Contains("always() && steps.source.outputs.found == 'true' && hashFiles('map-audit-decision.json') != ''")) `
    'map workflow cannot publish revocation evidence after a miss fails the audit step'
Assert-Contract ($map.Contains('certification.json')) `
    'certified map artifact has no provenance record'
Assert-Contract ($map.Contains('./tools/TestImpact/Measure-SelectionMiss.ps1 -SelfTest')) `
    'map certification can run without first proving that skipped failures are detected'
Assert-Contract ($map.Contains('./tools/TestImpact/New-ShardMapCertificate.ps1 -SelfTest')) `
    'the shipping certification policy is not tested before use'
Assert-Contract ($map.Contains('./tools/TestImpact/Test-ValidationReuseModes.ps1')) `
    'the map workflow does not exercise post-merge certificate modes'
Assert-Contract ($map.Contains('found=false')) `
    'automatic no-source bootstrap is still represented as a workflow failure'
Assert-Contract ($map.Contains('[ "$conclusion" != ''cancelled'' ]')) `
    'a cancelled workflow with a fully completed shard matrix cannot be harvested'
Assert-Contract ($map.Contains('.status == "completed" and (.conclusion == "success" or .conclusion == "failure")')) `
    'cancelled coverage workflows are accepted without proving every shard independently completed'
Assert-Contract ($map.Contains("candidate_ready: `${{ steps.source.outputs.found }}")) `
    'map workflow does not expose whether this invocation produced a candidate'
Assert-Contract ($map.Contains("if: steps.audit.outputs.certified == 'true'")) `
    'certified artifact upload is not gated by the audit decision'
$mapBuildJob = Get-JobBlock -WorkflowText $map -Job 'build-map'
$auditStep = Get-StepBlock -JobBlock $mapBuildJob -Step 'Audit selection against the source run'
$certificateInvocation = "& (Join-Path `$auditTools 'New-ShardMapCertificate.ps1')"
Assert-Contract ($auditStep.Contains($certificateInvocation)) `
    'the workflow does not execute the tested certification policy'
Assert-Contract ($auditStep.Contains('-Basis HistoricalReplay')) `
    'the map lifecycle lost its non-circular historical replay path'
Assert-Contract ($auditStep.Contains('-Basis FreshCoverage')) `
    'a complete fresh coverage map cannot break a bootstrap/no-reduction deadlock'
Assert-Contract ($mapCertificateText.Contains('enum ShardMapCertificationBasis')) `
    'map certification basis is represented by strings instead of a closed enum'
Assert-Contract ($auditStep.Contains('-MapFile shard-map.json')) `
    'fresh coverage certification does not audit the candidate map that was just measured'
Assert-Contract ($auditStep.Contains('-CurrentChangeBaseSha $sourceSha')) `
    'historically merged selector-control changes still wedge every later selection audit'
Assert-Contract ($auditStep.Contains('-ShardManifestFile $auditManifest') -and
        $auditStep.Contains("yq -o=json -I=0 '.shard' .github/test-shards.yml")) `
    'the nightly audit does not replay test-source routing, so its misses are never measured'
Assert-Contract ($auditStep.Contains('if ([int] $historicalDecision.missCount -gt 0)')) `
    'a historical selection miss can be hidden by fresh-coverage fallback'
Assert-Contract ($auditStep.Contains('(-not $certified -and $auditExit -eq 0)')) `
    'fresh-coverage fallback can run after a historical miss failed the audit'
Assert-Contract ($auditStep.Contains('-CandidateMapRunId ([long] $env:GITHUB_RUN_ID)')) `
    'fresh coverage evidence is not bound to the run that published its candidate map'
Assert-Contract ($map -match "if \[ '.*needs\.build-map\.result.*' = 'success' \] && \[ '.*candidate_ready.*' = 'true' \]") `
    'coverage dispatch can cite the current map run when no candidate was produced'

# coverage-run has no checkout by design. Every gh command in that job must therefore identify the
# repository explicitly instead of asking git to infer it from a nonexistent worktree.
$coverageRun = Get-JobBlock -WorkflowText $map -Job 'coverage-run'
$runListCommands = @(Get-ContinuedShellCommand -Text $coverageRun -CommandPattern `
    '^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\$\()?gh run list\b')
Assert-Contract ($runListCommands.Count -ge 2) `
    'coverage-run no longer performs both source and in-flight run lookups'
foreach ($command in $runListCommands) {
    Assert-Contract ($command.Contains('--repo "$GITHUB_REPOSITORY"')) `
        "coverage-run has an unscoped gh run list command: $command"
}

$workflowRunCommands = @(Get-ContinuedShellCommand -Text $coverageRun -CommandPattern `
    '^\s*gh workflow run\b')
Assert-Contract ($workflowRunCommands.Count -ge 2) `
    'coverage-run no longer covers both mapped and bootstrap dispatches'
foreach ($command in $workflowRunCommands) {
    Assert-Contract ($command.Contains('--repo "$GITHUB_REPOSITORY"')) `
        "coverage-run has an unscoped gh workflow run command: $command"
}

$apiCommands = @(Get-ContinuedShellCommand -Text $coverageRun -CommandPattern `
    '^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\$\()?gh api\b')
Assert-Contract ($apiCommands.Count -ge 1) `
    'coverage-run no longer validates candidate artifact availability'
foreach ($command in $apiCommands) {
    Assert-Contract ($command.Contains('repos/${GITHUB_REPOSITORY}/')) `
        "coverage-run has a gh api command without an explicit repository endpoint: $command"
}

if ($failures.Count -gt 0) {
    Write-Host 'CI impact workflow contract FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host "CI impact workflow contract passed ($($expensiveJobs.Count) expensive jobs gated)."
exit 0
