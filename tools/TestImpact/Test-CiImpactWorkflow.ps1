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
    [string] $MapWorkflow = '.github/workflows/test-impact-map.yml'
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

$validation = Get-Content -LiteralPath $ValidationWorkflow -Raw
$map = Get-Content -LiteralPath $MapWorkflow -Raw

# Every job in this list consumes meaningful runner time. Dependency-based incidental skipping is
# not enough: a dependency can be removed during a refactor while the job remains runnable. Each
# job must explicitly depend on validation-source and consume its JSON boolean decision.
$expensiveJobs = @(
    'codeql',
    'build',
    'build-compat',
    'select-shards',
    'test-net10-sharded',
    'parameter-enumeration-sweep',
    'model-shape-conformance-windows',
    'test-regression-analysis',
    'sonarcloud',
    'size-check',
    'ci-test-analysis'
)

foreach ($job in $expensiveJobs) {
    $block = Get-JobBlock -WorkflowText $validation -Job $job
    if (-not $block) { continue }
    $header = Get-JobHeader -JobBlock $block
    Assert-Contract ([bool] $header) `
        "expensive job '$job' has no job-level header before its steps"
    Assert-Contract (Test-JobDependency -JobHeader $header -Dependency 'validation-source') `
        "expensive job '$job' does not explicitly depend on validation-source"
    $jobIf = [Regex]::Match($header, '(?m)^    if:\s*(?<value>[^\r\n]+)\s*$')
    Assert-Contract $jobIf.Success `
        "expensive job '$job' has no job-level execution condition"
    Assert-Contract ($jobIf.Success -and
        $jobIf.Groups['value'].Value.Contains('fromJSON(needs.validation-source.outputs.execute_expensive)')) `
        "expensive job '$job' does not consume the typed execute_expensive decision"
}

$resolver = Get-JobBlock -WorkflowText $validation -Job 'validation-source'
Assert-Contract ($resolver.Contains('steps.resolve.outputs.execute_expensive || steps.defaults.outputs.execute_expensive')) `
    'validation-source does not publish execute_expensive'
Assert-Contract ($resolver.Contains("echo 'execute_expensive=true'")) `
    'the resolver lacks a fail-closed full-validation default'
Assert-Contract ($resolver.Contains("echo 'execute_expensive=false'")) `
    'the resolver never suppresses expensive work after exact-tree certification'
$resolveStep = Get-StepBlock -JobBlock $resolver -Step 'Resolve exact-tree PR run'
Assert-Contract ([bool] $resolveStep) `
    'the exact-tree resolver step is absent'
Assert-Contract ($resolveStep.Contains('continue-on-error: true')) `
    'an unexpected resolver failure blocks dependents instead of retaining fail-closed defaults'

$selectorJob = Get-JobBlock -WorkflowText $validation -Job 'select-shards'
$selfTestStep = Get-StepBlock -JobBlock $selectorJob -Step 'Verify the impact tooling'
Assert-Contract ([bool] $selfTestStep) `
    'select-shards no longer runs the impact tooling self-tests'
Assert-Contract (-not $selfTestStep.Contains('continue-on-error: true')) `
    'checked-in impact tooling can fail its self-tests while CI remains green'

$promotion = Get-JobBlock -WorkflowText $validation -Job 'promote-ci-test-analysis'
Assert-Contract ($promotion.Contains("needs.validation-source.outputs.reuse == 'true'")) `
    'certified artifact promotion is not restricted to exact-tree reuse'

$validationCertificate = Get-JobBlock -WorkflowText $validation -Job 'certify-validation'
Assert-Contract ($validationCertificate.Contains("needs.ci-gate.result == 'success'")) `
    'validation certificate can be published before the required CI Gate succeeds'
Assert-Contract ($validationCertificate.Contains('ci-validation-certificate-')) `
    'validation certificate is not published as a distinct artifact'
Assert-Contract ($resolver.Contains('ci-validation-certificate-')) `
    'exact-tree resolver does not require the CI Gate validation certificate'

$gate = Get-JobBlock -WorkflowText $validation -Job 'ci-gate'
foreach ($job in $expensiveJobs) {
    Assert-Contract ($gate -match "(?m)^\s+- $([Regex]::Escape($job))\s*$") `
        "CI Gate does not depend on expensive job '$job', so its failure cannot block validation"
}
Assert-Contract ($gate -match 'required=\("validation-source:\$SOURCE_RESULT"\)') `
    'the reuse-aware gate must start with only validation-source as universally required'
Assert-Contract ($gate -match '(?s)if \[ "\$REUSED_VALIDATION" = "true" \].*?promote-ci-test-analysis.*?else.*?build:\$BUILD_RESULT.*?sonarcloud:\$SONAR_RESULT') `
    'the gate does not exclude build and SonarCloud from certified-reuse mode'

# A repository variable left the shipped feature permanently in shadow mode. Certification is the
# authorization boundary now; no second switch may silently restore the full matrix.
Assert-Contract (-not $validation.Contains('TEST_IMPACT_MODE')) `
    'TEST_IMPACT_MODE shadow/enforce switching is still present'
Assert-Contract ($validation.Contains('certified-shard-map')) `
    'PR selection does not consume a certified map artifact'
Assert-Contract ($validation.Contains('candidate-shard-map')) `
    'coverage audit runs do not consume an explicit candidate map artifact'
Assert-Contract ($validation -match '(?s)if \[ "\$FORCE_COVERAGE" = ''true'' \].*?--name candidate-shard-map.*?else.*?--name certified-shard-map') `
    'candidate and certified artifacts are not separated at the execution boundary'

# The map builder publishes a new candidate, but only the previous candidate that was exercised by
# a complete coverage run may become certified. Reusing one artifact name for both states recreates
# the original unproven-rollout bug.
Assert-Contract ($map.Contains('name: candidate-shard-map')) `
    'map workflow does not publish candidate-shard-map'
Assert-Contract ($map.Contains('name: certified-shard-map')) `
    'map workflow does not publish certified-shard-map'
Assert-Contract ($map.Contains('certification.json')) `
    'certified map artifact has no provenance record'
Assert-Contract ($map.Contains('found=false')) `
    'automatic no-source bootstrap is still represented as a workflow failure'
Assert-Contract ($map.Contains("candidate_ready: `${{ steps.source.outputs.found }}")) `
    'map workflow does not expose whether this invocation produced a candidate'
Assert-Contract ($map.Contains("if: steps.audit.outputs.certified == 'true'")) `
    'certified artifact upload is not gated by the audit decision'
Assert-Contract ($map -match '(?s)if \(\$auditExit -eq 0.*?\$a\.WouldSkip -gt 0.*?\$a\.Failed -gt 0\)') `
    'map certification does not require a real failing-shard opportunity as well as reduction'
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
