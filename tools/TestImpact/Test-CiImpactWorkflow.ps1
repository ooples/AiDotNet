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
    Assert-Contract ($block -match '(?m)^\s+needs:(?:[^\r\n]*validation-source|\s*\r?\n(?:\s+-[^\r\n]*\r?\n)*\s+- validation-source\s*$)') `
        "expensive job '$job' does not explicitly depend on validation-source"
    Assert-Contract ($block.Contains('fromJSON(needs.validation-source.outputs.execute_expensive)')) `
        "expensive job '$job' does not consume the typed execute_expensive decision"
}

$resolver = Get-JobBlock -WorkflowText $validation -Job 'validation-source'
Assert-Contract ($resolver.Contains('steps.resolve.outputs.execute_expensive || steps.defaults.outputs.execute_expensive')) `
    'validation-source does not publish execute_expensive'
Assert-Contract ($resolver.Contains("echo 'execute_expensive=true'")) `
    'the resolver lacks a fail-closed full-validation default'
Assert-Contract ($resolver.Contains("echo 'execute_expensive=false'")) `
    'the resolver never suppresses expensive work after exact-tree certification'
Assert-Contract ($resolver.Contains('continue-on-error: true')) `
    'an unexpected resolver failure blocks dependents instead of retaining fail-closed defaults'

$selectorJob = Get-JobBlock -WorkflowText $validation -Job 'select-shards'
$selfTestStep = [Regex]::Match(
    $selectorJob,
    '(?ms)^\s+- name: Verify the impact tooling\s*$.*?(?=^\s+- name: |\z)').Value
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
Assert-Contract ($map -match "if \[ '.*needs\.build-map\.result.*' = 'success' \] && \[ '.*candidate_ready.*' = 'true' \]") `
    'coverage dispatch can cite the current map run when no candidate was produced'

# coverage-run has no checkout by design. Every gh command in that job must therefore identify the
# repository explicitly instead of asking git to infer it from a nonexistent worktree.
$coverageRun = Get-JobBlock -WorkflowText $map -Job 'coverage-run'
$repoTargets = [Regex]::Matches($coverageRun, '--repo\s+"\$GITHUB_REPOSITORY"').Count
Assert-Contract ($repoTargets -ge 3) `
    "coverage-run has only $repoTargets explicit repository target(s); run lookup, in-flight lookup, and dispatch all require one"

if ($failures.Count -gt 0) {
    Write-Host 'CI impact workflow contract FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host "CI impact workflow contract passed ($($expensiveJobs.Count) expensive jobs gated)."
exit 0
