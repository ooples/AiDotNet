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
    [string] $RequiredArtifactReceiver = 'tools/TestImpact/Receive-RequiredArtifact.ps1'
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
$selectorText = Get-Content -LiteralPath $Selector -Raw
$certifiedMapResolverText = Get-Content -LiteralPath $CertifiedMapResolver -Raw
$requiredArtifactReceiverText = Get-Content -LiteralPath $RequiredArtifactReceiver -Raw

# Every job in this list consumes meaningful runner time. Dependency-based incidental skipping is
# not enough: each job must explicitly consume both independent typed decisions -- whether exact
# PR validation is reusable, and whether this change can affect runtime/model/test behavior.
$expensiveJobs = @(
    'codeql',
    'build',
    'build-compat',
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
    Assert-Contract (Test-JobDependency -JobHeader $header -Dependency 'select-shards') `
        "expensive job '$job' does not explicitly depend on select-shards"
    $jobIf = [Regex]::Match($header, '(?m)^    if:\s*(?<value>[^\r\n]+)\s*$')
    Assert-Contract $jobIf.Success `
        "expensive job '$job' has no job-level execution condition"
    Assert-Contract ($jobIf.Success -and
        $jobIf.Groups['value'].Value.Contains('fromJSON(needs.validation-source.outputs.execute_expensive)')) `
        "expensive job '$job' does not consume the typed execute_expensive decision"
    if ($job -ne 'codeql') {
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

foreach ($job in @(
    'test-regression-analysis', 'sonarcloud', 'ci-test-analysis',
    'promote-ci-test-analysis', 'ci-gate', 'certify-validation'
)) {
    $header = Get-JobHeader -JobBlock (Get-JobBlock -WorkflowText $validation -Job $job)
    Assert-Contract ($header.Contains('always()') -and $header.Contains('!cancelled()')) `
        "job '$job' can keep an explicitly canceled run alive"
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
$selectorHeader = Get-JobHeader -JobBlock $selectorJob
Assert-Contract (Test-JobDependency -JobHeader $selectorHeader -Dependency 'validation-source') `
    'select-shards does not depend on validation-source'
Assert-Contract ($selectorHeader.Contains('fromJSON(needs.validation-source.outputs.execute_expensive)')) `
    'select-shards does not consume the typed execute_expensive decision'
Assert-Contract ($selectorHeader.Contains('requires_validation: ${{ steps.select.outputs.requires_validation }}')) `
    'select-shards does not publish its typed runtime-validation decision'
$selfTestStep = Get-StepBlock -JobBlock $selectorJob -Step 'Verify the impact tooling'
Assert-Contract ([bool] $selfTestStep) `
    'select-shards no longer runs the impact tooling self-tests'
Assert-Contract (-not $selfTestStep.Contains('continue-on-error: true')) `
    'checked-in impact tooling can fail its self-tests while CI remains green'
Assert-Contract ($selfTestStep.Contains('./tools/TestImpact/Receive-RequiredArtifact.ps1 -SelfTest')) `
    'the required-artifact transport policy is not executed against its self-tests'
Assert-Contract ($selfTestStep.Contains('./tools/TestImpact/Test-ValidationReuseModes.ps1')) `
    'the post-merge certificate modes are not exercised by primary CI'
$selectStep = Get-StepBlock -JobBlock $selectorJob -Step 'Select'
Assert-Contract (-not $selectStep.Contains('-AuditUnchangedMap')) `
    'ordinary PR selection was given the audit-only unchanged-map capability'
Assert-Contract ($selectStep.Contains('-ClassifyOnly')) `
    'non-runtime classification still depends on a coverage map being available'
Assert-Contract ($selectStep.Contains('-BaseSha $env:PR_BASE_SHA')) `
    'non-runtime classification does not use the exact PR base-to-head path set'
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
Assert-Contract ($selectorText.Contains('diff --no-renames --name-only')) `
    'path classification can hide the source side of a rename'

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

$artifactConsumers = @('test-net10-sharded', 'model-shape-conformance-windows')
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
foreach ($stepName in @(
    'Upload test-impact digest',
    'Upload coverage + test results',
    'Upload crash evidence',
    'Upload compact test diagnostics')) {
    $upload = Get-StepBlock -JobBlock $testConsumer -Step $stepName
    Assert-Contract ($upload.Contains("if: always() && steps.download-build-artifacts.outcome == 'success'")) `
        "'$stepName' can amplify an artifact-service failure after the required download failed"
}
$shapeConsumer = Get-JobBlock -WorkflowText $validation -Job 'model-shape-conformance-windows'
$shapeUpload = Get-StepBlock -JobBlock $shapeConsumer -Step 'Upload window report'
Assert-Contract ($shapeUpload.Contains("if: always() && steps.download-build-artifacts.outcome == 'success'")) `
    'shape-conformance can upload after its required build artifact was unavailable'

$promotion = Get-JobBlock -WorkflowText $validation -Job 'promote-ci-test-analysis'
Assert-Contract ($promotion.Contains("needs.validation-source.outputs.reuse == 'true'")) `
    'certified artifact promotion is not restricted to exact-tree reuse'
Assert-Contract ($promotion.Contains('fromJSON(needs.validation-source.outputs.reused_requires_validation)')) `
    'non-runtime reuse can still attempt to promote test artifacts that do not exist'

$validationCertificate = Get-JobBlock -WorkflowText $validation -Job 'certify-validation'
Assert-Contract ($validationCertificate.Contains("needs.ci-gate.result == 'success'")) `
    'validation certificate can be published before the required CI Gate succeeds'
Assert-Contract ((Test-JobDependency -JobHeader (Get-JobHeader -JobBlock $validationCertificate) `
        -Dependency 'select-shards')) `
    'validation certificate cannot bind the runtime-validation decision'
Assert-Contract ($validationCertificate.Contains('schemaVersion: 2')) `
    'validation certificate schema does not include the runtime-validation decision'
Assert-Contract ($validationCertificate.Contains('requiresValidation: $requiresValidation')) `
    'validation certificate omits the runtime-validation decision'
Assert-Contract ($validationCertificate.Contains('ci-validation-certificate-')) `
    'validation certificate is not published as a distinct artifact'
Assert-Contract ($resolver.Contains('ci-validation-certificate-')) `
    'exact-tree resolver does not require the CI Gate validation certificate'
Assert-Contract ($resolver.Contains('reused_requires_validation: ${{ steps.resolve.outputs.reused_requires_validation || steps.defaults.outputs.reused_requires_validation }}')) `
    'exact-tree resolver does not publish the reused validation scope'
Assert-Contract ($resolver.Contains('if .schemaVersion == 1 then true else .requiresValidation end')) `
    'legacy certificates are not conservatively treated as full runtime validation'
Assert-Contract ($resolver -match '(?s)if \[ "\$requires_validation" = ''true'' \]; then.*?analysis_id.*?coverage_count') `
    'runtime certificates no longer require analysis and coverage artifacts'

$gate = Get-JobBlock -WorkflowText $validation -Job 'ci-gate'
foreach ($job in @('select-shards') + $expensiveJobs) {
    Assert-Contract ($gate -match "(?m)^\s+- $([Regex]::Escape($job))\s*$") `
        "CI Gate does not depend on expensive job '$job', so its failure cannot block validation"
}
Assert-Contract ($gate -match 'required=\("validation-source:\$SOURCE_RESULT"\)') `
    'the reuse-aware gate must start with only validation-source as universally required'
Assert-Contract ($gate -match '(?s)if \[ "\$REUSED_VALIDATION" = "true" \].*?promote-ci-test-analysis.*?else.*?sonarcloud:\$SONAR_RESULT.*?build:\$BUILD_RESULT') `
    'the gate does not exclude build and SonarCloud from certified-reuse mode'
Assert-Contract ($gate.Contains('REQUIRES_VALIDATION: ${{ needs.select-shards.outputs.requires_validation }}')) `
    'CI Gate does not consume the runtime-validation decision'
Assert-Contract ($gate.Contains('REUSED_REQUIRES_VALIDATION: ${{ needs.validation-source.outputs.reused_requires_validation }}')) `
    'CI Gate does not consume the reused validation scope'
Assert-Contract ($gate -match '(?s)if \[ "\$REUSED_VALIDATION" = "true" \]; then.*?REUSED_REQUIRES_VALIDATION.*?promote-ci-test-analysis') `
    'CI Gate cannot reuse a non-runtime certificate without requiring absent test artifacts'
Assert-Contract ($gate -match '(?s)required\+=\("select-shards:\$SELECT_RESULT" "codeql:\$CODEQL_RESULT" "sonarcloud:\$SONAR_RESULT"\).*?if \[ "\$REQUIRES_VALIDATION" != "false" \].*?build:\$BUILD_RESULT') `
    'CI Gate does not allow a successful selector to suppress expensive jobs for non-runtime changes'
Assert-Contract ($gate.Contains('Mode: non-runtime-only change; model and test validation suppressed.')) `
    'CI Gate does not report the non-runtime validation mode'

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
    Assert-Contract ($step.Contains('fromJSON(needs.select-shards.outputs.requires_validation')) `
        "SonarCloud step '$stepName' can run for a non-runtime change"
}

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

# A later map audit is also a revocation boundary. If it fails or withholds certification after a
# miss, selection must fail closed instead of scanning backward until an older certificate happens
# to download successfully.
$mapDownload = Get-StepBlock -JobBlock $selectorJob -Step 'Download the shard map'
Assert-Contract ($mapDownload.Contains('./tools/TestImpact/Resolve-CertifiedShardMap.ps1')) `
    'PR selection bypasses the executable certified-map resolver'
Assert-Contract ($mapDownload.Contains('-ResultFile map-resolution.json')) `
    'PR selection does not consume the resolver decision explicitly'
Assert-Contract ($validation.Contains('./tools/TestImpact/Resolve-CertifiedShardMap.ps1 -SelfTest')) `
    'the certified-map revocation policy is not executed against its adversarial fixtures'

# The map builder publishes a new candidate, but only the previous candidate that was exercised by
# a complete coverage run may become certified. Reusing one artifact name for both states recreates
# the original unproven-rollout bug.
Assert-Contract ($map.Contains('name: candidate-shard-map')) `
    'map workflow does not publish candidate-shard-map'
Assert-Contract ($map.Contains('name: certified-shard-map')) `
    'map workflow does not publish certified-shard-map'
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
Assert-Contract ($map.Contains("candidate_ready: `${{ steps.source.outputs.found }}")) `
    'map workflow does not expose whether this invocation produced a candidate'
Assert-Contract ($map.Contains("if: steps.audit.outputs.certified == 'true'")) `
    'certified artifact upload is not gated by the audit decision'
$mapBuildJob = Get-JobBlock -WorkflowText $map -Job 'build-map'
$auditStep = Get-StepBlock -JobBlock $mapBuildJob -Step 'Audit selection against the source run'
$auditRead = '$a = Get-Content audit.json -Raw | ConvertFrom-Json'
$certificateInvocation = "& (Join-Path `$auditTools 'New-ShardMapCertificate.ps1')"
$auditReadIndex = $auditStep.IndexOf($auditRead, [StringComparison]::Ordinal)
$certificateIndex = $auditStep.IndexOf($certificateInvocation, [StringComparison]::Ordinal)
Assert-Contract ($certificateIndex -ge 0) `
    'the workflow does not execute the tested certification policy'
Assert-Contract ($auditReadIndex -ge 0 -and $certificateIndex -gt $auditReadIndex) `
    'the workflow invokes certification before loading the audited result'
Assert-Contract ($auditStep -match "(?m)^          $([Regex]::Escape($certificateInvocation))") `
    'the certificate invocation is nested under a conditional instead of running at audit scope'
if ($auditReadIndex -ge 0 -and $certificateIndex -gt $auditReadIndex) {
    $beforeCertificate = $auditStep.Substring(
        $auditReadIndex + $auditRead.Length,
        $certificateIndex - ($auditReadIndex + $auditRead.Length))
    Assert-Contract (-not ($beforeCertificate -match '(?m)^\s*(?:if|elseif|switch|foreach|while|do|return|exit)\b')) `
        'control flow between audit loading and certification can bypass certificate generation'
    Assert-Contract (-not $beforeCertificate.Contains('$a.Failed')) `
        'certificate generation is gated on the observed shard failure count'
}
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
