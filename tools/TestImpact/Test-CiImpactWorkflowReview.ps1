<#
.SYNOPSIS
    Proves workflow contract checks reject unsafe wiring, not just the checked-in happy path.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'ReviewFixtureCleanup.ps1')

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workflowPath = Join-Path $repositoryRoot '.github/workflows/sonarcloud.yml'
$contractPath = Join-Path $PSScriptRoot 'Test-CiImpactWorkflow.ps1'
$workflow = Get-Content -LiteralPath $workflowPath -Raw
$mapHeadLine = '(?m)^[ \t]+-PullRequestHeadSha \$env:PR_HEAD_SHA `[ \t]*\r?$'
if ([regex]::Matches($workflow, $mapHeadLine).Count -ne 1) {
    throw 'The fixture must identify exactly one map-backed PR-head argument, excluding the classifier.'
}
$cases = @(
    [pscustomobject]@{
        Name = 'map-head-removed'
        Reason = 'map-backed selector.*pull.request'
        Content = [regex]::Replace($workflow, $mapHeadLine, '')
    },
    [pscustomobject]@{
        Name = 'map-head-wrong-variable'
        Reason = 'map-backed selector.*pull.request'
        Content = [regex]::Replace($workflow, $mapHeadLine, '                    -PullRequestHeadSha $env:WRONG_HEAD_SHA `')
    },
    [pscustomobject]@{
        Name = 'map-head-only-in-comment'
        Reason = 'map-backed selector.*pull.request'
        Content = [regex]::Replace($workflow, $mapHeadLine, '                    # -PullRequestHeadSha $env:PR_HEAD_SHA `')
    },
    [pscustomobject]@{
        Name = 'delta-map-id-renamed'
        Reason = 'delta.map step.*identity'
        Content = $workflow.Replace('        id: delta-map', '        id: wrong-delta-map')
    },
    [pscustomobject]@{
        Name = 'delta-map-id-only-in-comment'
        Reason = 'delta.map step.*identity'
        Content = $workflow.Replace('        id: delta-map', '        # id: delta-map')
    }
)
$deltaMapPattern = '(?ms)^      - name: Resolve certified shard map for delta reuse\r?\n.*?(?=^      - name:|\z)'
$checkoutPattern = '(?ms)^      - name: Checkout validation reuse policy\r?\n.*?(?=^      - name:|\z)'
$checkout = [regex]::Match($workflow, $checkoutPattern).Value
$credentialInput = '          persist-credentials: false'
if (-not $checkout.Contains($credentialInput)) { throw 'Missing checkout credential input for negative controls.' }
foreach ($mutation in @(
    @{ Name = 'checkout-credentials-comment-decoy'; Value = "          persist-credentials: true`n          # persist-credentials: false" },
    @{ Name = 'checkout-credentials-comment-only'; Value = '          # persist-credentials: false' },
    @{ Name = 'checkout-credentials-duplicate'; Value = "          persist-credentials: false`n          persist-credentials: true" },
    @{ Name = 'checkout-credentials-duplicate-false'; Value = "$credentialInput`n$credentialInput" },
    @{ Name = 'checkout-credentials-outside-with'; Value = "        env:`n$credentialInput" },
    @{ Name = 'checkout-credentials-nested-decoy'; Value = "          unused: |`n            persist-credentials: false" },
    @{ Name = 'checkout-credentials-duplicate-with'; Value = "        with:`n$credentialInput" }
)) {
    $cases += [pscustomobject]@{
        Name = $mutation.Name
        Reason = 'validation-source persists checkout credentials'
        Content = $workflow.Replace($checkout, $checkout.Replace($credentialInput, $mutation.Value))
    }
}
$deltaMapMatch = [regex]::Match($workflow, $deltaMapPattern)
if (-not $deltaMapMatch.Success) { throw 'The fixture must identify the delta-map step.' }
$unboundedMapStep = [regex]::Replace($deltaMapMatch.Value, '(?m)^        timeout-minutes:[^\r\n]*\r?\n', '')
$cases += @(
    [pscustomobject]@{
        Name = 'delta-map-timeout-removed'
        Reason = 'delta.map.*bounded|delta.map.*budget'
        Content = $workflow.Replace($deltaMapMatch.Value, $unboundedMapStep)
    },
    [pscustomobject]@{
        Name = 'delta-map-timeout-consumes-job'
        Reason = 'delta.map.*bounded|delta.map.*budget'
        Content = $workflow.Replace($deltaMapMatch.Value, $unboundedMapStep.Replace(
            '        id: delta-map', "        id: delta-map`n        timeout-minutes: 50"))
    },
    [pscustomobject]@{
        Name = 'resolver-wait-consumes-job'
        Reason = 'delta.map.*budget'
        Content = [regex]::Replace($workflow, '(?m)^            -WaitMinutes [0-9]+[ \t]*\r?$', '            -WaitMinutes 45')
    },
    [pscustomobject]@{
        Name = 'resolver-job-budget-shrunk'
        Reason = 'delta.map.*budget'
        Content = [regex]::Replace($workflow, '(?ms)(^  validation-source:\r?\n.*?^    timeout-minutes:) [0-9]+', '${1} 45')
    }
)

$importGate = "needs.validation-source.outputs.import_run_id != '' && needs.select-shards.outputs.escalated == 'false'"
foreach ($stepName in @('Download pull-request shard results for delta import', 'Import pull-request shard results',
        'Download pull-request diagnostics for delta import', 'Import pull-request diagnostics',
        'Download pull-request coverage for delta import', 'Import pull-request coverage')) {
    $stepPattern = '(?ms)^      - name: ' + [regex]::Escape($stepName) + '\r?\n.*?(?=^      - name:|\z)'
    $stepBlock = [regex]::Match($workflow, $stepPattern).Value
    if (-not $stepBlock.Contains($importGate)) { throw "Missing complete import gate on $stepName" }
    $cases += [pscustomobject]@{
        Name = ($stepName -replace ' ', '-') + '-escalation-guard-removed'
        Reason = 'delta import.*escalation'
        Content = $workflow.Replace($stepBlock, $stepBlock.Replace($importGate, "needs.validation-source.outputs.import_run_id != ''"))
    }
    $cases += [pscustomobject]@{
        Name = ($stepName -replace ' ', '-') + '-guard-disjoined'
        Reason = 'delta import.*escalation'
        Content = $workflow.Replace($stepBlock, $stepBlock.Replace($importGate, $importGate.Replace(' && ', ' || ')))
    }
}
foreach ($pair in @(
        @{ Download = 'Download pull-request shard results for delta import'; Import = 'Import pull-request shard results' },
        @{ Download = 'Download pull-request diagnostics for delta import'; Import = 'Import pull-request diagnostics' },
        @{ Download = 'Download pull-request coverage for delta import'; Import = 'Import pull-request coverage' })) {
    $downloadPattern = '(?ms)^      - name: ' + [regex]::Escape($pair.Download) + '\r?\n.*?(?=^      - name:|\z)'
    $download = [regex]::Match($workflow, $downloadPattern).Value
    $runId = '          run-id: ${{ needs.validation-source.outputs.import_run_id }}'
    if (-not $download.Contains($runId)) { throw 'Missing source-bound download run id.' }
    # Put the right text in the importer's comment, in the SAME job: a job-wide Contains passes.
    $mutated = $workflow.Replace($download, $download.Replace($runId, '          run-id: 1'))
    $mutated = $mutated.Replace('      - name: ' + $pair.Import,
        '      - name: ' + $pair.Import + "`n        # run-id: `${{ needs.validation-source.outputs.import_run_id }}")
    $cases += [pscustomobject]@{
        Name = ($pair.Download -replace ' ', '-') + '-run-id-decoy'
        Reason = 'delta download.*matching PR artifacts'
        Content = $mutated
    }
    foreach ($field in @('uses', 'run-id', 'pattern', 'path')) {
        $fieldPattern = '(?m)^(?<indent> +)' + $field + ': (?<value>[^\r\n]+)\r?$'
        $fieldMatch = [regex]::Match($download, $fieldPattern)
        if (-not $fieldMatch.Success) { throw "Missing download field $field for a comment decoy." }
        $replacement = $fieldMatch.Groups['indent'].Value + $field + ': wrong-value' + "`n" +
            $fieldMatch.Groups['indent'].Value + '# ' + $field + ': ' + $fieldMatch.Groups['value'].Value
        $cases += [pscustomobject]@{
            Name = ($pair.Download -replace ' ', '-') + '-' + $field + '-same-step-comment-decoy'
            Reason = 'delta download.*matching PR artifacts'
            Content = $workflow.Replace($download, $download.Replace($fieldMatch.Value, $replacement))
        }
    }
    $importPattern = '(?ms)^      - name: ' + [regex]::Escape($pair.Import) + '\r?\n.*?(?=^      - name:|\z)'
    $import = [regex]::Match($workflow, $importPattern).Value
    $invocation = '          ./tools/TestImpact/Import-PullRequestShardArtifacts.ps1'
    if (-not $import.Contains($invocation)) { throw 'Missing importer invocation for a comment decoy.' }
    $cases += [pscustomobject]@{
        Name = ($pair.Import -replace ' ', '-') + '-invocation-commented'
        Reason = 'delta importer.*matching download'
        Content = $workflow.Replace($import, $import.Replace($invocation, '          # ./tools/TestImpact/Import-PullRequestShardArtifacts.ps1'))
    }
}
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ('aidotnet-ci-contract-review-' + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
try {
    New-Item -ItemType Directory -Path $fixture | Out-Null
    Push-Location $repositoryRoot
    try {
        $baseline = @(& pwsh -NoProfile -File $contractPath 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "The unmodified workflow must pass first: $($baseline -join [Environment]::NewLine)" }
        foreach ($case in $cases) {
            if ($case.Content -ceq $workflow) { throw "$($case.Name): negative control did not mutate the workflow" }
            if (-not $case.Content.Contains('-PullRequestHeadSha $env:PR_HEAD_SHA -OutFile path-classification.json')) {
                throw 'A map-scoping negative control accidentally changed the classifier.'
            }
            $path = Join-Path $fixture ($case.Name + '.yml')
            Set-Content -LiteralPath $path -Value $case.Content -Encoding utf8
            $output = @(& pwsh -NoProfile -File $contractPath -ValidationWorkflow $path 2>&1)
            if ($LASTEXITCODE -eq 0) {
                [void] $failures.Add("$($case.Name): unsafe workflow wiring passed the contract")
            }
            elseif (($output -join [Environment]::NewLine) -notmatch $case.Reason) {
                [void] $failures.Add("$($case.Name): rejected for an unrelated reason: $($output -join [Environment]::NewLine)")
            }
            else {
                Write-Host "Rejected unsafe workflow: $($case.Name)"
            }
        }
    }
    finally { Pop-Location }
}
finally {
    Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-ci-contract-review-'
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Host $failure }
    exit 1
}
Write-Host "Workflow review controls passed: baseline accepted, $($cases.Count) unsafe mutations rejected."
exit 0
