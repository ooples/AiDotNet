[CmdletBinding()]
param([Parameter(Mandatory)] [string] $MatrixJson, [Parameter(Mandatory)] [string] $PlanPath,
    [Parameter(Mandatory)] [string] $Repository, [Parameter(Mandatory)] [string] $RunId,
    [Parameter(Mandatory)] [string] $RunAttempt, [Parameter(Mandatory)] [string] $TestedSha,
    [Parameter(Mandatory)] [string] $SourceArtifactId, [Parameter(Mandatory)] [string] $SourceArtifactDigest,
    [Parameter(Mandatory)] [bool] $TrustedSameRepository, [string] $SelfHostedRunnerToken = '',
    [string] $OutputDirectory = 'runner-failover-proof')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
$matrix = @($MatrixJson | ConvertFrom-Json)
$plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
$policyHash = Get-FailoverCoveragePolicyHash
foreach ($binding in @{ repository = $Repository; runId = $RunId; runAttempt = $RunAttempt; testedSha = $TestedSha
        sourceArtifactId = $SourceArtifactId; sourceArtifactDigest = $SourceArtifactDigest; coveragePolicyHash = $policyHash }.GetEnumerator()) {
    if ([string](Get-FailoverProperty $plan $binding.Key '') -cne [string]$binding.Value) { throw "Failover plan '$($binding.Key)' binding mismatch." }
}
$run = Invoke-FailoverApi "repos/$Repository/actions/runs/$RunId"
if ([string]$run.run_attempt -cne $RunAttempt -or $run.conclusion -eq 'cancelled') { throw 'Run was cancelled or its attempt changed.' }
$source = Invoke-FailoverApi "repos/$Repository/actions/artifacts/$SourceArtifactId"
Assert-FailoverSourceArtifact $source $SourceArtifactId $SourceArtifactDigest $TestedSha $RunId
$jobs = @(Invoke-FailoverApi "repos/$Repository/actions/runs/$RunId/attempts/$RunAttempt/jobs" -Paginate)
$artifacts = @(Invoke-FailoverApi "repos/$Repository/actions/runs/$RunId/artifacts" -Paginate)
[object[]]$runners = if ($TrustedSameRepository -and $SelfHostedRunnerToken) {
    @(Invoke-FailoverApi "repos/$Repository/actions/runners" -Paginate -Token $SelfHostedRunnerToken)
} else { @() }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$downloadRoot = Join-Path $env:RUNNER_TEMP "shard-reconciliation-$RunId-$RunAttempt"
New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
$downloaded = @{}; $downloadIndex = 0
function Receive-FailoverEvidence {
    param([string] $Name)
    $found = @($artifacts | Where-Object { $_.name -ceq $Name -and $_.expired -eq $false })
    if ($found.Count -ne 1) { throw "Missing/duplicate artifact '$Name'." }
    $artifact = $found[0]
    if ([string]$artifact.digest -notmatch '^sha256:[a-f0-9]{64}$' -or [string]$artifact.workflow_run.id -cne $RunId) { throw "Artifact '$Name' has no valid immutable digest/run binding." }
    if (-not $downloaded.ContainsKey([string]$artifact.id)) {
        $directory = Join-Path $downloadRoot ([string]$artifact.id)
        # This receiver fails (not warns) on archive digest mismatch before extraction.
        & "$PSScriptRoot/Receive-RequiredArtifact.ps1" -Repository $Repository -ArtifactId $artifact.id `
            -ExpectedDigest $artifact.digest -Destination $directory -RunId $RunId -JobIndex $script:downloadIndex
        if ($LASTEXITCODE -ne 0) { throw "Artifact '$Name' failed immutable archive verification." }
        $script:downloadIndex++
        $downloaded[[string]$artifact.id] = $directory
    }
    return [pscustomobject]@{ artifact = $artifact; directory = $downloaded[[string]$artifact.id] }
}
$receipts = [Collections.Generic.List[object]]::new()
$errors = [Collections.Generic.List[string]]::new()
$logicalNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$physicalNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$resolvedJobs = [Collections.Generic.List[object]]::new()
foreach ($physical in $matrix) {
    try {
        if (-not $physicalNames.Add([string]$physical.name)) { throw 'Duplicate physical shard definition.' }
        $primaryName = "Tests ($($physical.framework)) - $($physical.name)"
        $primaryJobs = @($jobs | Where-Object { $_.name -ceq $primaryName })
        if ($primaryJobs.Count -ne 1) { throw "Missing/duplicate primary '$primaryName'." }
        $primary = $primaryJobs[0]
        $entries = @($plan.attempts | Where-Object { $_.shard.name -ceq $physical.name })
        if ($entries.Count -ne 1 -or (Get-FailoverDefinitionHash $entries[0].shard) -cne (Get-FailoverDefinitionHash $physical) -or
            [string]$entries[0].primaryJob.id -cne [string]$primary.id) { throw 'Plan does not identify the immutable original shard/job.' }
        $alternateJobs = @($jobs | Where-Object { $_.name -ceq "Alternate attempt ($($physical.framework)) - $($physical.name)" })
        if ($alternateJobs.Count -gt 1) { throw 'More than one alternate attempt exists.' }
        $annotations = if ($primary.conclusion -eq 'failure') { @(Get-FailoverAnnotations $primary $Repository ([string]$run.head_sha)) } else { @() }
        $attempt = 'primary'; $alternate = $null
        if (-not (Test-FailoverSuccessfulJob $primary)) {
            if (-not (Test-FailoverInfrastructureFailure $primary $annotations)) { throw 'Primary test/compiler/unknown failure cannot be recovered.' }
            $planned = @($plan.alternate | Where-Object { $_.name -ceq $physical.name })
            if ($planned.Count -ne 1 -or $alternateJobs.Count -ne 1 -or -not $entries[0].recoveryAuthorized) { throw 'No single authorized alternate attempt exists.' }
            # Re-establish capabilities/trust independently; busy state may have changed since planning.
            $pool = Get-FailoverJobPool $primary $runners
            $alternate = $alternateJobs[0]
            $otherPool = Get-FailoverJobPool $alternate $runners
            if ($pool.pool -eq $otherPool.pool -or $pool.os -cne $otherPool.os -or $pool.arch -cne $otherPool.arch -or
                ($otherPool.pool -eq 'self-hosted' -and -not $TrustedSameRepository) -or
                [string]$planned[0].failoverOriginalJobId -cne [string]$primary.id -or
                (Get-FailoverDefinitionHash $planned[0]) -cne (Get-FailoverDefinitionHash $physical)) { throw 'Alternate pool/trust/OS/architecture/definition binding mismatch.' }
            if ($otherPool.pool -eq 'self-hosted' -and ((Get-FailoverProperty $physical 'heavy' $false) -or
                (Get-FailoverProperty $physical 'timeout' 0) -or @(Get-FailoverProperty $physical 'segments' @()).Count -gt 0)) { throw 'Alternate CPU pool cannot satisfy the shard capabilities.' }
            $attempt = 'alternate'
        } elseif ($alternateJobs.Count -ne 0) { throw 'Successful primary was scheduled for overwrite.' }
        $attemptName = "shard-attempt-$attempt-$TestedSha-$(Get-FailoverSlug $physical.name)-$RunAttempt"
        $evidence = Receive-FailoverEvidence $attemptName
        $allAttemptReceipts = @(Get-ChildItem -LiteralPath $evidence.directory -Recurse -File -Filter shard-attempt.json)
        $logical = @(Get-FailoverLogicalShards $physical)
        if ($allAttemptReceipts.Count -ne $logical.Count) { throw 'Packed/unpacked attempt has missing or duplicate logical receipts.' }
        foreach ($shard in $logical) {
            if (-not $logicalNames.Add([string]$shard.name)) { throw "Duplicate logical shard '$($shard.name)'." }
            $foundReceipt = @($allAttemptReceipts | Where-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).shard -ceq $shard.name })
            if ($foundReceipt.Count -ne 1) { throw "Missing/duplicate logical receipt '$($shard.name)'." }
            $attemptReceipt = Get-Content $foundReceipt[0].FullName -Raw | ConvertFrom-Json
            $canonical = Receive-FailoverEvidence "test-results-$TestedSha-$(Get-FailoverSlug $shard.name)"
            $canonicalReceipts = @(Get-ChildItem $canonical.directory -Recurse -File -Filter shard-attempt.json)
            if ($canonicalReceipts.Count -ne 1) { throw 'Missing/duplicate canonical receipt.' }
            $receipt = Get-Content $canonicalReceipts[0].FullName -Raw | ConvertFrom-Json
            $coverage = Receive-FailoverEvidence "coverage-$TestedSha-$(Get-FailoverSlug $shard.name)"
            $resolved = Assert-FailoverResolvedShard -Shard $shard -PrimaryJob $primary -PrimaryAnnotations $annotations `
                -AlternateJob $alternate -Receipt $receipt -AttemptReceipt $attemptReceipt -CanonicalArtifact $canonical.artifact `
                -AttemptArtifact $evidence.artifact -CanonicalDirectory $canonical.directory -CoverageArtifact $coverage.artifact -CoverageDirectory $coverage.directory `
                -AttemptDirectory $foundReceipt[0].DirectoryName -ExpectedAttemptArtifactName $attemptName `
                -TestedSha $TestedSha -RunId $RunId -RunAttempt $RunAttempt -SourceArtifactId $SourceArtifactId `
                -SourceArtifactDigest $SourceArtifactDigest -CoveragePolicyHash $policyHash
            $receipts.Add($resolved)
            $chosen = if ($alternate) { $alternate.PSObject.Copy() } else { $primary.PSObject.Copy() }
            $chosen.name = "Tests ($($shard.framework)) - $($shard.name)"
            $chosen | Add-Member failoverProvenance $resolved -Force
            $resolvedJobs.Add($chosen)
        }
    } catch { $errors.Add("$($physical.name): $($_.Exception.Message)"); Write-Warning $errors[-1] }
}
try { Assert-FailoverLogicalInventory -RequiredShards @($matrix | ForEach-Object { Get-FailoverLogicalShards $_ }) -ResolvedEvidence @($receipts) }
catch { $errors.Add($_.Exception.Message) }
$proof = [ordered]@{ schemaVersion = 1; complete = ($errors.Count -eq 0 -and $physicalNames.Count -eq $matrix.Count)
    requiresValidation = $true; requiresTests = $true
    testedSha = $TestedSha; runId = $RunId; runAttempt = $RunAttempt; sourceArtifactId = $SourceArtifactId
    sourceArtifactDigest = $SourceArtifactDigest; coveragePolicyHash = $policyHash
    requiredShards = @($matrix | ForEach-Object { Get-FailoverLogicalShards $_ }); shards = @($receipts); errors = @($errors) }
$proof | ConvertTo-Json -Depth 100 | Set-Content "$OutputDirectory/proof.json" -Encoding utf8
ConvertTo-Json -InputObject @($jobs) -Depth 100 | Set-Content "$OutputDirectory/original-jobs.json" -Encoding utf8
ConvertTo-Json -InputObject @($resolvedJobs) -Depth 100 | Set-Content "$OutputDirectory/resolved-jobs.json" -Encoding utf8
if (-not $proof.complete -or $matrix.Count -eq 0) { throw "Logical shard evidence gate failed: $($errors.Count) rejected shard job(s)." }
Write-Host "Verified $($receipts.Count) required logical shard receipt(s); original attempts remain in original-jobs.json."
