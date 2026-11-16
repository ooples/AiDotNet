[CmdletBinding()]
param([Parameter(Mandatory)] [string] $ShardJson, [Parameter(Mandatory)] [string] $Repository,
    [Parameter(Mandatory)] [string] $RunId, [Parameter(Mandatory)] [string] $RunAttempt,
    [Parameter(Mandatory)] [string] $TestedSha, [Parameter(Mandatory)] [string] $PlanArtifactId,
    [Parameter(Mandatory)] [string] $PlanArtifactDigest, [Parameter(Mandatory)] [bool] $TrustedSameRepository,
    [string] $SelfHostedRunnerToken = '')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
$shard = $ShardJson | ConvertFrom-Json
if ((Get-FailoverProperty $shard 'failoverAttempt' '') -cne 'alternate') { throw 'Alternate attempt marker is absent.' }
$run = Invoke-FailoverApi "repos/$Repository/actions/runs/$RunId"
if ([string]$run.run_attempt -cne $RunAttempt -or $run.conclusion -eq 'cancelled') { throw 'Run attempt changed or cancelled.' }
$metadata = Invoke-FailoverApi "repos/$Repository/actions/artifacts/$PlanArtifactId"
if ([string]$metadata.name -cne "runner-failover-plan-$TestedSha-$RunAttempt" -or
    [string]$metadata.digest -cne $PlanArtifactDigest -or [string]$metadata.workflow_run.id -cne $RunId -or $metadata.expired) { throw 'Recovery plan artifact binding is invalid.' }
$directory = Join-Path $env:RUNNER_TEMP "alternate-authorization-$PlanArtifactId"
& "$PSScriptRoot/Receive-RequiredArtifact.ps1" -Repository $Repository -ArtifactId $PlanArtifactId -ExpectedDigest $PlanArtifactDigest `
    -Destination $directory -RunId $RunId -JobIndex 0
if ($LASTEXITCODE -ne 0) { throw 'Recovery plan archive digest verification failed.' }
$plan = Get-Content (Join-Path $directory 'plan.json') -Raw | ConvertFrom-Json
if ([string]$plan.testedSha -cne $TestedSha -or [string]$plan.runId -cne $RunId -or [string]$plan.runAttempt -cne $RunAttempt -or
    [string]$plan.repository -cne $Repository -or [string]$plan.coveragePolicyHash -cne (Get-FailoverCoveragePolicyHash)) { throw 'Recovery plan run/SHA/policy is invalid.' }
$planned = @($plan.alternate | Where-Object { $_.name -ceq $shard.name })
if ($planned.Count -ne 1 -or (Get-FailoverDefinitionHash $planned[0]) -cne (Get-FailoverDefinitionHash $shard) -or
    (ConvertTo-Json -InputObject $planned[0].runner -Compress) -cne (ConvertTo-Json -InputObject $shard.runner -Compress) -or
    [string]$planned[0].failoverOriginalJobId -cne [string]$shard.failoverOriginalJobId) { throw 'Alternate shard does not match its unique authorization.' }
$jobs = @(Invoke-FailoverApi "repos/$Repository/actions/runs/$RunId/attempts/$RunAttempt/jobs" -Paginate)
$primary = @($jobs | Where-Object { [string]$_.id -ceq [string]$shard.failoverOriginalJobId })
if ($primary.Count -ne 1 -or $primary[0].name -cne "Tests ($($shard.framework)) - $($shard.name)" -or $primary[0].status -ne 'completed') { throw 'Original attempt job binding is invalid.' }
$annotations = @(Get-FailoverAnnotations $primary[0] $Repository ([string]$run.head_sha))
if (-not (Test-FailoverInfrastructureFailure $primary[0] $annotations)) { throw 'A non-infrastructure or successful primary cannot authorize replacement.' }
$alternates = @($jobs | Where-Object { $_.name -ceq "Alternate attempt ($($shard.framework)) - $($shard.name)" })
if ($alternates.Count -ne 1 -or $alternates[0].runner_name -cne $env:RUNNER_NAME) { throw 'Duplicate or differently assigned alternate attempt.' }
$runners = if ($TrustedSameRepository -and $SelfHostedRunnerToken) { @(Invoke-FailoverApi "repos/$Repository/actions/runners" -Paginate -Token $SelfHostedRunnerToken) } else { @() }
$originalPool = Get-FailoverJobPool $primary[0] $runners
$alternatePool = Get-FailoverJobPool $alternates[0] $runners
if ($originalPool.pool -eq $alternatePool.pool -or $originalPool.os -cne $alternatePool.os -or $originalPool.arch -cne $alternatePool.arch -or
    ($alternatePool.pool -eq 'self-hosted' -and -not $TrustedSameRepository)) { throw 'Alternate pool capability/trust mismatch.' }
Write-Host "Authorized one alternate attempt for original job $($primary[0].id); no functional failures are exempted."
