[CmdletBinding(DefaultParameterSetName = 'Json')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Json')] [string] $MatrixJson,
    [Parameter(Mandatory, ParameterSetName = 'Path')] [string] $MatrixJsonPath,
    [Parameter(Mandatory)] [string] $Repository,
    [Parameter(Mandatory)] [string] $RunId,
    [Parameter(Mandatory)] [string] $RunAttempt,
    [Parameter(Mandatory)] [string] $TestedSha,
    [Parameter(Mandatory)] [string] $SourceArtifactId,
    [Parameter(Mandatory)] [string] $SourceArtifactDigest,
    [Parameter(Mandatory)] [bool] $TrustedSameRepository,
    [string] $SelfHostedRunnerToken = '',
    [string] $OutputDirectory = 'runner-failover-plan'
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
# CI passes the shard matrix as a file: it can outgrow Linux's 128 KiB limit on one environment string.
if ($PSCmdlet.ParameterSetName -ceq 'Path') { $MatrixJson = Get-Content -LiteralPath $MatrixJsonPath -Raw }
if ([string]::IsNullOrWhiteSpace($MatrixJson)) { throw 'The shard matrix is empty.' }
$matrix = @($MatrixJson | ConvertFrom-Json)
if ($matrix.Count -eq 0) { throw 'Cannot plan failover for an empty shard inventory.' }
$run = Invoke-FailoverApi "repos/$Repository/actions/runs/$RunId"
if ([string]$run.run_attempt -cne $RunAttempt -or $run.conclusion -eq 'cancelled') { throw 'Run attempt changed or was cancelled.' }
$source = Invoke-FailoverApi "repos/$Repository/actions/artifacts/$SourceArtifactId"
Assert-FailoverSourceArtifact $source $SourceArtifactId $SourceArtifactDigest $TestedSha $RunId
$jobs = @(Invoke-FailoverApi "repos/$Repository/actions/runs/$RunId/attempts/$RunAttempt/jobs" -Paginate)
$runners = @(if ($TrustedSameRepository -and $SelfHostedRunnerToken) {
    Invoke-FailoverApi "repos/$Repository/actions/runners" -Paginate -Token $SelfHostedRunnerToken
})
$alternate = [Collections.Generic.List[object]]::new()
$attempts = [Collections.Generic.List[object]]::new()
$names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$refused = [Collections.Generic.List[object]]::new()
$repeated = [Collections.Generic.List[object]]::new()
$selfHostedAuthorized = 0
$previousLoaded = $false; $previousRun = $null; $previousJobs = @()
foreach ($shard in $matrix) {
    if (-not $names.Add([string]$shard.name)) { throw "Duplicate matrix shard '$($shard.name)'." }
    $name = "Tests ($($shard.framework)) - $($shard.name)"
    $matches = @($jobs | Where-Object { $_.name -ceq $name })
    if ($matches.Count -ne 1 -or $matches[0].status -ne 'completed') { throw "Missing, duplicate or unfinished primary job '$name'." }
    $job = $matches[0]
    $annotations = @(if (-not (Test-FailoverSuccessfulJob $job) -and (Get-FailoverProperty $job 'check_run_url' '')) {
        Get-FailoverAnnotations $job $Repository ([string]$run.head_sha)
    })
    $entry = [ordered]@{ shard = $shard; primaryJob = $job; annotations = $annotations; recoveryAuthorized = $false; reason = 'Primary must independently prove success.' }
    if (-not (Test-FailoverSuccessfulJob $job) -and (Test-FailoverInfrastructureFailure $job $annotations)) {
        try {
            # A hosted shard that lost its runner on the previous run as well is not rerun: see
            # Test-FailoverLostOnPreviousRun. Self-hosted primaries are exempt - that pool is opportunistic and
            # a machine that is switched off two nights running says nothing about the shard. If the previous
            # run cannot be read the shard is not rerun either; the history is part of the evidence.
            if (-not (Test-IsSelfHostedFailoverJob $job)) {
                if (-not $previousLoaded) {
                    try {
                        $previousRun = Get-FailoverPreviousRun $run $Repository
                        $previousJobs = @(if ($null -ne $previousRun) {
                            Invoke-FailoverApi "repos/$Repository/actions/runs/$($previousRun.id)/jobs?filter=all" -Paginate
                        })
                    } catch { throw "The previous run could not be read, so the shard is not rerun: $($_.Exception.Message)" }
                    $previousLoaded = $true
                }
                if (Test-FailoverLostOnPreviousRun -JobName $name -PreviousRun $previousRun -PreviousJobs $previousJobs -Repository $Repository) {
                    $repeated.Add([pscustomobject]@{ shard = [string]$shard.name; previousRunId = [string]$previousRun.id })
                    throw "Lost its runner on the previous run ($($previousRun.id)) as well; not rerun. A shard that loses its runner twice running is exhausting it."
                }
            }
            $copy = New-FailoverAlternateShard $shard $job $runners $TrustedSameRepository -SelfHostedAuthorized $selfHostedAuthorized
            if (@($copy.runner) -contains 'self-hosted') { $selfHostedAuthorized++ }
            $alternate.Add($copy)
            $entry.recoveryAuthorized = $true
            $entry.reason = "Recoverable runner failure ($($job.conclusion)); one alternate $(@($copy.runner) -join '/') attempt."
        } catch { $entry.reason = $_.Exception.Message; $refused.Add([pscustomobject]@{ shard = [string]$shard.name; reason = $entry.reason }) }
    }
    $attempts.Add([pscustomobject]$entry)
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$plan = [ordered]@{ schemaVersion = 1; repository = $Repository; runId = $RunId; runAttempt = $RunAttempt; testedSha = $TestedSha
    sourceArtifactId = $SourceArtifactId; sourceArtifactDigest = $SourceArtifactDigest; coveragePolicyHash = Get-FailoverCoveragePolicyHash
    attempts = @($attempts); alternate = @($alternate); repeatedRunnerLoss = @($repeated) }
$plan | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath "$OutputDirectory/plan.json" -Encoding utf8
if ($env:GITHUB_OUTPUT) {
    "has_recovery=$($alternate.Count -gt 0)".ToLowerInvariant() | Add-Content -LiteralPath $env:GITHUB_OUTPUT
    "matrix=$(ConvertTo-Json -InputObject @($alternate) -Depth 100 -Compress)" | Add-Content -LiteralPath $env:GITHUB_OUTPUT
}
foreach ($attempt in $attempts) { Write-Host "$($attempt.shard.name): $($attempt.reason)" }
# A SHARD THAT LOST ITS RUNNER MUST STAY VISIBLE. A rerun that passes leaves every check green, and a runner
# that the shard itself starved of memory looks exactly like one GitHub lost. So each one is annotated on the
# run and listed in the summary, whether it is rerun or refused.
foreach ($copy in $alternate) {
    if (@($copy.runner) -contains 'self-hosted') {
        Write-Host "::warning title=Shard rerun after runner loss::'$($copy.name)' lost its hosted runner (job $($copy.failoverOriginalJobId)) and is rerun once on a self-hosted runner. If it is lost again on the next run it will not be rerun."
    } else {
        Write-Host "::warning title=Shard rerun after runner loss::'$($copy.name)' did not finish on its self-hosted runner (job $($copy.failoverOriginalJobId)) and is rerun once on ubuntu-latest."
    }
}
foreach ($item in $refused) {
    Write-Host "::error title=Lost shard not rerun::'$($item.shard)': $($item.reason)"
}
if ($env:GITHUB_STEP_SUMMARY -and ($alternate.Count -gt 0 -or $refused.Count -gt 0)) {
    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add('### Shards that lost their runner')
    $lines.Add('')
    $lines.Add('| Shard | Original job | Decision |')
    $lines.Add('| --- | --- | --- |')
    foreach ($attempt in @($attempts | Where-Object { $_.recoveryAuthorized -or @($refused | Where-Object shard -ceq $_.shard.name).Count -gt 0 })) {
        $decision = if ($attempt.recoveryAuthorized) { 'rerun once' } else { 'NOT rerun' }
        $lines.Add("| $($attempt.shard.name) | $($attempt.primaryJob.id) | ${decision}: $(([string]$attempt.reason) -replace '\|', '/') |")
    }
    $lines.Add('')
    $lines.Add("Self-hosted alternates are capped at one per idle runner; $selfHostedAuthorized authorised in this run.")
    $lines | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Encoding utf8
}
