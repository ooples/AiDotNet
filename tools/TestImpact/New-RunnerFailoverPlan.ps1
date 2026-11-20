[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $MatrixJson,
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
foreach ($shard in $matrix) {
    if (-not $names.Add([string]$shard.name)) { throw "Duplicate matrix shard '$($shard.name)'." }
    $name = "Tests ($($shard.framework)) - $($shard.name)"
    $matches = @($jobs | Where-Object { $_.name -ceq $name })
    if ($matches.Count -ne 1 -or $matches[0].status -ne 'completed') { throw "Missing, duplicate or unfinished primary job '$name'." }
    $job = $matches[0]
    $annotations = @(if ($job.conclusion -eq 'failure') { Get-FailoverAnnotations $job $Repository ([string]$run.head_sha) })
    $entry = [ordered]@{ shard = $shard; primaryJob = $job; annotations = $annotations; recoveryAuthorized = $false; reason = 'Primary must independently prove success.' }
    if (Test-FailoverInfrastructureFailure $job $annotations) {
        try {
            $copy = New-FailoverAlternateShard $shard $job $runners $TrustedSameRepository
            $alternate.Add($copy)
            $entry.recoveryAuthorized = $true
            $entry.reason = "Explicit terminal runner infrastructure failure; one alternate $($copy.runner) attempt."
        } catch { $entry.reason = $_.Exception.Message }
    }
    $attempts.Add([pscustomobject]$entry)
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$plan = [ordered]@{ schemaVersion = 1; repository = $Repository; runId = $RunId; runAttempt = $RunAttempt; testedSha = $TestedSha
    sourceArtifactId = $SourceArtifactId; sourceArtifactDigest = $SourceArtifactDigest; coveragePolicyHash = Get-FailoverCoveragePolicyHash
    attempts = @($attempts); alternate = @($alternate) }
$plan | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath "$OutputDirectory/plan.json" -Encoding utf8
if ($env:GITHUB_OUTPUT) {
    "has_recovery=$($alternate.Count -gt 0)".ToLowerInvariant() | Add-Content -LiteralPath $env:GITHUB_OUTPUT
    "matrix=$(ConvertTo-Json -InputObject @($alternate) -Depth 100 -Compress)" | Add-Content -LiteralPath $env:GITHUB_OUTPUT
}
foreach ($attempt in $attempts) { Write-Host "$($attempt.shard.name): $($attempt.reason)" }
