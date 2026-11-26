[CmdletBinding()]
param([Parameter(Mandatory)] [string] $ShardJson, [Parameter(Mandatory)] [string] $SourceArtifactId,
    [Parameter(Mandatory)] [string] $SourceArtifactDigest, [Parameter(Mandatory)] [string] $TestOutcome,
    [ValidateSet('primary', 'alternate')] [string] $Attempt = 'primary')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
$checkedOutSha = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $checkedOutSha -cne $env:GITHUB_SHA) { throw 'Attempt checkout does not match the tested commit.' }
$shard = $ShardJson | ConvertFrom-Json
$logical = @(Get-FailoverLogicalShards $shard)
for ($index = 0; $index -lt $logical.Count; $index++) {
    $item = $logical[$index]
    $directory = if ($logical.Count -gt 1 -or @(Get-FailoverProperty $shard 'segments' @()).Count -gt 0) { "TestResults/slot-$index" } else { 'TestResults' }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $invocations = @(Get-ChildItem -LiteralPath $directory -Recurse -File -Filter coverage-invocation.json)
    if ($TestOutcome -eq 'success' -and $invocations.Count -ne 1) { throw 'Missing or duplicate actual coverage invocation evidence.' }
    $collectCoverage = if ($invocations.Count -eq 1) { (Get-Content $invocations[0].FullName -Raw | ConvertFrom-Json).collectCoverage } else { $false }
    if ($collectCoverage -isnot [bool] -or ($TestOutcome -eq 'success' -and
        $collectCoverage -ne (Get-FailoverExpectedCoverage $item (Get-FailoverCoveragePolicy)))) { throw 'Executed instrumentation disagrees with required coverage policy.' }
    [ordered]@{ schemaVersion = 1; shard = [string]$item.name; definitionHash = Get-FailoverDefinitionHash $item
        testedSha = $checkedOutSha; runId = $env:GITHUB_RUN_ID; runAttempt = $env:GITHUB_RUN_ATTEMPT
        sourceArtifactId = $SourceArtifactId; sourceArtifactDigest = $SourceArtifactDigest
        attempt = $Attempt; originalJobId = [string](Get-FailoverProperty $shard 'failoverOriginalJobId' '')
        coveragePolicyHash = Get-FailoverCoveragePolicyHash
        collectCoverage = $collectCoverage
        runnerOs = $env:RUNNER_OS; runnerArch = $env:RUNNER_ARCH; runnerName = $env:RUNNER_NAME
        testOutcome = $TestOutcome } | ConvertTo-Json -Depth 20 | Set-Content "$directory/shard-attempt.json" -Encoding utf8
}
