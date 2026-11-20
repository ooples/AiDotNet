[CmdletBinding()]
param([Parameter(Mandatory)] [string] $MatrixJson,
    [Parameter(Mandatory)] [string] $RequiresValidation, [Parameter(Mandatory)] [string] $RequiresTests,
    [string] $OutputDirectory = 'runner-failover-proof')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
$matrix = ConvertFrom-Json -InputObject $MatrixJson -NoEnumerate
if ($RequiresValidation -cnotin @('true', 'false') -or $RequiresTests -cne 'false' -or
    $matrix -isnot [array] -or $matrix.Count -ne 0) { throw 'Only an explicitly empty, non-runtime selection may emit empty shard proof.' }
$sha = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sha -cne $env:GITHUB_SHA -or $sha -notmatch '^[a-f0-9]{40}$' -or
    $env:GITHUB_RUN_ID -notmatch '^[1-9][0-9]*$' -or $env:GITHUB_RUN_ATTEMPT -notmatch '^[1-9][0-9]*$') { throw 'Empty proof run/commit binding is invalid.' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
[ordered]@{ schemaVersion = 1; complete = $true; requiresValidation = [bool]::Parse($RequiresValidation); requiresTests = $false
    testedSha = $sha; runId = $env:GITHUB_RUN_ID; runAttempt = $env:GITHUB_RUN_ATTEMPT
    sourceArtifactId = ''; sourceArtifactDigest = ''; coveragePolicyHash = Get-FailoverCoveragePolicyHash
    requiredShards = @(); shards = @(); errors = @() } |
    ConvertTo-Json -Depth 20 | Set-Content "$OutputDirectory/proof.json" -Encoding utf8
'[]' | Set-Content "$OutputDirectory/original-jobs.json" -Encoding utf8
'[]' | Set-Content "$OutputDirectory/resolved-jobs.json" -Encoding utf8
