<#
.SYNOPSIS
    Executes the exact typed certificate and reuse policies used by CI.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$certificateWriter = Join-Path $PSScriptRoot 'New-CiValidationCertificate.ps1'
$resolver = Join-Path $PSScriptRoot 'Resolve-CiValidationReuse.ps1'

& $certificateWriter -SelfTest
if ($LASTEXITCODE -ne 0) { throw 'CI validation certificate self-test failed' }
& $resolver -SelfTest
if ($LASTEXITCODE -ne 0) { throw 'CI validation reuse self-test failed' }
& (Join-Path $PSScriptRoot 'Resume-DeferredCi.ps1') -SelfTest
if ($LASTEXITCODE -ne 0) { throw 'Deferred CI reconciliation self-test failed' }
& (Join-Path $PSScriptRoot 'Import-PullRequestShardArtifacts.ps1') -SelfTest
if ($LASTEXITCODE -ne 0) { throw 'pull-request shard artifact import self-test failed' }

# Exercise the actual script parameter binder, not only extracted policy functions: a missing
# certified map makes GitHub substitute empty strings for both optional delta inputs.
$bindingOutput = Join-Path ([System.IO.Path]::GetTempPath()) "reuse-binding-$([guid]::NewGuid().ToString('N')).txt"
try {
    foreach ($inputs in @(
        @{ MapFile = ''; ShardManifestFile = '' },
        @{ MapFile = ''; ShardManifestFile = 'absent-manifest.json' },
        @{ MapFile = 'absent-map.json'; ShardManifestFile = '' },
        @{}
    )) {
        & $resolver -Repository 'fixture/repo' -CommitSha ('a' * 40) -EventName pull_request `
            -ExpectedBaseBranch master -GitHubOutput $bindingOutput -WaitMinutes 0 @inputs
        if ($LASTEXITCODE -ne 0) { throw 'Missing-map invocation failed before emitting a decision' }
        $decision = @(Get-Content -LiteralPath $bindingOutput)
        foreach ($required in @('reuse=false', 'execute_validation=true', 'execute_quality=true')) {
            if ($decision -cnotcontains $required) { throw "Missing-map invocation did not emit $required" }
        }
        Remove-Item -LiteralPath $bindingOutput
    }
}
finally {
    if (Test-Path -LiteralPath $bindingOutput) { Remove-Item -LiteralPath $bindingOutput }
}
Write-Host 'Missing-map script invocation: 4 cases passed.'
Write-Host 'Validation reuse mode proof passed (typed partial/complete scopes, fail-closed artifacts/tree checks, delta decisions and shard imports).'
exit 0
