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
& (Join-Path $PSScriptRoot 'Import-PullRequestShardArtifacts.ps1') -SelfTest
if ($LASTEXITCODE -ne 0) { throw 'pull-request shard artifact import self-test failed' }

Write-Host 'Validation reuse mode proof passed (typed partial/complete scopes, fail-closed artifacts/tree checks, delta decisions and shard imports).'
exit 0
