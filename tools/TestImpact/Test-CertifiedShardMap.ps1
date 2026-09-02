<#
.SYNOPSIS
    Validates that a shard map carries a failure-bearing, zero-miss audit certificate from its run.
#>
[CmdletBinding(DefaultParameterSetName = 'Validate')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Validate')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'Validate')] [string] $CertificateFile,
    [Parameter(Mandatory, ParameterSetName = 'Validate')] [long] $CertificationRunId,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-RequiredInteger {
    param([object] $Value, [string] $Name, [long] $Minimum = 0)
    $parsed = 0L
    if ($null -eq $Value -or -not [long]::TryParse(
        [string] $Value,
        [Globalization.NumberStyles]::Integer,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref] $parsed)) {
        throw "$Name must be an integer"
    }
    if ($parsed -lt $Minimum) { throw "$Name must be at least $Minimum" }
    return $parsed
}

function Assert-CertifiedShardMap {
    param([object] $Map, [object] $Certificate, [long] $ExpectedCertificationRunId)

    foreach ($name in 'schemaVersion', 'sha') {
        if (-not $Map.PSObject.Properties[$name]) { throw "map is missing $name" }
    }
    foreach ($name in 'schemaVersion', 'candidateMapRunId', 'candidateMapSha',
        'auditSourceRunId', 'auditSourceSha', 'certificationRunId', 'escalated',
        'wouldRun', 'wouldSkip', 'failedShards', 'missCount') {
        if (-not $Certificate.PSObject.Properties[$name]) { throw "certificate is missing $name" }
    }

    $schema = ConvertTo-RequiredInteger -Value $Certificate.schemaVersion -Name 'certificate schemaVersion' -Minimum 1
    if ($schema -ne 1) { throw "unsupported certification schema $schema" }
    $actualRun = ConvertTo-RequiredInteger -Value $Certificate.certificationRunId -Name 'certificationRunId' -Minimum 1
    if ($actualRun -ne $ExpectedCertificationRunId) {
        throw "certificate belongs to run $actualRun, not artifact run $ExpectedCertificationRunId"
    }
    [void] (ConvertTo-RequiredInteger -Value $Certificate.candidateMapRunId -Name 'candidateMapRunId' -Minimum 1)
    [void] (ConvertTo-RequiredInteger -Value $Certificate.auditSourceRunId -Name 'auditSourceRunId' -Minimum 1)
    $misses = ConvertTo-RequiredInteger -Value $Certificate.missCount -Name 'missCount'
    if ($misses -ne 0) { throw "certificate records $misses selection miss(es)" }
    if ($Certificate.escalated -isnot [bool]) { throw 'escalated must be a JSON boolean' }
    if ([bool] $Certificate.escalated) { throw 'certificate records an escalated plan, not reduction' }
    [void] (ConvertTo-RequiredInteger -Value $Certificate.wouldRun -Name 'wouldRun' -Minimum 1)
    [void] (ConvertTo-RequiredInteger -Value $Certificate.wouldSkip -Name 'wouldSkip' -Minimum 1)
    [void] (ConvertTo-RequiredInteger -Value $Certificate.failedShards -Name 'failedShards' -Minimum 1)

    $mapSha = [string] $Map.sha
    $candidateSha = [string] $Certificate.candidateMapSha
    if ($mapSha -cnotmatch '^[0-9a-f]{40}$') { throw 'map sha is not a full lowercase commit id' }
    if ($candidateSha -cne $mapSha) { throw 'certificate and map identify different source commits' }
    if ([string] $Certificate.auditSourceSha -cnotmatch '^[0-9a-f]{40}$') {
        throw 'auditSourceSha is not a full lowercase commit id'
    }
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-Rejection {
        param([scriptblock] $Action, [string] $What, [string] $ExpectedPattern)
        $message = $null
        try { & $Action } catch { $message = [string] $_.Exception.Message }
        if ($null -eq $message) {
            [void] $failures.Add($What)
        }
        elseif ($message -notmatch $ExpectedPattern) {
            [void] $failures.Add("$What (rejected for the wrong reason: $message)")
        }
    }

    $sha = '0123456789abcdef0123456789abcdef01234567'
    $map = [pscustomobject]@{ schemaVersion = 1; sha = $sha }
    $certificate = [pscustomobject]@{
        schemaVersion = 1
        candidateMapRunId = 10
        candidateMapSha = $sha
        auditSourceRunId = 11
        auditSourceSha = '89abcdef0123456789abcdef0123456789abcdef'
        certificationRunId = 12
        escalated = $false
        wouldRun = 2
        wouldSkip = 1
        failedShards = 1
        missCount = 0
    }
    try {
        Assert-CertifiedShardMap -Map $map -Certificate $certificate -ExpectedCertificationRunId 12
    }
    catch {
        [void] $failures.Add("valid certificate rejected: $($_.Exception.Message)")
    }

    $bad = $certificate.PSObject.Copy(); $bad.missCount = 1
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a miss-bearing certificate was accepted' 'selection miss'
    $bad = $certificate.PSObject.Copy(); $bad.certificationRunId = 99
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate from another run was accepted' 'certificate belongs to run'
    $bad = $certificate.PSObject.Copy(); $bad.candidateMapSha = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate for another map was accepted' 'different source commits'
    $bad = $certificate.PSObject.Copy(); $bad.escalated = $true
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'an escalated plan was accepted as reduction proof' 'escalated plan'
    $bad = $certificate.PSObject.Copy(); $bad.escalated = 'false'
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a stringly typed escalation flag was accepted' 'JSON boolean'
    $bad = $certificate.PSObject.Copy(); $bad.wouldSkip = 0
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a plan that skipped nothing was accepted as reduction proof' 'wouldSkip must be at least 1'
    $bad = $certificate.PSObject.Copy(); $bad.failedShards = 0
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate without a failure opportunity was accepted' 'failedShards must be at least 1'
    $badMap = $map.PSObject.Copy(); $badMap.sha = $sha.ToUpperInvariant()
    $bad = $certificate.PSObject.Copy(); $bad.candidateMapSha = $badMap.sha
    Assert-Rejection { Assert-CertifiedShardMap $badMap $bad 12 } `
        'an uppercase map commit ID was accepted' 'map sha is not a full lowercase commit id'
    $bad = $certificate.PSObject.Copy(); $bad.auditSourceSha = $bad.auditSourceSha.ToUpperInvariant()
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'an uppercase audit source commit ID was accepted' 'auditSourceSha is not a full lowercase commit id'
    $bad = $certificate.PSObject.Copy(); $bad.PSObject.Properties.Remove('auditSourceRunId')
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'an incomplete certificate was accepted' 'certificate is missing auditSourceRunId'

    if ($failures.Count -gt 0) {
        Write-Host 'Certified shard-map self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'Certified shard-map self-test passed.'
    exit 0
}

$mapObject = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
$certificateObject = Get-Content -LiteralPath $CertificateFile -Raw | ConvertFrom-Json
Assert-CertifiedShardMap $mapObject $certificateObject $CertificationRunId
Write-Host "certified map provenance is valid for run $CertificationRunId"
exit 0
