<#
.SYNOPSIS
    Validates that a shard map carries a complete, zero-miss audit certificate from its run.
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

enum ShardMapCertificationBasis {
    HistoricalReplay
    FreshCoverage
}

function ConvertTo-RequiredInteger {
    param([object] $Value, [string] $Name, [long] $Minimum = 0)
    $isInteger = $Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
    if (-not $isInteger) { throw "$Name must be a JSON integer" }
    $number = [long] $Value
    if ($number -lt $Minimum) { throw "$Name must be at least $Minimum" }
    return $number
}

function Assert-CertifiedShardMap {
    param([object] $Map, [object] $Certificate, [long] $ExpectedCertificationRunId)

    foreach ($name in 'schemaVersion', 'sha', 'knownShards', 'alwaysRun') {
        if (-not $Map.PSObject.Properties[$name]) { throw "map is missing $name" }
    }
    foreach ($name in 'schemaVersion', 'candidateMapRunId', 'candidateMapSha',
        'auditSourceRunId', 'auditSourceSha', 'certificationRunId', 'escalated',
        'auditedShards', 'wouldRun', 'wouldSkip', 'failedShards', 'missCount') {
        if (-not $Certificate.PSObject.Properties[$name]) { throw "certificate is missing $name" }
    }

    $mapSchema = ConvertTo-RequiredInteger -Value $Map.schemaVersion -Name 'map schemaVersion' -Minimum 1
    if ($mapSchema -ne 1) { throw "unsupported map schema $mapSchema" }
    $schema = ConvertTo-RequiredInteger -Value $Certificate.schemaVersion -Name 'certificate schemaVersion' -Minimum 1
    if ($schema -ne 2 -and $schema -ne 3) { throw "unsupported certification schema $schema" }
    $basis = [ShardMapCertificationBasis]::HistoricalReplay
    if ($schema -eq 3) {
        if (-not $Certificate.PSObject.Properties['basis']) {
            throw 'schema-v3 certificate is missing basis'
        }
        $basisText = [string] $Certificate.basis
        if (-not [Enum]::TryParse[ShardMapCertificationBasis]($basisText, $false, [ref] $basis)) {
            throw "certificate has unsupported basis '$basisText'"
        }
    }
    $actualRun = ConvertTo-RequiredInteger -Value $Certificate.certificationRunId -Name 'certificationRunId' -Minimum 1
    if ($actualRun -ne $ExpectedCertificationRunId) {
        throw "certificate belongs to run $actualRun, not artifact run $ExpectedCertificationRunId"
    }
    $candidateRun = ConvertTo-RequiredInteger -Value $Certificate.candidateMapRunId -Name 'candidateMapRunId' -Minimum 1
    [void] (ConvertTo-RequiredInteger -Value $Certificate.auditSourceRunId -Name 'auditSourceRunId' -Minimum 1)
    $misses = ConvertTo-RequiredInteger -Value $Certificate.missCount -Name 'missCount'
    if ($misses -ne 0) { throw "certificate records $misses selection miss(es)" }
    if ($Certificate.escalated -isnot [bool]) { throw 'escalated must be a JSON boolean' }
    if ([bool] $Certificate.escalated) { throw 'certificate records an escalated plan, not reduction' }
    $audited = ConvertTo-RequiredInteger -Value $Certificate.auditedShards -Name 'auditedShards' -Minimum 2
    if ($Map.knownShards -isnot [array] -or $Map.alwaysRun -isnot [array]) {
        throw 'map knownShards and alwaysRun must be arrays'
    }
    $mapShards = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($nameValue in @($Map.knownShards) + @($Map.alwaysRun)) {
        $name = [string] $nameValue
        if ([string]::IsNullOrWhiteSpace($name)) { throw 'map contains an empty shard name' }
        if (-not $mapShards.Add($name)) { throw "map contains duplicate or overlapping shard '$name'" }
    }
    $mapShardCount = @($Map.knownShards).Count + @($Map.alwaysRun).Count
    if ($audited -ne $mapShardCount) {
        throw "auditedShards ($audited) does not cover the map shard universe ($mapShardCount)"
    }
    $wouldRun = ConvertTo-RequiredInteger -Value $Certificate.wouldRun -Name 'wouldRun' -Minimum 1
    $wouldSkip = ConvertTo-RequiredInteger -Value $Certificate.wouldSkip -Name 'wouldSkip' -Minimum 1
    if ($wouldRun + $wouldSkip -ne $audited) {
        throw "wouldRun ($wouldRun) plus wouldSkip ($wouldSkip) must equal auditedShards ($audited)"
    }
    # A clean complete matrix is valid evidence. Requiring a naturally occurring failure made
    # certification depend on CI being red, so a healthy repository could never enable selection.
    # The miss detector's adversarial tests prove that a skipped failure is rejected; this field
    # records what the real full-matrix audit observed and must be a non-negative integer.
    $failed = ConvertTo-RequiredInteger -Value $Certificate.failedShards -Name 'failedShards'
    if ($failed -gt $audited) { throw "failedShards ($failed) exceeds auditedShards ($audited)" }
    if ($failed -gt $wouldRun) {
        throw "failedShards ($failed) exceeds wouldRun ($wouldRun) despite a zero-miss certificate"
    }

    $mapSha = [string] $Map.sha
    $candidateSha = [string] $Certificate.candidateMapSha
    if ($mapSha -cnotmatch '^[0-9a-f]{40}$') { throw 'map sha is not a full lowercase commit id' }
    if ($candidateSha -cne $mapSha) { throw 'certificate and map identify different source commits' }
    if ([string] $Certificate.auditSourceSha -cnotmatch '^[0-9a-f]{40}$') {
        throw 'auditSourceSha is not a full lowercase commit id'
    }
    if ($basis -eq [ShardMapCertificationBasis]::FreshCoverage -and
        [string] $Certificate.auditSourceSha -cne $mapSha) {
        throw 'fresh-coverage evidence must audit the map source tree itself'
    }
    if ($basis -eq [ShardMapCertificationBasis]::FreshCoverage -and $candidateRun -ne $actualRun) {
        throw 'fresh-coverage evidence must bind the candidate map to its certification run'
    }
    if ($basis -eq [ShardMapCertificationBasis]::HistoricalReplay -and $candidateRun -eq $actualRun) {
        throw 'historical replay must cite a candidate map from an earlier workflow run'
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
    $map = [pscustomobject]@{
        schemaVersion = 1
        sha = $sha
        knownShards = @('Alpha', 'Beta')
        alwaysRun = @('Always')
    }
    $certificate = [pscustomobject]@{
        schemaVersion = 2
        candidateMapRunId = 10
        candidateMapSha = $sha
        auditSourceRunId = 11
        auditSourceSha = '89abcdef0123456789abcdef0123456789abcdef'
        certificationRunId = 12
        escalated = $false
        auditedShards = 3
        wouldRun = 2
        wouldSkip = 1
        failedShards = 0
        missCount = 0
    }
    try {
        Assert-CertifiedShardMap -Map $map -Certificate $certificate -ExpectedCertificationRunId 12
    }
    catch {
        [void] $failures.Add("valid certificate rejected: $($_.Exception.Message)")
    }

    $freshCertificate = $certificate.PSObject.Copy()
    $freshCertificate.schemaVersion = 3
    $freshCertificate | Add-Member -NotePropertyName basis -NotePropertyValue FreshCoverage
    $freshCertificate.candidateMapRunId = 12
    $freshCertificate.auditSourceSha = $sha
    try {
        Assert-CertifiedShardMap -Map $map -Certificate $freshCertificate -ExpectedCertificationRunId 12
    }
    catch {
        [void] $failures.Add("valid fresh-coverage certificate rejected: $($_.Exception.Message)")
    }
    $badFresh = $freshCertificate.PSObject.Copy()
    $badFresh.auditSourceSha = $certificate.auditSourceSha
    Assert-Rejection { Assert-CertifiedShardMap $map $badFresh 12 } `
        'fresh coverage for another tree was accepted' 'must audit the map source tree'
    $badFresh = $freshCertificate.PSObject.Copy()
    $badFresh.candidateMapRunId = 10
    Assert-Rejection { Assert-CertifiedShardMap $map $badFresh 12 } `
        'fresh coverage from another map run was accepted' 'bind the candidate map'

    $bad = $certificate.PSObject.Copy(); $bad.missCount = 1
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a miss-bearing certificate was accepted' 'selection miss'
    $bad = $certificate.PSObject.Copy(); $bad.certificationRunId = 99
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate from another run was accepted' 'certificate belongs to run'
    $bad = $certificate.PSObject.Copy(); $bad.certificationRunId = '12'
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a string certification run id was accepted' 'must be a JSON integer'
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
    $bad = $certificate.PSObject.Copy(); $bad.auditedShards = 4
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate that did not audit the full map universe was accepted' 'does not cover the map shard universe'
    $bad = $certificate.PSObject.Copy(); $bad.wouldRun = 1
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate whose selection did not partition the audited matrix was accepted' 'must equal auditedShards'
    $bad = $certificate.PSObject.Copy(); $bad.failedShards = -1
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate with a negative failure count was accepted' 'failedShards must be at least 0'
    $withObservedFailure = $certificate.PSObject.Copy(); $withObservedFailure.failedShards = 1
    try {
        Assert-CertifiedShardMap -Map $map -Certificate $withObservedFailure -ExpectedCertificationRunId 12
    }
    catch {
        [void] $failures.Add("a valid certificate with an observed failure was rejected: $($_.Exception.Message)")
    }
    $bad = $certificate.PSObject.Copy(); $bad.failedShards = 4
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a certificate with more failures than audited shards was accepted' 'exceeds auditedShards'
    $bad = $certificate.PSObject.Copy(); $bad.failedShards = 3
    Assert-Rejection { Assert-CertifiedShardMap $map $bad 12 } `
        'a zero-miss certificate with more failures than selected shards was accepted' 'exceeds wouldRun'
    $badMap = $map.PSObject.Copy(); $badMap.schemaVersion = 2
    Assert-Rejection { Assert-CertifiedShardMap $badMap $certificate 12 } `
        'a map with an unsupported schema was accepted' 'unsupported map schema'
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
