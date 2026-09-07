<#
.SYNOPSIS
    Produces a run-bound certificate for a complete, zero-miss shard-map audit.

.DESCRIPTION
    Keeps the certification decision out of workflow YAML so the shipping policy can be exercised
    directly. A complete clean matrix is eligible: correctness of the miss detector is covered by
    adversarial tests, while the real audit establishes the exact shard universe and whether the
    selector would reduce it. Requiring CI to fail before a map can be certified deadlocks a healthy
    repository in full-matrix mode.
#>
[CmdletBinding(DefaultParameterSetName = 'Build')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $AuditFile,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $OutcomesFile,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [long] $CandidateMapRunId,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [long] $AuditSourceRunId,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $AuditSourceSha,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [long] $CertificationRunId,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $OutDirectory,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-RequiredInteger {
    param($Value, [string] $Name, [long] $Minimum = 0)

    $isInteger = $Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
    if (-not $isInteger) { throw "$Name must be an integer" }
    $number = [long] $Value
    if ($number -lt $Minimum) { throw "$Name must be at least $Minimum" }
    return $number
}

function Get-CertificationDecision {
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] $Audit,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Outcomes,
        [Parameter(Mandatory)] [long] $MapRunId,
        [Parameter(Mandatory)] [long] $SourceRunId,
        [Parameter(Mandatory)] [string] $SourceSha,
        [Parameter(Mandatory)] [long] $CertificateRunId
    )

    foreach ($name in 'schemaVersion', 'sha', 'knownShards', 'alwaysRun') {
        if (-not $Map.PSObject.Properties[$name]) { throw "map is missing $name" }
    }
    foreach ($name in 'Escalated', 'TotalShards', 'WouldRun', 'WouldSkip',
        'WouldRunShards', 'WouldSkipShards', 'Failed', 'Missed', 'MissCount') {
        if (-not $Audit.PSObject.Properties[$name]) { throw "audit is missing $name" }
    }
    if ((ConvertTo-RequiredInteger $Map.schemaVersion 'map schemaVersion' 1) -ne 1) {
        throw "unsupported map schemaVersion $($Map.schemaVersion)"
    }
    $mapSha = [string] $Map.sha
    if ($mapSha -cnotmatch '^[0-9a-f]{40}$') { throw 'map sha must be a lowercase 40-character Git SHA' }
    if ($SourceSha -cnotmatch '^[0-9a-f]{40}$') { throw 'audit source sha must be a lowercase 40-character Git SHA' }
    [void] (ConvertTo-RequiredInteger $MapRunId 'candidate map run id' 1)
    [void] (ConvertTo-RequiredInteger $SourceRunId 'audit source run id' 1)
    [void] (ConvertTo-RequiredInteger $CertificateRunId 'certification run id' 1)
    if ($Map.knownShards -isnot [array] -or $Map.alwaysRun -isnot [array]) {
        throw 'map knownShards and alwaysRun must be arrays'
    }
    if ($Audit.Escalated -isnot [bool]) { throw 'audit Escalated must be a JSON boolean' }
    if ($Audit.WouldRunShards -isnot [array] -or $Audit.WouldSkipShards -isnot [array] -or
        $Audit.Missed -isnot [array]) {
        throw 'audit shard partitions and Missed must be JSON arrays'
    }

    $universe = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($nameValue in @($Map.knownShards) + @($Map.alwaysRun)) {
        $name = [string] $nameValue
        if ([string]::IsNullOrWhiteSpace($name)) { throw 'map contains an empty shard name' }
        if (-not $universe.Add($name)) { throw "map contains duplicate or overlapping shard '$name'" }
    }
    if ($universe.Count -lt 2) { throw 'map must contain at least two shards to prove reduction' }

    $outcomeNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $failed = 0
    foreach ($outcome in $Outcomes) {
        if (-not $outcome.PSObject.Properties['shard'] -or -not $outcome.PSObject.Properties['outcome']) {
            throw 'every outcome must contain shard and outcome'
        }
        $name = [string] $outcome.shard
        if (-not $universe.Contains($name)) { throw "outcome names unknown shard '$name'" }
        if (-not $outcomeNames.Add($name)) { throw "duplicate outcome for shard '$name'" }
        $conclusion = [string] $outcome.outcome
        if ($conclusion -notin @('success', 'failure')) {
            throw "shard '$name' has unauditable outcome '$conclusion'"
        }
        if ($conclusion -eq 'failure') { $failed++ }
    }
    if ($outcomeNames.Count -ne $universe.Count) {
        $missing = @($universe | Where-Object { -not $outcomeNames.Contains($_) } | Sort-Object)
        throw "outcomes do not cover the map shard universe (missing: $($missing -join ', '))"
    }

    $runNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($nameValue in @($Audit.WouldRunShards)) {
        $name = [string] $nameValue
        if (-not $universe.Contains($name)) { throw "audit WouldRunShards names unknown shard '$name'" }
        if (-not $runNames.Add($name)) { throw "audit WouldRunShards contains duplicate '$name'" }
    }
    $skipNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($nameValue in @($Audit.WouldSkipShards)) {
        $name = [string] $nameValue
        if (-not $universe.Contains($name)) { throw "audit WouldSkipShards names unknown shard '$name'" }
        if ($runNames.Contains($name)) { throw "audit shard partitions overlap at '$name'" }
        if (-not $skipNames.Add($name)) { throw "audit WouldSkipShards contains duplicate '$name'" }
    }
    if ($runNames.Count + $skipNames.Count -ne $universe.Count) {
        throw 'audit shard partitions do not cover the map shard universe'
    }

    $total = ConvertTo-RequiredInteger $Audit.TotalShards 'audit TotalShards' 1
    $wouldRun = ConvertTo-RequiredInteger $Audit.WouldRun 'audit WouldRun'
    $wouldSkip = ConvertTo-RequiredInteger $Audit.WouldSkip 'audit WouldSkip'
    $auditFailed = ConvertTo-RequiredInteger $Audit.Failed 'audit Failed'
    $missCount = ConvertTo-RequiredInteger $Audit.MissCount 'audit MissCount'
    if ($total -ne $universe.Count) { throw 'audit TotalShards does not match the map shard universe' }
    if ($auditFailed -ne $failed) { throw 'audit Failed does not match the source outcomes' }
    if ($missCount -ne @($Audit.Missed).Count) { throw 'audit MissCount does not match audit Missed' }
    if ($wouldRun + $wouldSkip -ne $total) {
        throw 'audit WouldRun plus WouldSkip must equal TotalShards'
    }
    if ($wouldRun -ne $runNames.Count -or $wouldSkip -ne $skipNames.Count) {
        throw 'audit shard partition counts do not match their shard arrays'
    }
    if ([bool] $Audit.Escalated -and ($runNames.Count -ne $universe.Count -or $skipNames.Count -ne 0)) {
        throw 'an escalated audit must run the complete shard universe'
    }

    $expectedMissed = @($Outcomes |
        Where-Object { [string] $_.outcome -eq 'failure' -and $skipNames.Contains([string] $_.shard) } |
        ForEach-Object { [string] $_.shard } | Sort-Object)
    $reportedMissed = @($Audit.Missed | ForEach-Object { [string] $_ } | Sort-Object)
    if (($expectedMissed -join "`n") -cne ($reportedMissed -join "`n")) {
        throw 'audit Missed does not equal the failed shards in WouldSkipShards'
    }
    if ($missCount -ne $expectedMissed.Count) {
        throw 'audit MissCount does not match the recomputed selection misses'
    }

    $eligible = -not [bool] $Audit.Escalated -and $missCount -eq 0 -and
        $wouldRun -gt 0 -and $wouldSkip -gt 0
    $certificate = $null
    if ($eligible) {
        $certificate = [pscustomobject] [ordered]@{
            schemaVersion = 2
            candidateMapRunId = $MapRunId
            candidateMapSha = $mapSha
            auditSourceRunId = $SourceRunId
            auditSourceSha = $SourceSha
            certificationRunId = $CertificateRunId
            escalated = $false
            auditedShards = $total
            wouldRun = $wouldRun
            wouldSkip = $wouldSkip
            failedShards = $failed
            missCount = 0
        }
    }
    return [pscustomobject]@{ Eligible = $eligible; Certificate = $certificate }
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) } }
    function Assert-Rejected { param([scriptblock] $Action, [string] $What)
        $rejected = $false
        try { & $Action | Out-Null } catch { $rejected = $true }
        Assert-True $rejected $What
    }

    $sha = '0123456789abcdef0123456789abcdef01234567'
    $map = [pscustomobject]@{ schemaVersion = 1; sha = $sha; knownShards = @('A', 'B'); alwaysRun = @('Always') }
    $outcomes = @(
        [pscustomobject]@{ shard = 'A'; outcome = 'success' },
        [pscustomobject]@{ shard = 'B'; outcome = 'success' },
        [pscustomobject]@{ shard = 'Always'; outcome = 'success' }
    )
    $audit = [pscustomobject]@{
        Escalated = $false; TotalShards = 3; WouldRun = 2; WouldSkip = 1
        WouldRunShards = @('A', 'Always'); WouldSkipShards = @('B')
        Failed = 0; Missed = @(); MissCount = 0
    }
    $decision = Get-CertificationDecision $map $audit $outcomes 10 11 $sha 12
    Assert-True $decision.Eligible 'a complete clean reduction audit was not certifiable'
    Assert-True ($decision.Certificate.failedShards -eq 0) 'a clean certificate did not record zero failures'

    $failedOutcomes = @($outcomes | ForEach-Object { $_.PSObject.Copy() })
    $failedOutcomes[0].outcome = 'failure'
    $failureAudit = $audit.PSObject.Copy(); $failureAudit.Failed = 1
    $decision = Get-CertificationDecision $map $failureAudit $failedOutcomes 10 11 $sha 12
    Assert-True $decision.Eligible 'a zero-miss audit with a selected failure was not certifiable'

    $skippedFailureOutcomes = @($outcomes | ForEach-Object { $_.PSObject.Copy() })
    $skippedFailureOutcomes[1].outcome = 'failure'
    $missAudit = $audit.PSObject.Copy(); $missAudit.Failed = 1; $missAudit.Missed = @('B'); $missAudit.MissCount = 1
    $decision = Get-CertificationDecision $map $missAudit $skippedFailureOutcomes 10 11 $sha 12
    Assert-True (-not $decision.Eligible) 'an audit with a skipped failure was certifiable'
    $falseZeroMiss = $missAudit.PSObject.Copy(); $falseZeroMiss.Missed = @(); $falseZeroMiss.MissCount = 0
    Assert-Rejected { Get-CertificationDecision $map $falseZeroMiss $skippedFailureOutcomes 10 11 $sha 12 } `
        'a false zero-miss report whose counts still balanced was accepted'
    $escalated = $audit.PSObject.Copy(); $escalated.Escalated = $true; $escalated.WouldRun = 3; $escalated.WouldSkip = 0
    $escalated.WouldRunShards = @('A', 'B', 'Always'); $escalated.WouldSkipShards = @()
    $decision = Get-CertificationDecision $map $escalated $outcomes 10 11 $sha 12
    Assert-True (-not $decision.Eligible) 'an escalated audit was certifiable'
    $noReduction = $audit.PSObject.Copy(); $noReduction.WouldRun = 3; $noReduction.WouldSkip = 0
    $noReduction.WouldRunShards = @('A', 'B', 'Always'); $noReduction.WouldSkipShards = @()
    $decision = Get-CertificationDecision $map $noReduction $outcomes 10 11 $sha 12
    Assert-True (-not $decision.Eligible) 'an audit that skips nothing was certifiable'

    Assert-Rejected { Get-CertificationDecision $map $audit @($outcomes[0..1]) 10 11 $sha 12 } `
        'incomplete outcomes were accepted'
    $duplicate = @($outcomes | ForEach-Object { $_.PSObject.Copy() }); $duplicate[2].shard = 'A'
    Assert-Rejected { Get-CertificationDecision $map $audit $duplicate 10 11 $sha 12 } `
        'duplicate outcomes were accepted'
    $badCount = $audit.PSObject.Copy(); $badCount.TotalShards = 4
    Assert-Rejected { Get-CertificationDecision $map $badCount $outcomes 10 11 $sha 12 } `
        'an audit count outside the map universe was accepted'
    $stringBoolean = $audit.PSObject.Copy(); $stringBoolean.Escalated = 'false'
    Assert-Rejected { Get-CertificationDecision $map $stringBoolean $outcomes 10 11 $sha 12 } `
        'a string audit boolean was accepted'
    $stringCount = $audit.PSObject.Copy(); $stringCount.TotalShards = '3'
    Assert-Rejected { Get-CertificationDecision $map $stringCount $outcomes 10 11 $sha 12 } `
        'a string audit count was accepted'
    $overlap = $audit.PSObject.Copy(); $overlap.WouldSkipShards = @('A')
    Assert-Rejected { Get-CertificationDecision $map $overlap $outcomes 10 11 $sha 12 } `
        'overlapping selected and skipped shard arrays were accepted'
    $impossibleFailureCount = $audit.PSObject.Copy(); $impossibleFailureCount.Failed = 3
    $threeFailures = @($outcomes | ForEach-Object { $_.PSObject.Copy() })
    $threeFailures[0].outcome = 'failure'; $threeFailures[1].outcome = 'failure'; $threeFailures[2].outcome = 'failure'
    Assert-Rejected { Get-CertificationDecision $map $impossibleFailureCount $threeFailures 10 11 $sha 12 } `
        'a zero-miss audit with more failures than selected shards was accepted'

    if ($failures.Count -gt 0) {
        Write-Host 'New-ShardMapCertificate self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'New-ShardMapCertificate self-test passed.'
    exit 0
}

$mapObject = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
$auditObject = Get-Content -LiteralPath $AuditFile -Raw | ConvertFrom-Json
$outcomeObjects = @(Get-Content -LiteralPath $OutcomesFile -Raw | ConvertFrom-Json)
$decision = Get-CertificationDecision -Map $mapObject -Audit $auditObject -Outcomes $outcomeObjects `
    -MapRunId $CandidateMapRunId -SourceRunId $AuditSourceRunId -SourceSha $AuditSourceSha `
    -CertificateRunId $CertificationRunId

if (-not $decision.Eligible) {
    Write-Host 'candidate did not complete a zero-miss reduction audit - not certifying it'
    exit 0
}

$certificatePath = Join-Path $OutDirectory 'certification.json'
$mapPath = Join-Path $OutDirectory 'shard-map.json'
if ((Test-Path -LiteralPath $certificatePath) -or (Test-Path -LiteralPath $mapPath)) {
    throw "refusing to overwrite an existing certification in $OutDirectory"
}
New-Item -ItemType Directory -Path $OutDirectory -Force | Out-Null
Copy-Item -LiteralPath $MapFile -Destination $mapPath
$decision.Certificate | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $certificatePath -Encoding utf8
Write-Host "certified shard map from run $CandidateMapRunId after complete audit run $AuditSourceRunId"
