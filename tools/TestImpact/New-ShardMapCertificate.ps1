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
    [Parameter(Mandatory, ParameterSetName = 'Build')]
    [ValidateSet('HistoricalReplay', 'FreshCoverage')]
    [string] $Basis,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $OutDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $DecisionFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum ShardMapAuditDisposition {
    Certified
    SafeNoReduction
    RevokedMiss
}

enum ShardMapCertificationBasis {
    HistoricalReplay
    FreshCoverage
}

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
        [Parameter(Mandatory)] [long] $CertificateRunId,
        [ShardMapCertificationBasis] $CertificationBasis = [ShardMapCertificationBasis]::HistoricalReplay
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
    if ($CertificationBasis -eq [ShardMapCertificationBasis]::FreshCoverage) {
        if ($SourceSha -cne $mapSha) {
            throw 'fresh-coverage evidence must audit the map source tree itself'
        }
        if ($MapRunId -ne $CertificateRunId) {
            throw 'fresh-coverage evidence must bind the candidate map to its certification run'
        }
    }
    elseif ($MapRunId -eq $CertificateRunId) {
        throw 'historical replay must cite a candidate map from an earlier workflow run'
    }
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

    # A shard retired from the manifest keeps its slot in knownShards, because the file index
    # addresses shards by position and renumbering would invalidate every recorded line range.
    # No job runs it, so it can never produce an outcome. Demanding one deadlocks certification
    # permanently: the previous map only advances when something certifies, so the first
    # retirement freezes every later map in full-matrix mode. Exclude retired slots from the
    # coverage requirement instead, and require that they stay silent.
    $retired = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    if ($Map.PSObject.Properties['retiredShards']) {
        if ($Map.retiredShards -isnot [array]) { throw 'map retiredShards must be an array' }
        # Only an INDEXED shard can legitimately be retired-but-present. Complete-CiMapWorkloads
        # drops a retired always-run entry outright and keeps a retired indexed one solely because
        # the file index addresses shards by position. Accepting any universe member would let the
        # exemption shield a LIVE always-run shard from the coverage requirement - precisely what
        # that requirement exists to catch.
        $indexed = [System.Collections.Generic.HashSet[string]]::new(
            [string[]] @(@($Map.knownShards) | ForEach-Object { [string] $_ }), [StringComparer]::Ordinal)
        foreach ($nameValue in @($Map.retiredShards)) {
            $name = [string] $nameValue
            if (-not $universe.Contains($name)) { throw "retiredShards names unknown shard '$name'" }
            if (-not $indexed.Contains($name)) {
                throw "retiredShards names '$name', which is not an indexed shard"
            }
            if (-not $retired.Add($name)) { throw "map retiredShards contains duplicate '$name'" }
        }
    }
    $expected = [System.Collections.Generic.HashSet[string]]::new(
        [string[]] @($universe | Where-Object { -not $retired.Contains($_) }), [StringComparer]::Ordinal)
    if ($expected.Count -eq 0) { throw 'every shard in the map is retired' }

    $outcomeNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $failed = 0
    foreach ($outcome in $Outcomes) {
        if (-not $outcome.PSObject.Properties['shard'] -or -not $outcome.PSObject.Properties['outcome']) {
            throw 'every outcome must contain shard and outcome'
        }
        $name = [string] $outcome.shard
        if (-not $universe.Contains($name)) { throw "outcome names unknown shard '$name'" }
        if ($retired.Contains($name)) { throw "retired shard '$name' produced an outcome" }
        if (-not $outcomeNames.Add($name)) { throw "duplicate outcome for shard '$name'" }
        $conclusion = [string] $outcome.outcome
        if ($conclusion -notin @('success', 'failure')) {
            throw "shard '$name' has unauditable outcome '$conclusion'"
        }
        if ($conclusion -eq 'failure') { $failed++ }
    }
    if ($outcomeNames.Count -ne $expected.Count) {
        $missing = @($expected | Where-Object { -not $outcomeNames.Contains($_) } | Sort-Object)
        throw "outcomes do not cover the live map shard universe (missing: $($missing -join ', '))"
    }

    # The audit partitions the shards the selector can actually dispatch, which Measure-SelectionMiss
    # takes from the shard manifest ($AllShards.Count -> TotalShards). A retired slot is not in the
    # manifest, so it appears in neither partition: compare against the live set, not the raw
    # universe, or every retirement fails this check instead of the outcome check above.
    $runNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($nameValue in @($Audit.WouldRunShards)) {
        $name = [string] $nameValue
        if (-not $universe.Contains($name)) { throw "audit WouldRunShards names unknown shard '$name'" }
        if ($retired.Contains($name)) { throw "audit WouldRunShards names retired shard '$name'" }
        if (-not $runNames.Add($name)) { throw "audit WouldRunShards contains duplicate '$name'" }
    }
    $skipNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($nameValue in @($Audit.WouldSkipShards)) {
        $name = [string] $nameValue
        if (-not $universe.Contains($name)) { throw "audit WouldSkipShards names unknown shard '$name'" }
        if ($retired.Contains($name)) { throw "audit WouldSkipShards names retired shard '$name'" }
        if ($runNames.Contains($name)) { throw "audit shard partitions overlap at '$name'" }
        if (-not $skipNames.Add($name)) { throw "audit WouldSkipShards contains duplicate '$name'" }
    }
    if ($runNames.Count + $skipNames.Count -ne $expected.Count) {
        throw 'audit shard partitions do not cover the live map shard universe'
    }

    $total = ConvertTo-RequiredInteger $Audit.TotalShards 'audit TotalShards' 1
    $wouldRun = ConvertTo-RequiredInteger $Audit.WouldRun 'audit WouldRun'
    $wouldSkip = ConvertTo-RequiredInteger $Audit.WouldSkip 'audit WouldSkip'
    $auditFailed = ConvertTo-RequiredInteger $Audit.Failed 'audit Failed'
    $missCount = ConvertTo-RequiredInteger $Audit.MissCount 'audit MissCount'
    if ($total -ne $expected.Count) { throw 'audit TotalShards does not match the live map shard universe' }
    if ($auditFailed -ne $failed) { throw 'audit Failed does not match the source outcomes' }
    if ($missCount -ne @($Audit.Missed).Count) { throw 'audit MissCount does not match audit Missed' }
    if ($wouldRun + $wouldSkip -ne $total) {
        throw 'audit WouldRun plus WouldSkip must equal TotalShards'
    }
    if ($wouldRun -ne $runNames.Count -or $wouldSkip -ne $skipNames.Count) {
        throw 'audit shard partition counts do not match their shard arrays'
    }
    if ([bool] $Audit.Escalated -and ($runNames.Count -ne $expected.Count -or $skipNames.Count -ne 0)) {
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
    $disposition = if ($missCount -gt 0) {
        [ShardMapAuditDisposition]::RevokedMiss
    }
    elseif ($eligible) {
        [ShardMapAuditDisposition]::Certified
    }
    else {
        [ShardMapAuditDisposition]::SafeNoReduction
    }
    $certificate = $null
    if ($eligible) {
        $certificate = [pscustomobject] [ordered]@{
            schemaVersion = 3
            basis = $CertificationBasis.ToString()
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
    return [pscustomobject]@{
        Eligible = $eligible
        Disposition = $disposition
        Certificate = $certificate
        Audit = [pscustomobject] [ordered]@{
            schemaVersion = 2
            basis = $CertificationBasis.ToString()
            candidateMapRunId = $MapRunId
            auditSourceRunId = $SourceRunId
            auditSourceSha = $SourceSha
            certificationRunId = $CertificateRunId
            disposition = $disposition.ToString()
            escalated = [bool] $Audit.Escalated
            auditedShards = $total
            wouldRun = $wouldRun
            wouldSkip = $wouldSkip
            failedShards = $failed
            missCount = $missCount
        }
    }
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
    Assert-True ($decision.Disposition -eq [ShardMapAuditDisposition]::Certified) `
        'a certifiable audit did not produce the Certified disposition'
    Assert-True ($decision.Certificate.schemaVersion -eq 3 -and
        $decision.Certificate.basis -ceq 'HistoricalReplay') `
        'the default historical certification basis was not recorded'
    Assert-True ($decision.Certificate.failedShards -eq 0) 'a clean certificate did not record zero failures'

    $freshDecision = Get-CertificationDecision $map $audit $outcomes 12 11 $sha 12 `
        -CertificationBasis FreshCoverage
    Assert-True ($freshDecision.Certificate.basis -ceq 'FreshCoverage') `
        'the fresh-coverage certification basis was not recorded'
    Assert-Rejected { Get-CertificationDecision $map $audit $outcomes 10 11 $sha 12 `
            -CertificationBasis FreshCoverage } `
        'fresh-coverage certification accepted a candidate from another map workflow run'
    Assert-Rejected { Get-CertificationDecision $map $audit $outcomes 12 11 `
            '89abcdef0123456789abcdef0123456789abcdef' 12 -CertificationBasis FreshCoverage } `
        'fresh-coverage certification accepted outcomes from a different source tree'

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
    Assert-True ($decision.Disposition -eq [ShardMapAuditDisposition]::RevokedMiss) `
        'a selection miss did not explicitly revoke older evidence'
    $falseZeroMiss = $missAudit.PSObject.Copy(); $falseZeroMiss.Missed = @(); $falseZeroMiss.MissCount = 0
    Assert-Rejected { Get-CertificationDecision $map $falseZeroMiss $skippedFailureOutcomes 10 11 $sha 12 } `
        'a false zero-miss report whose counts still balanced was accepted'
    $escalated = $audit.PSObject.Copy(); $escalated.Escalated = $true; $escalated.WouldRun = 3; $escalated.WouldSkip = 0
    $escalated.WouldRunShards = @('A', 'B', 'Always'); $escalated.WouldSkipShards = @()
    $decision = Get-CertificationDecision $map $escalated $outcomes 10 11 $sha 12
    Assert-True (-not $decision.Eligible) 'an escalated audit was certifiable'
    Assert-True ($decision.Disposition -eq [ShardMapAuditDisposition]::SafeNoReduction) `
        'a zero-miss escalation was treated as a revocation'
    $noReduction = $audit.PSObject.Copy(); $noReduction.WouldRun = 3; $noReduction.WouldSkip = 0
    $noReduction.WouldRunShards = @('A', 'B', 'Always'); $noReduction.WouldSkipShards = @()
    $decision = Get-CertificationDecision $map $noReduction $outcomes 10 11 $sha 12
    Assert-True (-not $decision.Eligible) 'an audit that skips nothing was certifiable'

    # A shard retired from the manifest keeps its indexed slot but no longer has a job. Demanding
    # an outcome for it froze certification permanently on 2026-09-17 and every pull request ran
    # the full matrix from 09-20 on. Retirement must stay certifiable.
    $retiredMap = [pscustomobject]@{
        schemaVersion = 1; sha = $sha
        knownShards = @('A', 'B', 'Retired'); alwaysRun = @('Always'); retiredShards = @('Retired')
    }
    # Measure-SelectionMiss partitions the SHARD MANIFEST (TotalShards = $AllShards.Count), so a
    # retired slot appears in neither partition and in no total. Asserting it the other way is what
    # let the first attempt at this fix pass locally and still fail CI at the partition check.
    $retiredAudit = [pscustomobject]@{
        Escalated = $false; TotalShards = 3; WouldRun = 2; WouldSkip = 1
        WouldRunShards = @('A', 'Always'); WouldSkipShards = @('B')
        Failed = 0; Missed = @(); MissCount = 0
    }
    $decision = Get-CertificationDecision $retiredMap $retiredAudit $outcomes 10 11 $sha 12
    Assert-True $decision.Eligible 'a retired indexed shard blocked certification'
    Assert-True ($decision.Disposition -eq [ShardMapAuditDisposition]::Certified) `
        'a map carrying a retirement did not produce the Certified disposition'
    $retiredOutcomes = @($outcomes | ForEach-Object { $_.PSObject.Copy() }) +
        [pscustomobject]@{ shard = 'Retired'; outcome = 'success' }
    Assert-Rejected { Get-CertificationDecision $retiredMap $retiredAudit $retiredOutcomes 10 11 $sha 12 } `
        'a retired shard reporting an outcome was accepted'
    $retiredInPartition = $retiredAudit.PSObject.Copy()
    $retiredInPartition.WouldSkipShards = @('B', 'Retired'); $retiredInPartition.WouldSkip = 2
    $retiredInPartition.TotalShards = 4
    Assert-Rejected { Get-CertificationDecision $retiredMap $retiredInPartition $outcomes 10 11 $sha 12 } `
        'an audit partitioning a retired shard was accepted'
    $unknownRetired = $retiredMap.PSObject.Copy(); $unknownRetired.retiredShards = @('Absent')
    Assert-Rejected { Get-CertificationDecision $unknownRetired $retiredAudit $outcomes 10 11 $sha 12 } `
        'retiredShards naming a shard outside the map universe was accepted'
    # An always-run entry is DROPPED on retirement rather than kept, so naming one here could only
    # exempt a shard that is still live. The fixture has to isolate that: give the falsely-retired
    # always-run shard NO outcome, so without the indexed check it is silently exempted and the
    # map certifies while the shard the requirement exists to police reported nothing. Asserting it
    # with an outcome present proves nothing - the "retired shard produced an outcome" guard throws
    # first and the case passes whether or not the indexed check is there.
    $falselyRetired = [pscustomobject]@{
        schemaVersion = 1; sha = $sha
        knownShards = @('A', 'B'); alwaysRun = @('Always'); retiredShards = @('Always')
    }
    $twoLiveAudit = [pscustomobject]@{
        Escalated = $false; TotalShards = 2; WouldRun = 1; WouldSkip = 1
        WouldRunShards = @('A'); WouldSkipShards = @('B')
        Failed = 0; Missed = @(); MissCount = 0
    }
    $withoutAlways = @($outcomes | Where-Object { $_.shard -cne 'Always' })
    Assert-Rejected { Get-CertificationDecision $falselyRetired $twoLiveAudit $withoutAlways 10 11 $sha 12 } `
        'retiredShards naming a live always-run shard exempted it from the coverage requirement'
    $allRetired = $retiredMap.PSObject.Copy()
    $allRetired.retiredShards = @('A', 'B', 'Retired', 'Always')
    Assert-Rejected { Get-CertificationDecision $allRetired $retiredAudit $outcomes 10 11 $sha 12 } `
        'a map whose every shard is retired was certifiable'

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
$certificationBasis = [ShardMapCertificationBasis] $Basis
$decision = Get-CertificationDecision -Map $mapObject -Audit $auditObject -Outcomes $outcomeObjects `
    -MapRunId $CandidateMapRunId -SourceRunId $AuditSourceRunId -SourceSha $AuditSourceSha `
    -CertificateRunId $CertificationRunId -CertificationBasis $certificationBasis

if (Test-Path -LiteralPath $DecisionFile) {
    throw "refusing to overwrite an existing audit decision at $DecisionFile"
}
$decisionParent = Split-Path -Parent $DecisionFile
if ($decisionParent) { New-Item -ItemType Directory -Path $decisionParent -Force | Out-Null }
$decision.Audit | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $DecisionFile -Encoding utf8

if (-not $decision.Eligible) {
    Write-Host "candidate audit disposition is $($decision.Disposition) - not certifying it"
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
exit 0
