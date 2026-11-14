<#
.SYNOPSIS
    Downloads the newest positively certified shard map that has not been explicitly revoked.

.DESCRIPTION
    Cancellation is not negative audit evidence, and a successful audit that could not reduce a
    particular historical diff does not invalidate an older zero-miss certificate. The resolver
    therefore scans backward through those neutral states. It stops only at an explicit audited
    selection miss or at a failed run that predates typed audit decisions and cannot be classified
    safely. This preserves fail-closed revocation without letting queue cancellation disable
    selective CI repo-wide.

    A run that failed (or timed out, or never started) WITHOUT uploading a typed audit decision
    carries no audit evidence either: the audit step writes its decision before it exits, so such a
    run broke before or outside the audit. It is neutral, with one limit. The newest earlier
    certified map is used only while it is no older than FailedRunFallbackMaxAgeDays; past that, or
    when no earlier certified map exists, selection fails closed exactly as before. A run that
    uploaded a decision keeps its meaning: RevokedMiss still revokes everything older, and a
    decision that cannot be read still blocks.

.PARAMETER FailedRunFallbackMaxAgeDays
    THE setting for the rule above: how old (in days, by the map run's creation time) a certified
    map may be and still be used once a newer map run has failed without a typed audit decision.
    Default 14. 0 restores the old behaviour, where any such failure blocks every older map.
#>
[CmdletBinding(DefaultParameterSetName = 'Resolve')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $Repository,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $OutDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $ResultFile,
    [Parameter(ParameterSetName = 'Resolve')] [string] $MapBranch = 'master',
    [Parameter(ParameterSetName = 'Resolve')] [ValidateRange(0, 3650)] [int] $FailedRunFallbackMaxAgeDays = 14,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum CertifiedShardMapResolutionStatus {
    Found
    NoCertifiedRun
    RevokedByAuditMiss
    UnclassifiedFailure
    CertificateUnavailable
    FailedRunFallbackExpired
}

enum ShardMapAuditDisposition {
    Certified
    SafeNoReduction
    RevokedMiss
}

enum ShardMapCertificationBasis {
    HistoricalReplay
    FreshCoverage
}

function ConvertTo-AuditDecision {
    param($Decision, [long] $ExpectedRunId)

    foreach ($name in 'schemaVersion', 'certificationRunId', 'disposition', 'missCount') {
        if (-not $Decision.PSObject.Properties[$name]) { throw "audit decision is missing $name" }
    }
    $schema = 0
    if (-not [int]::TryParse([string] $Decision.schemaVersion, [ref] $schema) -or
        ($schema -ne 1 -and $schema -ne 2)) {
        throw 'audit decision schemaVersion must be the integer 1 or 2'
    }
    if ($schema -eq 2) {
        if (-not $Decision.PSObject.Properties['basis']) {
            throw 'schema-v2 audit decision is missing basis'
        }
        $basis = [ShardMapCertificationBasis]::HistoricalReplay
        if (-not [Enum]::TryParse[ShardMapCertificationBasis]([string] $Decision.basis, $false, [ref] $basis)) {
            throw "audit decision has unsupported basis '$($Decision.basis)'"
        }
    }
    $runId = 0L
    if (-not [long]::TryParse([string] $Decision.certificationRunId, [ref] $runId) -or
        $runId -ne $ExpectedRunId) {
        throw 'audit decision certificationRunId does not match its workflow run'
    }
    $missCount = 0
    if (-not [int]::TryParse([string] $Decision.missCount, [ref] $missCount) -or $missCount -lt 0) {
        throw 'audit decision missCount must be a non-negative integer'
    }
    $disposition = [ShardMapAuditDisposition]::Certified
    $dispositionText = [string] $Decision.disposition
    if (-not [Enum]::TryParse[ShardMapAuditDisposition]($dispositionText, $false, [ref] $disposition)) {
        throw "audit decision has unsupported disposition '$dispositionText'"
    }
    if (($disposition -eq [ShardMapAuditDisposition]::RevokedMiss) -ne ($missCount -gt 0)) {
        throw 'audit decision disposition contradicts its miss count'
    }
    return $disposition
}

function Resolve-CertifiedShardMapRun {
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Runs,
        [Parameter(Mandatory)] [scriptblock] $DownloadCertificate,
        [Parameter(Mandatory)] [scriptblock] $DownloadDecision,
        [int] $FailedRunFallbackMaxAgeDays = 14,
        [DateTimeOffset] $Now = [DateTimeOffset]::UtcNow
    )

    # Newest first: the runs that failed without a typed audit decision and were stepped over.
    $skippedFailures = [System.Collections.Generic.List[long]]::new()
    function New-Resolution {
        param([CertifiedShardMapResolutionStatus] $Status, [long] $RunId, $Candidate = $null)
        $created = if ($null -ne $Candidate) { [DateTimeOffset] $Candidate.CreatedAt } else { $null }
        return [pscustomobject]@{
            Status = $Status
            RunId = $RunId
            MapCreatedAt = $created
            MapAgeDays = $(if ($null -ne $created) { ($Now - $created).TotalDays } else { $null })
            SkippedFailedRunIds = @($skippedFailures)
        }
    }

    $targetRuns = [System.Collections.Generic.List[object]]::new()
    foreach ($run in $Runs) {
        if (-not $run.PSObject.Properties['workflowName'] -or
            [string] $run.workflowName -cne 'Test impact map') { continue }
        foreach ($property in 'databaseId', 'conclusion', 'createdAt', 'status') {
            if (-not $run.PSObject.Properties[$property]) { throw "map run is missing $property" }
        }
        if ([string] $run.status -cne 'completed') { continue }
        $runId = 0L
        if (-not [long]::TryParse([string] $run.databaseId, [ref] $runId) -or $runId -lt 1) {
            throw 'map run databaseId must be a positive integer'
        }
        $createdAt = [DateTimeOffset]::MinValue
        if (-not [DateTimeOffset]::TryParse(
            [string] $run.createdAt,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::AssumeUniversal,
            [ref] $createdAt)) {
            throw "map run $runId has an invalid createdAt timestamp"
        }
        [void] $targetRuns.Add([pscustomobject]@{
            RunId = $runId
            Conclusion = [string] $run.conclusion
            CreatedAt = $createdAt
        })
    }

    :runLoop foreach ($candidate in @($targetRuns | Sort-Object CreatedAt, RunId -Descending)) {
        $runId = [long] $candidate.RunId
        # Runs are visited newest first, so once one is past the cap every remaining run is too:
        # nothing older may stand in for the run that failed.
        if ($skippedFailures.Count -gt 0 -and
            ($Now - $candidate.CreatedAt) -gt [TimeSpan]::FromDays($FailedRunFallbackMaxAgeDays)) {
            return New-Resolution ([CertifiedShardMapResolutionStatus]::FailedRunFallbackExpired) $skippedFailures[0] $candidate
        }
        $decisionObject = & $DownloadDecision $runId
        if ($null -ne $decisionObject) {
            try { $disposition = ConvertTo-AuditDecision $decisionObject $runId }
            catch {
                # A decision was uploaded and cannot be read. That is not the absence of evidence.
                return New-Resolution ([CertifiedShardMapResolutionStatus]::UnclassifiedFailure) $runId
            }
            switch ($disposition) {
                ([ShardMapAuditDisposition]::RevokedMiss) {
                    return New-Resolution ([CertifiedShardMapResolutionStatus]::RevokedByAuditMiss) $runId
                }
                ([ShardMapAuditDisposition]::SafeNoReduction) { continue runLoop }
                ([ShardMapAuditDisposition]::Certified) {
                    if ([bool] (& $DownloadCertificate $runId)) {
                        return New-Resolution ([CertifiedShardMapResolutionStatus]::Found) $runId $candidate
                    }
                    return New-Resolution ([CertifiedShardMapResolutionStatus]::CertificateUnavailable) $runId
                }
            }
        }

        # Backward compatibility for certificates produced before typed audit decisions shipped.
        # A historical success may contain a positive certificate or may simply be bootstrap/no
        # reduction. A cancellation contains no negative evidence.
        if ($candidate.Conclusion -ceq 'success') {
            if ([bool] (& $DownloadCertificate $runId)) {
                return New-Resolution ([CertifiedShardMapResolutionStatus]::Found) $runId $candidate
            }
            continue
        }
        if ($candidate.Conclusion -ceq 'cancelled') { continue }
        # Any other conclusion without a typed decision: the run broke before or outside the audit
        # (a miss is always written down before the audit step exits). It is stepped over, and the
        # age cap at the top of the loop bounds how far back the search may then reach. A cap of 0
        # keeps the original hard boundary.
        if ($FailedRunFallbackMaxAgeDays -le 0) {
            return New-Resolution ([CertifiedShardMapResolutionStatus]::UnclassifiedFailure) $runId
        }
        [void] $skippedFailures.Add($runId)
    }

    if ($skippedFailures.Count -gt 0) {
        # Nothing certified behind the failed run: the pre-existing fail-closed status, naming it.
        return New-Resolution ([CertifiedShardMapResolutionStatus]::UnclassifiedFailure) $skippedFailures[0]
    }
    return New-Resolution ([CertifiedShardMapResolutionStatus]::NoCertifiedRun) 0L
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $Message)
        if (-not $Condition) { [void] $failures.Add($Message) }
    }
    function New-Run {
        param([long] $Id, [string] $Conclusion, [string] $CreatedAt,
            [string] $WorkflowName = 'Test impact map', [string] $Status = 'completed')
        return [pscustomobject]@{
            databaseId = $Id; conclusion = $Conclusion; createdAt = $CreatedAt
            workflowName = $WorkflowName; status = $Status
        }
    }
    function New-Decision {
        param([long] $RunId, [string] $Disposition, [int] $MissCount = 0)
        return [pscustomobject]@{
            schemaVersion = 1; certificationRunId = $RunId
            disposition = $Disposition; missCount = $MissCount
        }
    }

    $certificateAttempts = [System.Collections.Generic.List[long]]::new()
    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 300 'cancelled' '2026-09-08T03:00:00Z'),
        (New-Run 200 'success' '2026-09-08T02:00:00Z'),
        (New-Run 100 'success' '2026-09-08T01:00:00Z')
    ) -DownloadDecision { param($RunId) $null } -DownloadCertificate {
        param($RunId); [void] $certificateAttempts.Add($RunId); return $RunId -eq 100
    }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found -and $result.RunId -eq 100) `
        'cancelled and successful no-certificate runs disabled an older positive certificate'
    Assert-True (($certificateAttempts -join ',') -ceq '200,100') `
        'historical fallback did not examine positive evidence in newest-first order'

    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 400 'failure' '2026-09-08T04:00:00Z'),
        (New-Run 100 'success' '2026-09-08T01:00:00Z')
    ) -DownloadDecision { param($RunId) if ($RunId -eq 400) { New-Decision 400 RevokedMiss 1 } } `
      -DownloadCertificate { param($RunId) return $true }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::RevokedByAuditMiss) `
        'an explicit selection miss did not revoke older evidence'

    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 400 'failure' '2026-09-08T04:00:00Z'),
        (New-Run 100 'success' '2026-09-08T01:00:00Z')
    ) -DownloadDecision { param($RunId) if ($RunId -eq 400) { New-Decision 400 SafeNoReduction } } `
      -DownloadCertificate { param($RunId) return $RunId -eq 100 }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found -and $result.RunId -eq 100) `
        'a typed zero-miss no-reduction audit revoked older evidence'

    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 400 'failure' '2026-09-08T04:00:00Z'),
        (New-Run 100 'success' '2026-09-08T01:00:00Z')
    ) -DownloadDecision { param($RunId) $null } -DownloadCertificate { param($RunId) return $true } `
      -FailedRunFallbackMaxAgeDays 0
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::UnclassifiedFailure -and $result.RunId -eq 400) `
        'a cap of 0 did not keep an unclassified failure as a hard boundary'

    # A failed run without a typed decision is neutral while the map behind it is within the cap.
    $now = [DateTimeOffset]::Parse('2026-09-22T04:00:00Z', [Globalization.CultureInfo]::InvariantCulture)
    $failedThenCertified = {
        param([string] $MapCreatedAt, [int] $Cap = 14, [scriptblock] $Decision = { param($RunId) $null },
            [string] $FailedConclusion = 'failure')
        $script:fallbackAttempts = [System.Collections.Generic.List[long]]::new()
        Resolve-CertifiedShardMapRun -Runs @(
            (New-Run 400 $FailedConclusion '2026-09-22T03:00:00Z'),
            (New-Run 100 'success' $MapCreatedAt)
        ) -DownloadDecision $Decision -DownloadCertificate {
            param($RunId); [void] $script:fallbackAttempts.Add($RunId); return $true
        } -FailedRunFallbackMaxAgeDays $Cap -Now $now
    }
    $result = & $failedThenCertified '2026-09-21T04:00:00Z'
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found -and $result.RunId -eq 100) `
        'a failed run without a typed decision blocked a one-day-old certified map'
    Assert-True ((@($result.SkippedFailedRunIds) -join ',') -ceq '400' -and
        [Math]::Abs([double] $result.MapAgeDays - 1.0) -lt 0.001) `
        'the fallback did not record the failed run it stepped over and the age of the map it used'
    foreach ($conclusion in 'timed_out', 'startup_failure') {
        $result = & $failedThenCertified '2026-09-21T04:00:00Z' 14 { param($RunId) $null } $conclusion
        Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found) `
            "a $conclusion run without a typed decision blocked a recent certified map"
    }
    # The boundary: exactly at the cap is usable, one second past it is not, and an expired map is
    # never downloaded.
    $result = & $failedThenCertified '2026-09-08T04:00:00Z'
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found -and $result.RunId -eq 100) `
        'a certified map exactly at the age cap was refused'
    $result = & $failedThenCertified '2026-09-08T03:59:59Z'
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::FailedRunFallbackExpired -and
        $result.RunId -eq 400 -and $script:fallbackAttempts.Count -eq 0) `
        'a certified map one second past the age cap was used after a failed run'
    $result = & $failedThenCertified '2026-09-19T04:00:00Z' 2
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::FailedRunFallbackExpired) `
        'the age cap is not the configured setting'
    $result = & $failedThenCertified '2026-09-19T04:00:00Z' 3
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found) `
        'raising the configured age cap did not admit the map'
    # Without a failed run in front of it, the cap does not apply: this is not a general expiry.
    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 300 'cancelled' '2026-09-22T03:00:00Z'),
        (New-Run 100 'success' '2026-08-01T00:00:00Z')
    ) -DownloadDecision { param($RunId) $null } -DownloadCertificate { param($RunId) return $true } -Now $now
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found -and
        @($result.SkippedFailedRunIds).Count -eq 0) `
        'the failed-run age cap was applied where no map run had failed'
    # No certified map behind the failure: fail closed, naming the failed run.
    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 400 'failure' '2026-09-22T03:00:00Z'),
        (New-Run 300 'failure' '2026-09-22T02:00:00Z')
    ) -DownloadDecision { param($RunId) $null } -DownloadCertificate { param($RunId) return $true } -Now $now
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::UnclassifiedFailure -and $result.RunId -eq 400) `
        'failed runs with no certified map behind them did not fail closed'
    # Deliberate revocation is untouched: behind a neutral failure, at the head, and a decision
    # that was uploaded but cannot be read.
    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 400 'failure' '2026-09-22T03:00:00Z'),
        (New-Run 300 'failure' '2026-09-22T02:00:00Z'),
        (New-Run 100 'success' '2026-09-21T04:00:00Z')
    ) -DownloadDecision { param($RunId) if ($RunId -eq 300) { New-Decision 300 RevokedMiss 2 } } `
      -DownloadCertificate { param($RunId) return $true } -Now $now
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::RevokedByAuditMiss -and $result.RunId -eq 300) `
        'a failed run in front of an audited miss let an older certificate through'
    $result = & $failedThenCertified '2026-09-21T04:00:00Z' 14 { param($RunId) if ($RunId -eq 400) { New-Decision 400 RevokedMiss 1 } }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::RevokedByAuditMiss -and
        $script:fallbackAttempts.Count -eq 0) `
        'the failed-run fallback weakened an explicit revocation'
    $result = & $failedThenCertified '2026-09-21T04:00:00Z' 14 { param($RunId) if ($RunId -eq 400) { [pscustomobject]@{ malformed = $true } } }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::UnclassifiedFailure -and
        $script:fallbackAttempts.Count -eq 0) `
        'an unreadable typed decision was treated as a neutral failure'
    $result = & $failedThenCertified '2026-09-21T04:00:00Z' 14 { param($RunId) if ($RunId -eq 100) { New-Decision 100 Certified } }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found -and $result.RunId -eq 100) `
        'a typed certified map behind a failed run was not used'

    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 500 'success' '2026-09-08T05:00:00Z')
    ) -DownloadDecision { param($RunId) New-Decision 500 Certified } `
      -DownloadCertificate { param($RunId) return $false }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::CertificateUnavailable) `
        'a typed Certified decision was accepted without its certificate artifact'

    if ($failures.Count -gt 0) {
        Write-Host 'Resolve-CertifiedShardMap self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'Resolve-CertifiedShardMap self-test passed.'
    exit 0
}

if (Test-Path -LiteralPath $OutDirectory) {
    throw "refusing to mix a certified map into existing directory $OutDirectory"
}
if (Test-Path -LiteralPath $ResultFile) {
    throw "refusing to overwrite existing result file $ResultFile"
}

# The map workflow runs on every master push, so the window is counted in merges, not days. A run
# that finds no coverage source to harvest completes green and publishes nothing; at 20, that many
# merges after the source expired would have scrolled the last certified map out of sight and put
# every pull request on the full matrix while its artifact was still retained. 60 is about a week
# of merges. Only runs newer than the first certified one are ever inspected.
$runsJson = & gh run list --repo $Repository --workflow test-impact-map.yml --branch $MapBranch `
    --status completed --limit 60 --json databaseId,conclusion,createdAt,workflowName,status 2>$null
if ($LASTEXITCODE -ne 0) { throw 'could not query completed Test impact map workflow runs' }
$runs = @($runsJson | ConvertFrom-Json)
$decisionRoot = Join-Path ([IO.Path]::GetTempPath()) ("aidotnet-map-decisions-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $decisionRoot -Force | Out-Null
try {
    $resolution = Resolve-CertifiedShardMapRun -Runs $runs -FailedRunFallbackMaxAgeDays $FailedRunFallbackMaxAgeDays -DownloadCertificate {
        param([long] $RunId)
        & gh run download $RunId --repo $Repository --name certified-shard-map `
            --dir $OutDirectory *> $null
        return $LASTEXITCODE -eq 0
    } -DownloadDecision {
        param([long] $RunId)
        $directory = Join-Path $decisionRoot ([string] $RunId)
        & gh run download $RunId --repo $Repository --name shard-map-audit-decision `
            --dir $directory *> $null
        if ($LASTEXITCODE -ne 0) { return $null }
        $file = @(Get-ChildItem -LiteralPath $directory -Filter map-audit-decision.json -File -Recurse |
            Select-Object -First 1)
        if ($file.Count -eq 0) { return $null }
        try { return (Get-Content -LiteralPath $file[0].FullName -Raw | ConvertFrom-Json) }
        catch { return [pscustomobject]@{ malformed = $true } }
    }
}
finally {
    $resolvedDecisionRoot = [IO.Path]::GetFullPath($decisionRoot)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($resolvedDecisionRoot.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedDecisionRoot).StartsWith('aidotnet-map-decisions-', [StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedDecisionRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$found = $resolution.Status -eq [CertifiedShardMapResolutionStatus]::Found
[ordered]@{
    found = $found
    runId = [long] $resolution.RunId
    status = $resolution.Status.ToString()
    # Report-only (Write-SelectionReport.ps1): which map this is, how old, and whether it stands in
    # for newer map runs that failed without a typed audit decision.
    mapCreatedAt = $(if ($found) { $resolution.MapCreatedAt.UtcDateTime.ToString('yyyy-MM-ddTHH:mm:ssZ', [Globalization.CultureInfo]::InvariantCulture) } else { $null })
    mapAgeDays = $(if ($found) { [Math]::Round([double] $resolution.MapAgeDays, 2) } else { $null })
    skippedFailedRunIds = @($resolution.SkippedFailedRunIds)
    failedRunFallbackMaxAgeDays = $FailedRunFallbackMaxAgeDays
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $ResultFile -Encoding utf8

switch ($resolution.Status) {
    ([CertifiedShardMapResolutionStatus]::Found) {
        Write-Host "using certified shard map from run $($resolution.RunId)"
        if (@($resolution.SkippedFailedRunIds).Count -gt 0) {
            Write-Host ("::warning::newer map run(s) $(@($resolution.SkippedFailedRunIds) -join ', ') failed without a typed audit decision; " +
                "using the last certified map from run $($resolution.RunId), $([Math]::Round([double] $resolution.MapAgeDays, 1)) day(s) old " +
                "(allowed up to $FailedRunFallbackMaxAgeDays)")
        }
    }
    ([CertifiedShardMapResolutionStatus]::NoCertifiedRun) {
        Write-Host '::warning::no usable certified shard map exists; selection will fail closed'
    }
    ([CertifiedShardMapResolutionStatus]::RevokedByAuditMiss) {
        Write-Host "::warning::map audit run $($resolution.RunId) observed a selection miss; older evidence is revoked"
    }
    ([CertifiedShardMapResolutionStatus]::UnclassifiedFailure) {
        Write-Host "::warning::map run $($resolution.RunId) failed without a readable typed audit decision and no certified map stands behind it; selection will fail closed"
    }
    ([CertifiedShardMapResolutionStatus]::FailedRunFallbackExpired) {
        Write-Host "::warning::map run $($resolution.RunId) failed without a typed audit decision and the last certified map is older than $FailedRunFallbackMaxAgeDays day(s); selection will fail closed"
    }
    ([CertifiedShardMapResolutionStatus]::CertificateUnavailable) {
        Write-Host "::warning::map run $($resolution.RunId) claims certification but its certificate is unavailable"
    }
}
exit 0
