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
#>
[CmdletBinding(DefaultParameterSetName = 'Resolve')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $Repository,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $OutDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $ResultFile,
    [Parameter(ParameterSetName = 'Resolve')] [string] $MapBranch = 'master',
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
        [Parameter(Mandatory)] [scriptblock] $DownloadDecision
    )

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
        $decisionObject = & $DownloadDecision $runId
        if ($null -ne $decisionObject) {
            try { $disposition = ConvertTo-AuditDecision $decisionObject $runId }
            catch {
                return [pscustomobject]@{
                    Status = [CertifiedShardMapResolutionStatus]::UnclassifiedFailure
                    RunId = $runId
                }
            }
            switch ($disposition) {
                ([ShardMapAuditDisposition]::RevokedMiss) {
                    return [pscustomobject]@{
                        Status = [CertifiedShardMapResolutionStatus]::RevokedByAuditMiss
                        RunId = $runId
                    }
                }
                ([ShardMapAuditDisposition]::SafeNoReduction) { continue runLoop }
                ([ShardMapAuditDisposition]::Certified) {
                    if ([bool] (& $DownloadCertificate $runId)) {
                        return [pscustomobject]@{
                            Status = [CertifiedShardMapResolutionStatus]::Found
                            RunId = $runId
                        }
                    }
                    return [pscustomobject]@{
                        Status = [CertifiedShardMapResolutionStatus]::CertificateUnavailable
                        RunId = $runId
                    }
                }
            }
        }

        # Backward compatibility for certificates produced before typed audit decisions shipped.
        # A historical success may contain a positive certificate or may simply be bootstrap/no
        # reduction. A cancellation contains no negative evidence. An unclassified failure remains
        # a hard boundary because an old miss would otherwise be silently ignored.
        if ($candidate.Conclusion -ceq 'success') {
            if ([bool] (& $DownloadCertificate $runId)) {
                return [pscustomobject]@{
                    Status = [CertifiedShardMapResolutionStatus]::Found
                    RunId = $runId
                }
            }
            continue
        }
        if ($candidate.Conclusion -ceq 'cancelled') { continue }
        return [pscustomobject]@{
            Status = [CertifiedShardMapResolutionStatus]::UnclassifiedFailure
            RunId = $runId
        }
    }

    return [pscustomobject]@{
        Status = [CertifiedShardMapResolutionStatus]::NoCertifiedRun
        RunId = 0L
    }
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
    ) -DownloadDecision { param($RunId) $null } -DownloadCertificate { param($RunId) return $true }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::UnclassifiedFailure) `
        'an unclassified historical failure did not fail closed'

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

$runsJson = & gh run list --repo $Repository --workflow test-impact-map.yml --branch $MapBranch `
    --status completed --limit 20 --json databaseId,conclusion,createdAt,workflowName,status 2>$null
if ($LASTEXITCODE -ne 0) { throw 'could not query completed Test impact map workflow runs' }
$runs = @($runsJson | ConvertFrom-Json)
$decisionRoot = Join-Path ([IO.Path]::GetTempPath()) ("aidotnet-map-decisions-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $decisionRoot -Force | Out-Null
try {
    $resolution = Resolve-CertifiedShardMapRun -Runs $runs -DownloadCertificate {
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
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $ResultFile -Encoding utf8

switch ($resolution.Status) {
    ([CertifiedShardMapResolutionStatus]::Found) {
        Write-Host "using certified shard map from run $($resolution.RunId)"
    }
    ([CertifiedShardMapResolutionStatus]::NoCertifiedRun) {
        Write-Host '::warning::no usable certified shard map exists; selection will fail closed'
    }
    ([CertifiedShardMapResolutionStatus]::RevokedByAuditMiss) {
        Write-Host "::warning::map audit run $($resolution.RunId) observed a selection miss; older evidence is revoked"
    }
    ([CertifiedShardMapResolutionStatus]::UnclassifiedFailure) {
        Write-Host "::warning::map run $($resolution.RunId) failed without a typed audit decision; older evidence remains blocked"
    }
    ([CertifiedShardMapResolutionStatus]::CertificateUnavailable) {
        Write-Host "::warning::map run $($resolution.RunId) claims certification but its certificate is unavailable"
    }
}
exit 0
