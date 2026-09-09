<#
.SYNOPSIS
    Downloads the certified shard map, if and only if the newest completed map run certified one.

.DESCRIPTION
    The newest completed Test impact map run is a revocation boundary. A failed run or a successful
    run without a certified-shard-map artifact invalidates every older certificate. Keeping that
    policy here, rather than embedded in workflow YAML, lets adversarial fixtures execute the exact
    decision code and record every attempted artifact download.
#>
[CmdletBinding(DefaultParameterSetName = 'Resolve')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $Repository,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $OutDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Resolve')] [string] $ResultFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum CertifiedShardMapResolutionStatus {
    Found
    NoCompletedRun
    LatestRunFailed
    CertificateUnavailable
}

function Resolve-CertifiedShardMapRun {
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Runs,
        [Parameter(Mandatory)] [scriptblock] $DownloadCertificate
    )

    $targetRuns = [System.Collections.Generic.List[object]]::new()
    foreach ($run in $Runs) {
        if (-not $run.PSObject.Properties['workflowName'] -or
            [string] $run.workflowName -cne 'Test impact map') {
            continue
        }
        foreach ($property in 'databaseId', 'conclusion', 'createdAt', 'status') {
            if (-not $run.PSObject.Properties[$property]) {
                throw "map run is missing $property"
            }
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

    $latest = @($targetRuns | Sort-Object CreatedAt, RunId -Descending | Select-Object -First 1)
    if ($latest.Count -eq 0) {
        return [pscustomobject]@{
            Status = [CertifiedShardMapResolutionStatus]::NoCompletedRun
            RunId = 0L
        }
    }

    $candidate = $latest[0]
    if ($candidate.Conclusion -cne 'success') {
        return [pscustomobject]@{
            Status = [CertifiedShardMapResolutionStatus]::LatestRunFailed
            RunId = [long] $candidate.RunId
        }
    }

    $downloaded = [bool] (& $DownloadCertificate ([long] $candidate.RunId))
    if (-not $downloaded) {
        return [pscustomobject]@{
            Status = [CertifiedShardMapResolutionStatus]::CertificateUnavailable
            RunId = [long] $candidate.RunId
        }
    }

    return [pscustomobject]@{
        Status = [CertifiedShardMapResolutionStatus]::Found
        RunId = [long] $candidate.RunId
    }
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True {
        param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) }
    }
    function New-Run {
        param(
            [long] $Id,
            [string] $Conclusion,
            [string] $CreatedAt,
            [string] $WorkflowName = 'Test impact map',
            [string] $Status = 'completed'
        )
        return [pscustomobject]@{
            databaseId = $Id
            conclusion = $Conclusion
            createdAt = $CreatedAt
            workflowName = $WorkflowName
            status = $Status
        }
    }

    $runs = @(
        (New-Run 100 'success' '2026-09-08T01:00:00Z'),
        (New-Run 300 'success' '2026-09-08T03:00:00Z' 'Build & SonarCloud'),
        (New-Run 200 'failure' '2026-09-08T02:00:00Z')
    )
    $attempts = [System.Collections.Generic.List[long]]::new()
    $result = Resolve-CertifiedShardMapRun -Runs $runs -DownloadCertificate {
        param([long] $RunId)
        [void] $attempts.Add($RunId)
        return $true
    }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::LatestRunFailed) `
        'an older certificate survived a newer failed map run'
    Assert-True ($attempts.Count -eq 0) `
        'the resolver attempted a certificate download after the newest map run failed'

    $runs = @(
        (New-Run 100 'success' '2026-09-08T01:00:00Z'),
        (New-Run 201 'success' '2026-09-08T02:00:00Z')
    )
    $attempts = [System.Collections.Generic.List[long]]::new()
    $result = Resolve-CertifiedShardMapRun -Runs $runs -DownloadCertificate {
        param([long] $RunId)
        [void] $attempts.Add($RunId)
        return $false
    }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::CertificateUnavailable) `
        'a successful map run without a certificate did not revoke older evidence'
    Assert-True (($attempts -join ',') -ceq '201') `
        'the resolver probed an older certificate after the newest artifact was unavailable'

    $attempts = [System.Collections.Generic.List[long]]::new()
    $result = Resolve-CertifiedShardMapRun -Runs $runs -DownloadCertificate {
        param([long] $RunId)
        [void] $attempts.Add($RunId)
        return $RunId -eq 201
    }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::Found -and
        $result.RunId -eq 201) 'the newest certified map was not selected'
    Assert-True (($attempts -join ',') -ceq '201') `
        'selecting a valid newest certificate attempted more than one download'

    $attempts = [System.Collections.Generic.List[long]]::new()
    $result = Resolve-CertifiedShardMapRun -Runs @(
        (New-Run 400 'success' '2026-09-08T04:00:00Z' 'Build & SonarCloud')
    ) -DownloadCertificate {
        param([long] $RunId)
        [void] $attempts.Add($RunId)
        return $true
    }
    Assert-True ($result.Status -eq [CertifiedShardMapResolutionStatus]::NoCompletedRun) `
        'an unrelated workflow was accepted as a map workflow'
    Assert-True ($attempts.Count -eq 0) 'an unrelated workflow triggered an artifact download'

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

$runsJson = & gh run list --repo $Repository --workflow test-impact-map.yml --branch master `
    --status completed --limit 20 --json databaseId,conclusion,createdAt,workflowName,status 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'could not query completed Test impact map workflow runs'
}
$runs = @($runsJson | ConvertFrom-Json)
$resolution = Resolve-CertifiedShardMapRun -Runs $runs -DownloadCertificate {
    param([long] $RunId)
    & gh run download $RunId --repo $Repository --name certified-shard-map `
        --dir $OutDirectory *> $null
    return $LASTEXITCODE -eq 0
}

$found = $resolution.Status -eq [CertifiedShardMapResolutionStatus]::Found
[ordered]@{
    found = $found
    runId = [long] $resolution.RunId
    status = $resolution.Status.ToString()
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $ResultFile -Encoding utf8

switch ($resolution.Status) {
    ([CertifiedShardMapResolutionStatus]::Found) {
        Write-Host "using certified shard map from latest completed map run $($resolution.RunId)"
    }
    ([CertifiedShardMapResolutionStatus]::NoCompletedRun) {
        Write-Host '::warning::no completed Test impact map workflow exists; selection will fail closed'
    }
    ([CertifiedShardMapResolutionStatus]::LatestRunFailed) {
        Write-Host "::warning::latest completed map run $($resolution.RunId) failed; refusing to fall back to an older map"
    }
    ([CertifiedShardMapResolutionStatus]::CertificateUnavailable) {
        Write-Host "::warning::latest completed map run $($resolution.RunId) has no usable certificate; refusing to fall back to an older map"
    }
}
exit 0
