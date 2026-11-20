<#
.SYNOPSIS
    Builds, audits and certifies the rolling map on top of the map this run just certified.

.DESCRIPTION
    Runs in the map workflow, after "Upload certified map", only when repository variable
    TEST_IMPACT_ROLLING_MAP is shadow or on. The rules are in RollingShardMap.ps1; this script supplies them
    with facts:

      1. SOURCES. Scheduled runs of the test workflow on the map branch that uploaded a
         test-impact-rolling-source marker (so they ran with the flag on), are no older than the digest age
         limit, and whose commit descends from the anchor's. For each: every shard job's conclusion, the
         digests it uploaded, and the shard manifest committed at its commit. A run's own conclusion is not
         consulted: the nightly is usually "cancelled" or "failure" for reasons unrelated to any one shard,
         and rule (a) is about the shard job that produced each digest.
      2. BUILD. New-RollingShardMapObject: the newest valid digest per shard over the anchor.
      3. AUDIT. Measure-SelectionMiss.ps1 replays the rolling map with the shipping selector at the newest
         source whose shard jobs ALL finished as success or failure, in a temporary worktree of that commit,
         and intersects what it would skip with what failed there.
      4. CERTIFY. Get-RollingCertificationDecision. The certificate is written only for a clean reducing
         audit; the decision and the map are always written, so a refused map can be inspected.

    Nothing here can affect the certified map: it is already uploaded, and this step is continue-on-error.

.PARAMETER SourcesFile
    Instead of asking GitHub: a JSON array of { runId, sha, createdAt, outcomes: [{ shard, outcome }],
    digestDirectory }. Test-TestImpactEndToEnd.ps1 drives the whole build through this.

.OUTPUTS
    In OutDirectory: rolling-shard-map.json, rolling-decision.json, and rolling-certification.json when
    certified. With no usable digest only rolling-decision.json is written.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $AnchorMapFile,
    [Parameter(Mandatory)] [string] $AnchorCertificateFile,
    [Parameter(Mandatory)] [long] $AnchorRunId,
    [Parameter(Mandatory)] [string] $OutDirectory,
    [string] $Repository,
    [string] $SourcesFile,
    [string] $WorkflowFile = 'sonarcloud.yml',
    [string] $Branch = 'master',
    [string] $WorkDirectory,
    [int] $MaxDigestAgeDays = 0,
    [string] $Now
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RollingShardMap.ps1"
if ($MaxDigestAgeDays -le 0) { $MaxDigestAgeDays = $script:RollingMaxDigestAgeDays }
$clock = if ($Now) { [DateTimeOffset]::Parse($Now, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::AssumeUniversal) }
         else { [DateTimeOffset]::UtcNow }
if (-not $WorkDirectory) { $WorkDirectory = Join-Path ([IO.Path]::GetTempPath()) "rolling-map-$PID" }
New-Item -ItemType Directory -Path $WorkDirectory, $OutDirectory -Force | Out-Null
$WorkDirectory = (Resolve-Path -LiteralPath $WorkDirectory).Path
$OutDirectory = (Resolve-Path -LiteralPath $OutDirectory).Path

function Get-ShardOutcomes {
    # 'Tests (<framework>) - <shard>' -> conclusion, as the map workflow's own audit reads them.
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Jobs)
    $outcomes = [ordered]@{}
    foreach ($job in $Jobs) {
        $name = [string] $job.name
        if (-not $name.StartsWith('Tests (', [StringComparison]::Ordinal)) { continue }
        $split = $name.IndexOf(') - ', [StringComparison]::Ordinal)
        if ($split -lt 0) { throw "unrecognised shard job name: $name" }
        $outcomes[$name.Substring($split + 4)] = [string] $job.conclusion
    }
    return $outcomes
}

function Get-RollingSourceRecords {
    # The harvest: one record per scheduled run that took part, oldest first.
    $since = $clock.AddDays(-$MaxDigestAgeDays).UtcDateTime.ToString('yyyy-MM-ddTHH:mm:ssZ', [Globalization.CultureInfo]::InvariantCulture)
    $listing = gh api -X GET "repos/$Repository/actions/workflows/$WorkflowFile/runs" `
        -f event=schedule -f "branch=$Branch" -f status=completed -f per_page=50 -f "created=>=$since" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'listing scheduled runs failed' }
    $records = [System.Collections.Generic.List[object]]::new()
    foreach ($run in @($listing.workflow_runs | Sort-Object { [DateTimeOffset] $_.created_at })) {
        $id = [long] $run.id
        try {
            $artifactPages = gh api --paginate --slurp "repos/$Repository/actions/runs/$id/artifacts?per_page=100" | ConvertFrom-Json
            if ($LASTEXITCODE -ne 0) { throw 'listing its artifacts failed' }
            $artifacts = @($artifactPages | ForEach-Object { $_.artifacts } | Where-Object { -not $_.expired })
            if (@($artifacts | Where-Object { $_.name -ceq 'test-impact-rolling-source' }).Count -eq 0) {
                Write-Host "run $id has no rolling-source marker; it did not take part"
                continue
            }
            # 50 per page: per_page=100 answers 502 on a re-run of this size (see the test-analysis job).
            $jobPages = gh api --paginate --slurp "repos/$Repository/actions/runs/$id/jobs?per_page=50" | ConvertFrom-Json
            if ($LASTEXITCODE -ne 0) { throw 'listing its jobs failed' }
            $outcomes = Get-ShardOutcomes -Jobs @($jobPages | ForEach-Object { $_.jobs })
            $digestDirectory = Join-Path $WorkDirectory "digests-$id"
            if (@($artifacts | Where-Object { ([string] $_.name).StartsWith('impact-digest-', [StringComparison]::Ordinal) }).Count -gt 0) {
                gh run download $id --repo $Repository --pattern 'impact-digest-*' --dir $digestDirectory | Out-Null
                if ($LASTEXITCODE -ne 0) { throw 'downloading its digests failed' }
            }
            [void] $records.Add([pscustomobject]@{
                runId = $id; sha = [string] $run.head_sha; createdAt = [string] $run.created_at
                outcomes = @($outcomes.Keys | ForEach-Object { [pscustomobject]@{ shard = $_; outcome = $outcomes[$_] } })
                digestDirectory = $digestDirectory
            })
        }
        catch { Write-Host "::warning::run $id is not used as a rolling source: $($_.Exception.Message)" }
    }
    return @($records)
}

$anchorMap = Get-Content -LiteralPath $AnchorMapFile -Raw | ConvertFrom-Json
$anchorCertificate = Get-Content -LiteralPath $AnchorCertificateFile -Raw | ConvertFrom-Json
$anchorSha = [string] $anchorMap.sha

$records = if ($SourcesFile) { @(Get-Content -LiteralPath $SourcesFile -Raw | ConvertFrom-Json) }
           elseif ($Repository) { @(Get-RollingSourceRecords) }
           else { throw 'pass Repository or SourcesFile' }

$decisionFile = Join-Path $OutDirectory 'rolling-decision.json'
function Write-RollingDecision([string] $Disposition, [string] $Why, $Detail = $null) {
    $decision = [ordered]@{
        schemaVersion = 1; kind = 'rolling'; disposition = $Disposition; reason = $Why
        anchorRunId = $AnchorRunId; anchorSha = $anchorSha; sources = @($sourceSummaries); rejected = @($rejectedSummaries)
    }
    if ($null -ne $Detail) { foreach ($property in $Detail.PSObject.Properties) { if (-not $decision.Contains($property.Name)) { $decision[$property.Name] = $property.Value } } }
    $decision | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $decisionFile -Encoding utf8
    Write-Host "rolling map: $Disposition - $Why"
}

# Facts per source: ancestry, the manifest committed at its commit, and its digests.
$sources = [System.Collections.Generic.List[object]]::new()
$sourceSummaries = [System.Collections.Generic.List[object]]::new()
$rejectedSummaries = [System.Collections.Generic.List[object]]::new()
foreach ($record in @($records | Sort-Object { [DateTimeOffset] $_.createdAt })) {
    $sha = [string] $record.sha
    & git cat-file -e "$sha^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        [void] $sourceSummaries.Add([ordered]@{ runId = $record.runId; sha = $sha; used = $false; why = 'its commit is not present in this checkout' })
        continue
    }
    & git merge-base --is-ancestor $anchorSha $sha 2>$null
    $descends = $LASTEXITCODE -eq 0
    $manifest = @()
    $manifestBlob = ''
    $yaml = @(& git show "${sha}:.github/test-shards.yml" 2>$null)
    if ($LASTEXITCODE -eq 0) {
        $json = ($yaml -join "`n") | & yq -o=json -I=0 '.shard' '-'
        if ($LASTEXITCODE -ne 0) { throw "cannot parse .github/test-shards.yml at $sha" }
        $manifest = @(($json -join "`n") | ConvertFrom-Json)
        $manifestBlob = [string] (& git rev-parse "${sha}:.github/test-shards.yml")
    }
    $outcomes = @{}
    foreach ($outcome in @($record.outcomes)) { $outcomes[[string] $outcome.shard] = [string] $outcome.outcome }
    $digests = [System.Collections.Generic.List[object]]::new()
    if ($record.digestDirectory -and (Test-Path -LiteralPath $record.digestDirectory)) {
        foreach ($file in @(Get-ChildItem -LiteralPath $record.digestDirectory -Filter '*.digest.json' -Recurse -File | Sort-Object FullName)) {
            try { [void] $digests.Add((Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json)) }
            catch { [void] $rejectedSummaries.Add([ordered]@{ shard = $file.Name; runId = $record.runId; reason = 'the digest file is not readable JSON' }) }
        }
    }
    [void] $sources.Add([pscustomobject]@{
        RunId = [long] $record.runId; Sha = $sha; CreatedAt = [DateTimeOffset]::Parse([string] $record.createdAt, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::AssumeUniversal)
        DescendsFromAnchor = $descends; Manifest = $manifest; ManifestBlob = $manifestBlob; Outcomes = $outcomes; Digests = @($digests)
    })
    $unfinished = @($outcomes.Values | Where-Object { $_ -cnotin @('success', 'failure') }).Count
    [void] $sourceSummaries.Add([ordered]@{
        runId = $record.runId; sha = $sha; used = $true; descendsFromAnchor = $descends; digests = $digests.Count
        shardJobs = $outcomes.Count; failed = @($outcomes.Values | Where-Object { $_ -ceq 'failure' }).Count; unfinished = $unfinished
    })
}

$built = New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId $AnchorRunId -Sources @($sources) -Now $clock -MaxDigestAgeDays $MaxDigestAgeDays
foreach ($entry in $built.Rejected) { [void] $rejectedSummaries.Add($entry) }
Write-Host "rolling map: $($sources.Count) source run(s), $($built.Refreshed.Count) shard(s) refreshed, $($rejectedSummaries.Count) digest(s) not used"
if ($built.Refreshed.Count -eq 0) {
    Write-RollingDecision 'NoRefresh' 'no nightly digest passed the rules, so there is nothing to add to the certified weekly map'
    exit 0
}

$mapFile = Join-Path $OutDirectory 'rolling-shard-map.json'
$built.Map | ConvertTo-Json -Depth 12 -Compress | Set-Content -LiteralPath $mapFile -Encoding utf8
$mapHash = Get-RollingFileHash -Path $mapFile
$rollingMap = Get-Content -LiteralPath $mapFile -Raw | ConvertFrom-Json

# (c) The newest source whose every shard job finished. A cancelled or skipped shard is not evidence.
$auditSource = @($sources | Where-Object {
    $_.DescendsFromAnchor -and $_.Outcomes.Count -ge 2 -and
    @($_.Outcomes.Values | Where-Object { $_ -cnotin @('success', 'failure') }).Count -eq 0 -and
    ($clock - $_.CreatedAt) -le [TimeSpan]::FromDays($MaxDigestAgeDays)
} | Sort-Object CreatedAt | Select-Object -Last 1)
if ($auditSource.Count -eq 0) {
    Write-RollingDecision 'NoCompleteNightly' 'no source nightly finished every shard job, so the rolling map cannot be replayed against complete outcomes' ([pscustomobject]@{ mapHash = $mapHash; refreshed = $built.Refreshed })
    exit 0
}
$auditSource = $auditSource[0]
$outcomesFile = Join-Path $WorkDirectory 'rolling-outcomes.json'
$outcomeList = @($auditSource.Outcomes.Keys | Sort-Object | ForEach-Object { [pscustomobject]@{ shard = $_; outcome = $auditSource.Outcomes[$_] } })
ConvertTo-Json -InputObject $outcomeList -Depth 3 | Set-Content -LiteralPath $outcomesFile -Encoding utf8
$auditManifest = Join-Path $WorkDirectory 'rolling-audit-manifest.json'
ConvertTo-Json -InputObject @($auditSource.Manifest) -Depth 5 | Set-Content -LiteralPath $auditManifest -Encoding utf8
$auditFile = Join-Path $WorkDirectory 'rolling-audit.json'
Remove-Item -LiteralPath $auditFile -ErrorAction SilentlyContinue

# A worktree of the audited commit, so the replay diffs against exactly the tree those outcomes belong to
# without moving this checkout. The tooling is this run's, as in the map's own audit.
$tree = Join-Path $WorkDirectory 'audit-tree'
& git worktree add --detach --force $tree $auditSource.Sha 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "could not check out the audited commit $($auditSource.Sha)" }
try {
    Push-Location $tree
    try {
        & "$PSScriptRoot/Measure-SelectionMiss.ps1" -SelectorPath "$PSScriptRoot/Select-Shards.ps1" `
            -MapFile $mapFile -OutcomesFile $outcomesFile -CurrentChangeBaseSha $auditSource.Sha `
            -ShardManifestFile $auditManifest -OutFile $auditFile
    }
    finally { Pop-Location }
}
finally {
    & git worktree remove --force $tree 2>&1 | Out-Null
}
if (-not (Test-Path -LiteralPath $auditFile)) { throw 'the rolling audit produced no report' }
$audit = Get-Content -LiteralPath $auditFile -Raw | ConvertFrom-Json

$certification = Get-RollingCertificationDecision -RollingMap $rollingMap -RollingMapHash $mapHash -Refreshed $built.Refreshed `
    -AnchorMap $anchorMap -AnchorCertificate $anchorCertificate -AnchorRunId $AnchorRunId -Audit $audit -Outcomes $outcomeList `
    -AuditSourceRunId $auditSource.RunId -AuditSourceSha $auditSource.Sha -CertificationRunId $AnchorRunId `
    -Now $clock -MaxDigestAgeDays $MaxDigestAgeDays
$why = switch ($certification.Disposition) {
    'Certified' { "replayed against run $($auditSource.RunId): would run $($audit.WouldRun) of $($audit.TotalShards), $($audit.Failed) failed, none of them skipped" }
    'RevokedMiss' { "replayed against run $($auditSource.RunId), it would have skipped failed shard(s): $(@($audit.Missed) -join ', ')" }
    default { "replayed against run $($auditSource.RunId), it did not reduce the matrix (escalated: $($audit.Escalated)), so it proves nothing and the weekly map decides" }
}
Write-RollingDecision $certification.Disposition $why ([pscustomobject] @{ certificate = $certification.Decision; refreshed = $built.Refreshed })
if ($certification.Eligible) {
    $certification.Certificate | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutDirectory 'rolling-certification.json') -Encoding utf8
}
exit 0
