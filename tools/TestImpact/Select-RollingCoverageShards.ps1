<#
.SYNOPSIS
    Which heavy shards tonight's scheduled full run instruments for the rolling map.

.DESCRIPTION
    Only called when repository variable TEST_IMPACT_ROLLING_MAP is shadow or on, and only on the scheduled
    nightly. The order, the exclusions and the budget are Select-RollingCoverageShards in RollingShardMap.ps1;
    this script gathers what that pure function needs:

      * each shard's measured commit, from the rolling map built on the anchor (the anchor's own commit when
        there is no rolling map yet), ranked by how far it is ahead of the anchor;
      * which indexed shards are defined differently now than at their measured commit;
      * which shards the selector picks for everything that changed since they were measured. The shipping
        selector is run for that, against the same map, so "stale" means exactly what selection means by it.

    Any failure here must leave the nightly as it was: the caller instruments nothing extra when this throws.

.PARAMETER ShardManifestFile
    The current shard manifest as JSON (the select job's shard-manifest.json).

.PARAMETER AnchorMapFile
    The certified weekly map in effect.

.PARAMETER RollingMapFile
    Optional. The rolling map built on that anchor, for per-shard measured commits.

.PARAMETER ShardScriptPath
    .github/scripts/Get-ShardCoveragePolicy.ps1, the canonical heavy and timing shard policy.

.PARAMETER BudgetMinutes
    Predicted extra runner-minutes (repository variable TEST_IMPACT_ROLLING_BUDGET_MINUTES). Empty is the
    default, RollingBudgetMinutes in RollingShardMap.ps1.

.PARAMETER OutFile
    The plan: instrumented shards in order, the prediction, every candidate and every exclusion. Uploaded as
    the run's test-impact-rolling-source marker, which is what makes the run a rolling-map source.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $ShardManifestFile,
    [Parameter(Mandatory)] [string] $AnchorMapFile,
    [string] $RollingMapFile,
    [string] $ShardScriptPath = "$PSScriptRoot/../../.github/scripts/Get-ShardCoveragePolicy.ps1",
    [string] $BudgetMinutes,
    [string] $SelectorPath = "$PSScriptRoot/Select-Shards.ps1",
    [Parameter(Mandatory)] [string] $OutFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RollingShardMap.ps1"
. "$PSScriptRoot/CiWorkloadKinds.ps1"

$manifest = @(Get-Content -LiteralPath $ShardManifestFile -Raw | ConvertFrom-Json)
if ($manifest.Count -eq 0) { throw 'the shard manifest is empty' }
$anchor = Get-Content -LiteralPath $AnchorMapFile -Raw | ConvertFrom-Json
$anchorSha = [string] $anchor.sha
if (-not (Test-RollingObjectId $anchorSha)) { throw 'the anchor map sha is not a full lowercase commit id' }
& git cat-file -e "$anchorSha^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) { throw "the anchor's commit $anchorSha is not present in this checkout" }

# Only the shard job reads COVERAGE_ROLLING; a workload that runs elsewhere cannot be instrumented this way.
$testShards = @((Split-CiWorkloads -Shards $manifest).Tests)
$lists = Get-NoCoverageShardLists -ScriptLines @(Get-Content -LiteralPath $ShardScriptPath) -Manifest $manifest

# Each shard's measured commit. Without a usable rolling map every indexed shard is at the anchor's.
$measuredSha = @{}
foreach ($name in @($anchor.knownShards)) { $measuredSha[[string] $name] = $anchorSha }
$mapForStaleness = $AnchorMapFile
if ($RollingMapFile -and (Test-Path -LiteralPath $RollingMapFile)) {
    try {
        $rolling = Get-Content -LiteralPath $RollingMapFile -Raw | ConvertFrom-Json
        Assert-RollingShardMap -Map $rolling
        if ([string] $rolling.anchor.sha -cne $anchorSha) { throw 'it was built on another anchor' }
        foreach ($layer in @($rolling.layers)) {
            & git cat-file -e "$($layer.sha)^{commit}" 2>$null
            if ($LASTEXITCODE -ne 0) { throw "its commit $($layer.sha) is not present in this checkout" }
        }
        foreach ($provenance in $rolling.shards.PSObject.Properties) { $measuredSha[$provenance.Name] = [string] $provenance.Value.sha }
        $mapForStaleness = $RollingMapFile
    }
    catch {
        Write-Host "the rolling map is not usable for ordering ($($_.Exception.Message)); ordering from the anchor alone"
        $measuredSha = @{}
        foreach ($name in @($anchor.knownShards)) { $measuredSha[[string] $name] = $anchorSha }
    }
}

# Rank by commits ahead of the anchor: 0 is the anchor's own commit, larger is newer.
$rank = @{ $anchorSha = 0L }
foreach ($sha in @($measuredSha.Values | Sort-Object -Unique)) {
    if ($rank.ContainsKey($sha)) { continue }
    $count = & git rev-list --count "$anchorSha..$sha" 2>$null
    $rank[$sha] = if ($LASTEXITCODE -eq 0) { [long] $count } else { 0L }
}
$measuredOrder = @{}
foreach ($name in $measuredSha.Keys) { $measuredOrder[$name] = $rank[$measuredSha[$name]] }

# Redefined: the definition at the shard's measured commit against the definition now.
$current = @{}
foreach ($entry in $manifest) { $current[[string] $entry.name] = Get-ShardDefinitionHash $entry }
$redefined = [System.Collections.Generic.List[string]]::new()
foreach ($sha in $rank.Keys) {
    $source = @(& git show "${sha}:.github/test-shards.yml" 2>$null)
    if ($LASTEXITCODE -ne 0) { continue }
    $json = ($source -join "`n") | & yq -o=json -I=0 '.shard' '-'
    if ($LASTEXITCODE -ne 0) { throw "cannot parse .github/test-shards.yml at $sha" }
    $then = @{}
    foreach ($entry in @(($json -join "`n") | ConvertFrom-Json)) { $then[[string] $entry.name] = Get-ShardDefinitionHash $entry }
    foreach ($name in @($measuredSha.Keys | Where-Object { $measuredSha[$_] -ceq $sha })) {
        if ($current.ContainsKey($name) -and (-not $then.ContainsKey($name) -or $then[$name] -cne $current[$name])) { [void] $redefined.Add($name) }
    }
}

# Stale: what the selector picks for everything since each shard's own measured commit. BaseSha is HEAD, so
# nothing counts as "this change" and merged selection-control edits are history, as in the nightly audit.
$stale = @()
$staleNote = ''
$selectionFile = Join-Path ([IO.Path]::GetTempPath()) "rolling-staleness-$PID.json"
try {
    $head = [string] (& git rev-parse HEAD)
    & $SelectorPath -MapFile $mapForStaleness -BaseSha $head -ShardManifestFile $ShardManifestFile `
        -ExpectedShards @($manifest.name) -OutFile $selectionFile | Out-Null
    $selection = Get-Content -LiteralPath $selectionFile -Raw | ConvertFrom-Json
    if ([bool] $selection.escalate) {
        $staleNote = "the selector escalates for the changes since the map ($([string] $selection.reason)), so every shard is equally out of date and age decides"
    }
    else { $stale = @($selection.shards | ForEach-Object { [string] $_ }) }
}
catch { $staleNote = "staleness could not be established ($($_.Exception.Message)); age decides" }
finally { Remove-Item -LiteralPath $selectionFile -ErrorAction SilentlyContinue }

$plan = Select-RollingCoverageShards -Manifest $testShards -NoCoverageShards $lists.NoCoverage -TimingShards $lists.Timing `
    -AnchorMap $anchor -MeasuredOrder $measuredOrder -RedefinedShards @($redefined) -StaleShards $stale `
    -BudgetMinutes (Resolve-RollingBudgetMinutes $BudgetMinutes)

$marker = [ordered]@{
    schemaVersion         = 1
    kind                  = 'rolling-source'
    runId                 = [string] $env:GITHUB_RUN_ID
    runAttempt            = [string] $env:GITHUB_RUN_ATTEMPT
    sha                   = [string] $env:GITHUB_SHA
    anchorSha             = $anchorSha
    budgetMinutes         = $plan.BudgetMinutes
    predictedExtraMinutes = $plan.PredictedExtraMinutes
    instrument            = @($plan.Instrument)
    staleness             = $staleNote
    candidates            = @($plan.Candidates)
    excluded              = @($plan.Excluded)
}
$marker | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutFile -Encoding utf8

$byClass = @($plan.Candidates | Where-Object taken | Group-Object class | Sort-Object Name | ForEach-Object { "class $($_.Name): $($_.Count)" })
Write-Host "rolling map refresh: instrumenting $($plan.Instrument.Count) of $(@($plan.Candidates).Count) eligible heavy shard(s) tonight, predicted $($plan.PredictedExtraMinutes) of $($plan.BudgetMinutes) extra runner-minute(s) ($($byClass -join ', '))"
if ($staleNote) { Write-Host "  $staleNote" }
foreach ($name in $plan.Instrument) { Write-Host "  $name" }
exit 0
