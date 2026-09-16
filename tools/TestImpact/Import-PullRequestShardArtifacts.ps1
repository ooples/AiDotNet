<#
.SYNOPSIS
    Copies the pull request run's per-shard artifacts for the shards a partial post-merge run did
    not re-run, so the landed commit's ledger, analysis and coverage still cover every shard the
    pull request validated.

.DESCRIPTION
    Delta reuse (Resolve-CiValidationReuse.ps1) re-runs only the shards the base branch's later
    commits could have affected and takes the rest from the pull request's run. Every consumer that
    reads per-shard artifacts - the regression ledger, the aggregate analysis, Sonar's coverage -
    must see those imported shards too, or the landed commit would be recorded as having validated
    only the handful that re-ran.

    The staging directory holds the pull request run's artifacts downloaded one folder per
    artifact, named '<prefix>-<tested sha>-<slug>'. Exactly the listed shards are copied; a listed
    shard with no artifact is an error, because the evidence the reuse decision relied on is then
    missing and the landed commit cannot claim it.

.PARAMETER DestinationDirectory
    Each imported artifact keeps its own folder beneath this directory, exactly as a download with
    merge-multiple: false lays it out. Consumers that merge artifacts (Sonar) import into a
    subfolder their recursive glob already covers: every coverage artifact carries a root-level
    shard-metadata.json, so merging imports into one folder would collide by construction.
#>
[CmdletBinding(DefaultParameterSetName = 'Import')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Import')] [string] $StagingDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Import')] [string] $DestinationDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Import')] [ValidateSet('coverage', 'test-results')] [string] $ArtifactPrefix,
    [Parameter(Mandatory, ParameterSetName = 'Import')] [string] $ImportSha,
    [Parameter(Mandatory, ParameterSetName = 'Import')] [string] $ImportShardsJson,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-ShardSlug {
    # Identical to the test job's artifact naming.
    param([Parameter(Mandatory)] [string] $Name)
    return $Name -replace '[\\/:*?"<>|\s-]+', '_'
}

function Copy-ImportedShards {
    param(
        [Parameter(Mandatory)] [string] $Staging,
        [Parameter(Mandatory)] [string] $Destination,
        [Parameter(Mandatory)] [string] $Prefix,
        [Parameter(Mandatory)] [string] $Sha,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Shards
    )

    if ($Sha -notmatch '^[0-9a-f]{40}$') { throw "import sha '$Sha' is not a lowercase 40-character Git SHA" }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $missing = [System.Collections.Generic.List[string]]::new()
    $copied = [System.Collections.Generic.List[string]]::new()
    foreach ($shard in $Shards) {
        $folderName = "$Prefix-$Sha-$(ConvertTo-ShardSlug $shard)"
        $source = Join-Path $Staging $folderName
        if (-not (Test-Path -LiteralPath $source -PathType Container)) {
            [void] $missing.Add($shard)
            continue
        }
        $target = Join-Path $Destination $folderName
        if (Test-Path -LiteralPath $target) { throw "refusing to overwrite '$target' with an imported shard" }
        Copy-Item -LiteralPath $source -Destination $target -Recurse
        [void] $copied.Add($shard)
    }
    return [pscustomobject]@{ Copied = @($copied); Missing = @($missing) }
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $What) if (-not $Condition) { [void] $failures.Add($What) } }
    $sha = '0123456789abcdef0123456789abcdef01234567'
    $root = Join-Path ([System.IO.Path]::GetTempPath()) ("import-shards-" + [guid]::NewGuid().ToString('N'))
    try {
        foreach ($slug in @('Integration_E_G', 'Unit_10_RL', 'Integration_D')) {
            $folder = Join-Path $root "staging/coverage-$sha-$slug/$slug"
            New-Item -ItemType Directory -Path $folder -Force | Out-Null
            "<coverage $slug/>" | Set-Content -LiteralPath (Join-Path $folder 'coverage.opencover.xml')
        }
        $r = Copy-ImportedShards -Staging "$root/staging" -Destination "$root/per" -Prefix coverage -Sha $sha `
            -Shards @('Integration E-G', 'Unit - 10 RL')
        Assert-True ((@($r.Copied) -join ',') -eq 'Integration E-G,Unit - 10 RL') 'the listed shards were not all copied'
        Assert-True (Test-Path "$root/per/coverage-$sha-Integration_E_G/Integration_E_G/coverage.opencover.xml") `
            'the per-artifact layout lost the artifact folder'
        Assert-True (-not (Test-Path "$root/per/coverage-$sha-Integration_D")) `
            'a re-run shard was imported from the pull request, shadowing the fresh result'

        $r = Copy-ImportedShards -Staging "$root/staging" -Destination "$root/missing" -Prefix coverage -Sha $sha `
            -Shards @('Integration E-G', 'ModelFamily - Audio')
        Assert-True ((@($r.Missing) -join ',') -eq 'ModelFamily - Audio') 'a listed shard with no artifact was not reported missing'

        New-Item -ItemType Directory -Path "$root/clash/coverage-$sha-Integration_E_G" -Force | Out-Null
        $threw = $false
        try { Copy-ImportedShards -Staging "$root/staging" -Destination "$root/clash" -Prefix coverage -Sha $sha -Shards @('Integration E-G') | Out-Null }
        catch { $threw = $true }
        Assert-True $threw 'an import that would overwrite this run''s own output was allowed'
    }
    finally {
        Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($failures.Count -gt 0) {
        Write-Host 'Import-PullRequestShardArtifacts self-test FAILED:'
        foreach ($f in $failures) { Write-Host "  - $f" }
        exit 1
    }
    Write-Host 'Import-PullRequestShardArtifacts self-test passed.'
    exit 0
}

$shards = @($ImportShardsJson | ConvertFrom-Json | ForEach-Object { [string] $_ } | Where-Object { $_ })
$result = Copy-ImportedShards -Staging $StagingDirectory -Destination $DestinationDirectory -Prefix $ArtifactPrefix `
    -Sha $ImportSha -Shards $shards
Write-Host "imported $($result.Copied.Count) pull-request shard artifact(s) into $DestinationDirectory"
foreach ($shard in $result.Copied) { Write-Host "  $shard" }
if ($result.Missing.Count -gt 0) {
    throw "the pull request run has no $ArtifactPrefix artifact for: $($result.Missing -join ', '). Reuse relied on these results, so their absence cannot be papered over."
}
exit 0
