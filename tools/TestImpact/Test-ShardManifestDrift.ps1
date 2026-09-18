<#
.SYNOPSIS
    Proves selection stays reduced and safe when the shard manifest changes after the map was measured.

.DESCRIPTION
    A real repository whose committed .github/test-shards.yml changes between the map's commit and a
    pull request: one mapped shard is redefined, one is retired and one is added. Master's shard list
    changed 116 -> 209 in three days (2026-09-15..17), and any such edit used to send every pull
    request to the full matrix until a new map was certified.

    Needs yq on PATH, as the selector does on the runner.
#>
[CmdletBinding()]
param([string] $SelectorPath = (Join-Path $PSScriptRoot 'Select-Shards.ps1'))

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'ReviewFixtureCleanup.ps1')

$missMeasurer = Join-Path $PSScriptRoot 'Measure-SelectionMiss.ps1'
$fixture = Join-Path ([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) ('aidotnet-manifest-drift-' + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
$fixtureFailure = $null

function Assert-True {
    param([bool] $Condition, [string] $What)
    if (-not $Condition) { [void] $failures.Add($What) }
}

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments)] [string[]] $Arguments)
    $output = & git @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "git $($Arguments -join ' ') failed: $(($output | Out-String).Trim())" }
    return $output
}

function Get-Head { return ((Invoke-Git rev-parse HEAD) | Out-String).Trim() }

function Write-Manifest {
    param([object[]] $Shards)
    New-Item -ItemType Directory -Path .github -Force | Out-Null
    $lines = @('shard:')
    foreach ($shard in $Shards) {
        $lines += "  - name: $($shard.name)"
        $lines += '    project: tests/Proj/Proj.csproj'
        $lines += "    filter: '$($shard.filter)'"
        if ($shard.ContainsKey('offset')) {
            $lines += '    env:'
            $lines += "      ADNSHAPE_CONF_OFFSET: '$($shard.offset)'"
        }
    }
    $lines | Set-Content -LiteralPath .github/test-shards.yml -Encoding utf8
}

function Invoke-Selection {
    param([string] $Name, [string] $PullRequestHeadSha, [string] $Map = 'shard-map.json')
    $entries = @(& yq -o=json -I=0 '.shard' .github/test-shards.yml | ConvertFrom-Json)
    ConvertTo-Json -InputObject $entries -Depth 5 | Set-Content -LiteralPath "$Name-manifest.json" -Encoding utf8
    $output = @(& $SelectorPath -MapFile $Map -PullRequestHeadSha $PullRequestHeadSha `
        -ShardManifestFile "$Name-manifest.json" -ExpectedShards @($entries.name) -OutFile "$Name.json" 6>&1)
    foreach ($line in $output) { Write-Host "[$Name] $line" }
    Assert-True ($LASTEXITCODE -eq 0) "$Name`: the selector exited $LASTEXITCODE"
    return Get-Content -LiteralPath "$Name.json" -Raw | ConvertFrom-Json
}

function New-PullRequest {
    <# A GitHub-shaped refs/pull/N/merge: base tip first, pull request head second. #>
    param([string] $Base, [string] $Line, [string] $Label)
    Invoke-Git checkout --quiet --detach $Base
    $content = @(Get-Content -LiteralPath src/Feature.cs)
    $content[$Line - 1] = $content[$Line - 1] -replace 'return \d+', 'return 7'
    $content | Set-Content -LiteralPath src/Feature.cs -Encoding utf8
    Invoke-Git commit --quiet -am $Label
    $head = Get-Head
    Invoke-Git checkout --quiet --detach $Base
    Invoke-Git merge --quiet --no-ff --no-edit $head
    return $head
}

$measured = @(
    @{ name = 'Alpha'; filter = 'FullyQualifiedName~UnitTests.Alpha' },
    @{ name = 'Beta'; filter = 'FullyQualifiedName~UnitTests.Beta' },
    @{ name = 'Gamma'; filter = 'FullyQualifiedName~UnitTests.Gamma' },
    @{ name = 'Epsilon'; filter = 'FullyQualifiedName~UnitTests.Epsilon' },
    @{ name = 'Window'; filter = 'FullyQualifiedName~ModelContractConformanceTests'; offset = 5 },
    @{ name = 'Always'; filter = 'Category=Heavy' },
    @{ name = 'Dropped'; filter = 'Category=Legacy' }
)
$current = @(
    @{ name = 'Alpha'; filter = 'FullyQualifiedName~UnitTests.Alpha' },
    @{ name = 'Beta'; filter = 'FullyQualifiedName~UnitTests.Beta|FullyQualifiedName~UnitTests.Gamma' },
    @{ name = 'Epsilon'; filter = 'FullyQualifiedName~UnitTests.Epsilon' },
    @{ name = 'Window'; filter = 'FullyQualifiedName~ModelContractConformanceTests'; offset = 5 },
    @{ name = 'Always'; filter = 'Category=Heavy' },
    @{ name = 'Delta'; filter = 'FullyQualifiedName~UnitTests.Delta' }
)

try {
    New-Item -ItemType Directory -Path (Join-Path $fixture 'repo/src') -Force | Out-Null
    Push-Location (Join-Path $fixture 'repo')
    try {
        Invoke-Git init --quiet
        Invoke-Git config user.email 'ci-impact-fixture@example.invalid'
        Invoke-Git config user.name 'CI impact fixture'
        Invoke-Git config commit.gpgSign false
        $hooks = Join-Path $fixture 'disabled-hooks'
        New-Item -ItemType Directory -Path $hooks -Force | Out-Null
        Invoke-Git config core.hooksPath $hooks

        @('class Feature {', 'int Alpha() { return 1; }', '', 'int Beta() { return 1; }', '',
          'int Gamma() { return 1; }', '', 'int Epsilon() { return 1; }', '}') | Set-Content -LiteralPath src/Feature.cs -Encoding utf8
        Write-Manifest $measured
        Invoke-Git add src/Feature.cs .github/test-shards.yml
        Invoke-Git commit --quiet -m measured
        $mapSha = Get-Head

        [ordered]@{
            schemaVersion = 1
            sha = $mapSha
            knownShards = @('Alpha', 'Beta', 'Gamma', 'Epsilon', 'Window')
            alwaysRun = @('Always', 'Dropped')
            files = [ordered]@{
                'src/Feature.cs' = @(
                    [ordered]@{ s = 0; r = @(2, 2) },
                    [ordered]@{ s = 1; r = @(4, 4) },
                    [ordered]@{ s = 2; r = @(6, 6) },
                    [ordered]@{ s = 3; r = @(8, 8) },
                    [ordered]@{ s = 4; r = @(8, 8) }
                )
            }
        } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $fixture 'shard-map.json') -Encoding utf8
        Copy-Item (Join-Path $fixture 'shard-map.json') shard-map.json
        Set-Content -LiteralPath .git/info/exclude -Value @('*.json') -Encoding utf8

        # Control: an unchanged manifest selects exactly the reached shard and the always-run one.
        $unchangedHead = New-PullRequest -Base $mapSha -Line 2 -Label unchanged-manifest
        $r = Invoke-Selection -Name unchanged -PullRequestHeadSha $unchangedHead
        Assert-True (-not [bool] $r.escalate) "unchanged manifest escalated: $(@($r.reasons) -join '; ')"
        Assert-True ((@($r.shards) -join ',') -ceq 'Alpha,Always,Dropped') "unchanged manifest selected '$(@($r.shards) -join ',')'"

        # Master changes the manifest after the map: Beta redefined, Gamma and Dropped retired, Delta added.
        Invoke-Git checkout --quiet --detach $mapSha
        Write-Manifest $current
        Invoke-Git commit --quiet -am manifest-drift
        $driftSha = Get-Head

        # 1. A change reaching only Alpha stays reduced: redefined Beta and new Delta run, retired
        #    shards disappear, and the unchanged, unreached Epsilon and Window are still skipped: a
        #    manifest edit alone does not move models between position-sliced windows.
        $alphaHead = New-PullRequest -Base $driftSha -Line 2 -Label alpha-change
        $r = Invoke-Selection -Name drift-alpha -PullRequestHeadSha $alphaHead
        Assert-True (-not [bool] $r.escalate) "a change reaching only Alpha escalated after manifest drift: $(@($r.reasons) -join '; ')"
        Assert-True ((@($r.shards) -join ',') -ceq 'Alpha,Always,Beta,Delta') `
            "manifest drift selected '$(@($r.shards) -join ',')' instead of Alpha,Always,Beta,Delta"
        Assert-True (@($r.routes | Where-Object { $_ -like 'Beta <= changed its manifest definition*' }).Count -eq 1) `
            'the redefined shard was not reported as redefined'

        # 2. A change reaching only the retired indexed shard: Gamma's tests moved into Beta, whose
        #    definition changed, so Beta is mandatory and Gamma is simply dropped.
        $gammaHead = New-PullRequest -Base $driftSha -Line 6 -Label gamma-change
        $r = Invoke-Selection -Name drift-gamma -PullRequestHeadSha $gammaHead
        Assert-True (-not [bool] $r.escalate -and (@($r.shards) -join ',') -ceq 'Always,Beta,Delta') `
            "a change reaching a retired shard selected '$(@($r.shards) -join ',')' (escalate=$($r.escalate)): $(@($r.reasons) -join '; ')"

        # 3. A redefined shard's own lines still select it (it is mandatory anyway) without escalation.
        $betaHead = New-PullRequest -Base $driftSha -Line 4 -Label beta-change
        $r = Invoke-Selection -Name drift-beta -PullRequestHeadSha $betaHead
        Assert-True (-not [bool] $r.escalate -and (@($r.shards) -join ',') -ceq 'Always,Beta,Delta') `
            "a change reaching the redefined shard selected '$(@($r.shards) -join ',')' (escalate=$($r.escalate))"

        # 4. A map commit with no committed manifest, while HEAD has one: nothing proves any indexed
        #    shard unchanged, so all of them run. Same source and HEAD manifest as the control above,
        #    so only the missing measured manifest can explain the difference.
        Invoke-Git checkout --quiet --detach $mapSha
        Invoke-Git rm --quiet .github/test-shards.yml
        Invoke-Git commit --quiet -m no-manifest
        $bareMap = Get-Content shard-map.json -Raw | ConvertFrom-Json
        $bareMap.sha = Get-Head
        $bareMap | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $fixture 'bare-map.json') -Encoding utf8
        $bareHead = New-PullRequest -Base $mapSha -Line 2 -Label bare-change
        $r = Invoke-Selection -Name bare -PullRequestHeadSha $bareHead -Map (Join-Path $fixture 'bare-map.json')
        Assert-True (-not [bool] $r.escalate) "an untracked measured manifest escalated: $(@($r.reasons) -join '; ')"
        Assert-True ((@($r.shards) -join ',') -ceq 'Alpha,Always,Beta,Dropped,Epsilon,Gamma,Window') `
            "an untracked measured manifest did not make every indexed shard mandatory: '$(@($r.shards) -join ',')'"

        # 4b. No committed manifest at the map commit NOR at HEAD: nothing accounts for a retired
        #     shard's tests, so reaching one escalates.
        $plainRepo = Join-Path $fixture 'plain'
        Invoke-Git worktree add --quiet --detach $plainRepo $mapSha
        Push-Location $plainRepo
        try {
            Invoke-Git rm --quiet .github/test-shards.yml
            Invoke-Git commit --quiet -m untracked-manifest
            $plainBase = Get-Head
            $plainMap = Get-Content (Join-Path $fixture 'shard-map.json') -Raw | ConvertFrom-Json
            $plainMap.sha = $plainBase
            $plainMap | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $fixture 'plain-map.json') -Encoding utf8
            $plainHead = New-PullRequest -Base $plainBase -Line 6 -Label plain-gamma-change
            $entries = @($current | ForEach-Object { [pscustomobject] $_ })
            ConvertTo-Json -InputObject $entries -Depth 5 | Set-Content -LiteralPath (Join-Path $fixture 'plain-manifest.json') -Encoding utf8
            & $SelectorPath -MapFile (Join-Path $fixture 'plain-map.json') -PullRequestHeadSha $plainHead `
                -ShardManifestFile (Join-Path $fixture 'plain-manifest.json') -ExpectedShards @($entries.name) `
                -OutFile (Join-Path $fixture 'plain.json') 6>$null | Out-Null
            $r = Get-Content (Join-Path $fixture 'plain.json') -Raw | ConvertFrom-Json
            Assert-True ([bool] $r.escalate -and (@($r.reasons) -join ';') -like "*retired shard 'Gamma'*") `
                "an unaccountable retired shard did not escalate: $(@($r.reasons) -join '; ')"
        }
        finally { Pop-Location }

        # 5. The nightly HISTORICAL replay of a map measured before a manifest change. The replay is
        #    unscoped, so the manifest edit itself escalates it (nothing skipped, nothing missed) and
        #    the workflow certifies from fresh coverage instead. It must measure rather than fail on
        #    the retired shards, which is what the map-universe check used to do.
        Invoke-Git checkout --quiet --detach $driftSha
        $outcomes = @($current | ForEach-Object { [ordered]@{ shard = $_.name; outcome = 'success' } })
        ConvertTo-Json -InputObject $outcomes -Depth 3 | Set-Content -LiteralPath audit-outcomes.json -Encoding utf8
        $auditEntries = @(& yq -o=json -I=0 '.shard' .github/test-shards.yml | ConvertFrom-Json)
        ConvertTo-Json -InputObject $auditEntries -Depth 5 | Set-Content -LiteralPath audit-manifest.json -Encoding utf8
        & $missMeasurer -SelectorPath $SelectorPath -MapFile shard-map.json -OutcomesFile audit-outcomes.json `
            -CurrentChangeBaseSha $driftSha -ShardManifestFile audit-manifest.json -OutFile historical-audit.json 6>$null | Out-Null
        Assert-True ($LASTEXITCODE -eq 0) "the historical audit across manifest drift exited $LASTEXITCODE"
        $audit = Get-Content historical-audit.json -Raw | ConvertFrom-Json
        Assert-True ([int] $audit.TotalShards -eq 6 -and [int] $audit.MissCount -eq 0) `
            'the historical audit did not measure the audited run''s six shards without a miss'
        & $SelectorPath -MapFile shard-map.json -ExpectedShards @($auditEntries.name) -AuditUnchangedMap `
            -BaseSha $driftSha -ShardManifestFile audit-manifest.json -OutFile historical-selection.json 6>$null | Out-Null
        $historical = Get-Content historical-selection.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $historical.escalate -and [string] $historical.reason -ceq 'impact-unknown' -and
            (@($historical.reasons) -join ';') -like '*.github/test-shards.yml*') `
            "the historical replay escalated for '$($historical.reason)': $(@($historical.reasons) -join '; ')"

        # 6. The FRESH-coverage audit: the harvested map is measured at the run's own commit with that
        #    commit's manifest, so nothing is redefined and the unchanged tree exercises a reduction.
        [ordered]@{
            schemaVersion = 1
            sha = $driftSha
            knownShards = @('Alpha', 'Beta', 'Epsilon', 'Window')
            alwaysRun = @('Always', 'Delta')
            files = [ordered]@{
                'src/Feature.cs' = @(
                    [ordered]@{ s = 0; r = @(2, 2) },
                    [ordered]@{ s = 1; r = @(4, 4) },
                    [ordered]@{ s = 1; r = @(6, 6) },
                    [ordered]@{ s = 2; r = @(8, 8) },
                    [ordered]@{ s = 3; r = @(8, 8) }
                )
            }
        } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath fresh-map.json -Encoding utf8
        & $missMeasurer -SelectorPath $SelectorPath -MapFile fresh-map.json -OutcomesFile audit-outcomes.json `
            -CurrentChangeBaseSha $driftSha -ShardManifestFile audit-manifest.json -OutFile fresh-audit.json 6>$null | Out-Null
        Assert-True ($LASTEXITCODE -eq 0) "the fresh audit exited $LASTEXITCODE"
        $fresh = Get-Content fresh-audit.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $fresh.Escalated -and (@($fresh.WouldRunShards) -join ',') -ceq 'Always,Delta' -and
            (@($fresh.WouldSkipShards) -join ',') -ceq 'Alpha,Beta,Epsilon,Window') `
            "the fresh audit ran '$(@($fresh.WouldRunShards) -join ',')' and skipped '$(@($fresh.WouldSkipShards) -join ',')' (escalated=$($fresh.Escalated))"

        # 7. Master adds a model after the map. Position-sliced windows may now test other models, so
        #    an indexed window runs even though the pull request never reaches its lines.
        Invoke-Git checkout --quiet --detach $driftSha
        @('namespace Models;', 'public sealed class AddedModel { public AddedModel(int size) { } }') |
            Set-Content -LiteralPath src/AddedModel.cs -Encoding utf8
        Invoke-Git add src/AddedModel.cs
        Invoke-Git commit --quiet -m inventory-grows
        $grownSha = Get-Head
        $grownHead = New-PullRequest -Base $grownSha -Line 2 -Label alpha-after-inventory-change
        $r = Invoke-Selection -Name inventory -PullRequestHeadSha $grownHead
        Assert-True (-not [bool] $r.escalate -and (@($r.shards) -join ',') -ceq 'Alpha,Always,Beta,Delta,Window') `
            "an inventory change did not add the indexed window: '$(@($r.shards) -join ',')' (escalate=$($r.escalate))"
        Assert-True (@($r.routes | Where-Object { $_ -like 'Window <= reflection inventory changed*' }).Count -eq 1) `
            'the window was not reported as widened by the inventory change'
    }
    finally { Pop-Location }
}
catch {
    $fixtureFailure = $_
    throw
}
finally {
    Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-manifest-drift-' `
        -ThrowOnUnsafePath:($null -eq $fixtureFailure)
}

if ($failures.Count -gt 0) {
    Write-Host 'Shard manifest drift proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}
Write-Host 'Shard manifest drift proof passed: redefined and new shards run, retired shards are dropped or escalate when reached, and the audit measures across drift.'
exit 0
