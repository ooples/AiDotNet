<#
.SYNOPSIS
    Proves the rolling per-shard coverage map: every rule in RollingShardMap.ps1, then the whole path through
    real commits - digests written by New-CoverageDigest.ps1, the map built and certified by
    New-RollingShardMap.ps1, selected for by Select-Shards.ps1, and decided by Invoke-RollingSelection.ps1.

.DESCRIPTION
    Part 1 is pure. Part 2 builds a git repository in which a file's line numbers move between the commits two
    groups of shards were measured at, so a selection that translates the change once - to either commit - names
    the wrong shards and fails here (Select-Shards.ps1 -SelfTest 18b holds the same case without git).

    Needs yq on PATH for part 2, as the selector does.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RollingShardMap.ps1"

# Invoke-Shard.ps1's own enum; its coverage decision is evaluated below exactly as written there.
enum CoverageDisposition {
    Instrument
    Carried
    RunWithoutCoverage
}

$failures = [System.Collections.Generic.List[string]]::new()
function Assert-True {
    param([bool] $Condition, [string] $What)
    if (-not $Condition) { [void] $failures.Add($What) }
}
function Assert-ThrowsLike {
    param([scriptblock] $Action, [string] $Pattern, [string] $What)
    $message = $null
    try { & $Action | Out-Null } catch { $message = [string] $_.Exception.Message }
    if ($null -eq $message) { [void] $failures.Add("$What (nothing was refused)") }
    elseif ($message -notmatch $Pattern) { [void] $failures.Add("$What (refused for another reason: $message)") }
}
function Copy-Json($Value) { return ($Value | ConvertTo-Json -Depth 12 -Compress | ConvertFrom-Json) }

# ============================================================ part 1: the rules

# ---- the flag
foreach ($value in @($null, '', 'off', 'OFF', 'true', '1', 'yes', 'enabled', 'shadowed', 'on!')) {
    Assert-True ((Get-RollingMapMode $value) -ceq 'off') "TEST_IMPACT_ROLLING_MAP='$value' was not read as off"
}
Assert-True ((Get-RollingMapMode ' Shadow ') -ceq 'shadow' -and (Get-RollingMapMode 'ON') -ceq 'on') 'shadow and on are not recognised whatever their case'

# ---- shard definition hash
$definition = [pscustomobject]@{ name = 'A'; project = 'tests/T/T.csproj'; filter = 'FullyQualifiedName~A'; timeout = 60 }
$reordered = [ordered]@{ timeout = 60.0; filter = 'FullyQualifiedName~A'; runner = 'aidotnet-cpu'; project = 'tests/T/T.csproj'; name = 'A' }
Assert-True ((Get-ShardDefinitionHash $definition) -ceq (Get-ShardDefinitionHash $reordered)) 'key order, 60 against 60.0, or the per-run runner key changed a shard definition hash'
Assert-True ((Get-ShardDefinitionHash $definition) -cne (Get-ShardDefinitionHash ([pscustomobject]@{ name = 'A'; project = 'tests/T/T.csproj'; filter = 'FullyQualifiedName~B'; timeout = 60 }))) 'a changed filter did not change the shard definition hash'
Assert-True ((Get-ShardDefinitionHash ($definition | ConvertTo-Json -Compress | ConvertFrom-Json)) -ceq (Get-ShardDefinitionHash $definition)) 'a JSON round trip changed a shard definition hash'

# ---- the nightly budget: order and cap
$budgetManifest = @('L1', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'T1', 'A1', 'N1', 'R1', 'S1' | ForEach-Object { [pscustomobject]@{ name = $_ } })
$budgetAnchor = [pscustomobject]@{ knownShards = @('L1', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'T1', 'R1', 'S1'); alwaysRun = @('A1') }
$budgetArguments = @{
    Manifest = $budgetManifest; AnchorMap = $budgetAnchor
    NoCoverageShards = @('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'T1', 'A1', 'N1', 'R1', 'S1'); TimingShards = @('T1')
    MeasuredOrder = @{ H1 = 5; H3 = 2 }; RedefinedShards = @('R1'); StaleShards = @('S1', 'L1')
}
$plan = Select-RollingCoverageShards @budgetArguments -BudgetMinutes 100
Assert-True (($plan.Instrument -join ',') -ceq 'N1,R1,S1,H2') "budget order: unmeasured and redefined, then stale, then oldest gave '$($plan.Instrument -join ',')'"
Assert-True ($plan.PredictedExtraMinutes -eq 100) "four shards at 25 minutes predicted $($plan.PredictedExtraMinutes)"
Assert-True ((@($plan.Candidates | ForEach-Object name) -join ',') -ceq 'N1,R1,S1,H2,H4,H5,H6,H3,H1') "candidate order was '$(@($plan.Candidates | ForEach-Object name) -join ',')': within a class the oldest measurement goes first, then name"
Assert-True ((@($plan.Excluded | Sort-Object name | ForEach-Object name) -join ',') -ceq 'A1,L1,T1') "light, timing and anchor always-run shards were not all excluded ('$(@($plan.Excluded | ForEach-Object name) -join ',')')"
Assert-True (@($plan.Excluded | Where-Object { $_.name -ceq 'T1' -and $_.why -like 'timing shard*' }).Count -eq 1) 'a timing shard was not excluded as a timing shard'
Assert-True (@($plan.Excluded | Where-Object { $_.name -ceq 'A1' -and $_.why -like 'always-run in the anchor*' }).Count -eq 1) 'a shard the weekly run could not digest was not left to the weekly run'
Assert-True (((Select-RollingCoverageShards @budgetArguments -BudgetMinutes 60).Instrument -join ',') -ceq 'N1,R1') 'a 60-minute budget did not stop at two 25-minute shards'
Assert-True (@((Select-RollingCoverageShards @budgetArguments -BudgetMinutes 0).Instrument).Count -eq 0) 'a zero budget instrumented something'
$uneven = Select-RollingCoverageShards @budgetArguments -BudgetMinutes 100 -ShardExtraMinutes @{ S1 = 60 }
Assert-True (($uneven.Instrument -join ',') -ceq 'N1,R1,H2,H4' -and $uneven.PredictedExtraMinutes -eq 100) "a shard that does not fit was not passed over for cheaper ones ('$($uneven.Instrument -join ',')')"
# The default budget turns the whole heavy matrix over in six nights, nobody twice before everybody once.
$heavyNames = @(1..103 | ForEach-Object { 'Heavy {0:000}' -f $_ })
$rotationManifest = @($heavyNames | ForEach-Object { [pscustomobject]@{ name = $_ } })
$rotationAnchor = [pscustomobject]@{ knownShards = $heavyNames; alwaysRun = @() }
$measuredOn = @{}
$nights = 0
while ($measuredOn.Count -lt $heavyNames.Count -and $nights -lt 20) {
    $nights++
    $tonight = Select-RollingCoverageShards -Manifest $rotationManifest -NoCoverageShards $heavyNames -AnchorMap $rotationAnchor -MeasuredOrder $measuredOn
    Assert-True (@($tonight.Instrument).Count -eq [Math]::Min(18, $heavyNames.Count - $measuredOn.Count + @($tonight.Instrument | Where-Object { $measuredOn.ContainsKey($_) }).Count)) "night $nights instrumented $(@($tonight.Instrument).Count) shard(s) on the default budget"
    foreach ($name in $tonight.Instrument) {
        Assert-True (-not $measuredOn.ContainsKey($name) -or $measuredOn.Count -ge $heavyNames.Count) "'$name' was measured again on night $nights before every shard had been measured once"
        $measuredOn[$name] = $nights
    }
}
Assert-True ($nights -eq 6) "the default budget ($script:RollingBudgetMinutes min at $script:RollingShardExtraMinutes min a shard) re-measured 103 heavy shards in $nights night(s), not 6"
Assert-True ((Resolve-RollingBudgetMinutes '') -eq 450 -and (Resolve-RollingBudgetMinutes $null) -eq 450) 'an unset budget is not the default'
Assert-True ((Resolve-RollingBudgetMinutes ' 300 ') -eq 300 -and (Resolve-RollingBudgetMinutes '0') -eq 0) 'a numeric budget was not honoured'
Assert-True ((Resolve-RollingBudgetMinutes 'lots' 6>$null) -eq 450 -and (Resolve-RollingBudgetMinutes '-5' 6>$null) -eq 450) 'an unreadable budget did not fall back to the default'

# ---- the no-coverage lists come from the script that decides coverage
$shardScript = Join-Path $PSScriptRoot '../../.github/scripts/Invoke-Shard.ps1'
$shardScriptLines = @(Get-Content -LiteralPath $shardScript)
$lists = Get-NoCoverageShardLists -ScriptLines $shardScriptLines -Manifest @([pscustomobject]@{ name = 'Declared heavy'; heavy = $true }, [pscustomobject]@{ name = 'Light' })
Assert-True (($lists.Timing -join '|') -ceq 'Integration - Serial Perf|Integration - CPU Parallelism Probe') "the timing shards were read as '$($lists.Timing -join '|')'"
Assert-True ($lists.NoCoverage.Count -ge 50 -and $lists.NoCoverage -ccontains 'Declared heavy' -and $lists.NoCoverage -ccontains 'Integration S' -and
    $lists.NoCoverage -ccontains 'Integration - CPU Parallelism Probe' -and $lists.NoCoverage -cnotcontains 'Light') 'the no-coverage list does not hold the named heavy shards, the manifest''s heavy entries and the timing shards'
Assert-ThrowsLike { Get-NoCoverageShardLists -ScriptLines @('$other = @(', "'x'", ')') -Manifest @([pscustomobject]@{ name = 'a' }) } 'no longer defines' 'a shard script without the heavy list was accepted'

# ---- flag off: Invoke-Shard.ps1's coverage decision is the rule it was, for every shard kind and event
$decisionBlock = [regex]::Match((Get-Content -LiteralPath $shardScript -Raw), '(?ms)^\$forcedCoverageRun = .*?(?=^if \(\$forcedCoverageRun -and \$coverageDisposition)')
Assert-True $decisionBlock.Success 'the coverage decision could not be found in Invoke-Shard.ps1'
if ($decisionBlock.Success) {
    $decide = [scriptblock]::Create($decisionBlock.Value + "`n`$collectCoverage")
    $saved = @{ Event = $env:GITHUB_EVENT_NAME; Force = $env:SHARD_FORCE_COVERAGE; Rolling = $env:COVERAGE_ROLLING }
    try {
        $shardName = 'Heavy 001'
        foreach ($heavyShard in $true, $false) { foreach ($measuresTiming in $true, $false) {
        foreach ($eventName in 'pull_request', 'push', 'schedule', 'workflow_dispatch', 'merge_group') {
        foreach ($forced in 'true', '') { foreach ($coverageDisposition in [CoverageDisposition]::Instrument, [CoverageDisposition]::Carried) {
            $env:GITHUB_EVENT_NAME = $eventName
            $env:SHARD_FORCE_COVERAGE = $forced
            $legacy = if ($forced -eq 'true') { $coverageDisposition -eq [CoverageDisposition]::Instrument } elseif ($eventName -eq 'pull_request') { $false } else { -not ($heavyShard -or $measuresTiming) }
            $where = "heavy=$heavyShard timing=$measuresTiming event=$eventName forced='$forced' disposition=$coverageDisposition"
            foreach ($unset in $null, '', '[]', '["Another shard"]', 'not json') {
                $env:COVERAGE_ROLLING = $unset
                Assert-True ([bool] (& $decide 6>$null) -eq [bool] $legacy) "COVERAGE_ROLLING='$unset' changed the coverage decision ($where)"
            }
            $env:COVERAGE_ROLLING = '["Heavy 001"]'
            $expected = $legacy -or ($eventName -eq 'schedule' -and $forced -ne 'true' -and -not $measuresTiming)
            Assert-True ([bool] (& $decide 6>$null) -eq [bool] $expected) "a shard named in COVERAGE_ROLLING decided coverage=$(-not $expected) ($where)"
        } } } } }
    }
    finally { $env:GITHUB_EVENT_NAME = $saved.Event; $env:SHARD_FORCE_COVERAGE = $saved.Force; $env:COVERAGE_ROLLING = $saved.Rolling }
}

# ---- the map builder: which digests may enter
$shaA = 'a' * 40; $shaB = 'b' * 40; $shaC = 'c' * 40; $blob = 'd' * 40
$now = [DateTimeOffset]::Parse('2026-11-14T12:00:00Z')
$anchorMap = Copy-Json @{
    schemaVersion = 1; sha = $shaA; knownShards = @('S1', 'S2', 'S3'); alwaysRun = @('H')
    files = @{ 'src/X.cs' = @(@{ s = 0; r = @(1, 5) }, @{ s = 1; r = @(10, 12) }); 'src/Y.cs' = @(, @{ s = 2; r = @(1, 2) }) }
}
$entries = @('S1', 'S2', 'S3', 'H' | ForEach-Object { [pscustomobject]@{ name = $_; project = 'tests/T/T.csproj'; filter = "FullyQualifiedName~$_" } })
function New-TestDigest([string] $Shard, [string] $Sha, [long] $RunId, [hashtable] $Files = @{ 'src/X.cs' = @(3, 9) }, $Entry = $null) {
    if ($null -eq $Entry) { $Entry = [pscustomobject]@{ name = $Shard; project = 'tests/T/T.csproj'; filter = "FullyQualifiedName~$Shard" } }
    return Copy-Json ([ordered]@{
        schemaVersion = 1; shard = $Shard; files = $Files; measuredSha = $Sha; runId = $RunId; runAttempt = 1
        definitionHash = (Get-ShardDefinitionHash $Entry); manifestBlob = $blob
    })
}
function New-TestSource([long] $RunId, [string] $Sha, [object[]] $Digests, [hashtable] $Outcomes = @{ S1 = 'success'; S2 = 'success'; S3 = 'success'; H = 'success' },
    [double] $AgeDays = 1, [bool] $Descends = $true, [object[]] $Manifest = $entries) {
    return [pscustomobject]@{
        RunId = $RunId; Sha = $Sha; CreatedAt = $now.AddDays(-$AgeDays); DescendsFromAnchor = $Descends
        Manifest = $Manifest; ManifestBlob = $blob; Outcomes = $Outcomes; Digests = $Digests
    }
}
function Get-Rejection($Source) {
    $built = New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId 77 -Sources @($Source) -Now $now
    Assert-True ($built.Refreshed.Count -eq 0) "a digest that must be refused refreshed '$($built.Refreshed.Keys -join ',')'"
    return [string] (@($built.Rejected | ForEach-Object reason) -join ' | ')
}
$good = New-TestDigest 'S1' $shaB 501
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($good) @{ S1 = 'failure'; S2 = 'success' })) -match 'did not complete successfully \(failure\)') 'a digest from a FAILED shard job was not refused'
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($good) @{ S1 = 'cancelled' })) -match 'did not complete successfully \(cancelled\)') 'a digest from a cancelled shard job was not refused'
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($good) @{ S2 = 'success' })) -match 'did not complete successfully \(no job\)') 'a digest with no shard job behind it was not refused'
$redefined = New-TestDigest 'S1' $shaB 501 -Entry ([pscustomobject]@{ name = 'S1'; project = 'tests/T/T.csproj'; filter = 'FullyQualifiedName~Other' })
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($redefined))) -match 'differs from the manifest at its commit') 'a digest measured with another shard definition than its commit''s manifest was not refused'
$otherBlob = Copy-Json $good; $otherBlob.manifestBlob = 'e' * 40
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($otherBlob))) -match 'different shard manifest') 'a digest from a job that ran another manifest was not refused'
Assert-True ((Get-Rejection (New-TestSource 501 $shaC @($good))) -match 'not at its run''s commit') 'a digest measured at another commit than its run was not refused'
Assert-True ((Get-Rejection (New-TestSource 502 $shaB @($good))) -match 'belongs to run 501') 'a digest from another run was not refused'
$legacy = Copy-Json @{ schemaVersion = 1; shard = 'S1'; files = @{ 'src/X.cs' = @(3, 9) } }
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($legacy))) -match 'does not record its measuredSha') 'a digest that does not record its commit was not refused'
$carriedDigest = Copy-Json $good; $carriedDigest | Add-Member -NotePropertyName carriedFrom -NotePropertyValue $shaA
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($carriedDigest))) -match 'carried forward') 'a carried digest was taken for a measurement'
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($good) -Descends $false)) -match 'does not descend from the anchor') 'a digest from a commit that does not descend from the anchor was not refused'
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($good) -AgeDays 11)) -match 'older than 10 day') 'a digest older than the limit was not refused'
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @((New-TestDigest 'H' $shaB 501)))) -match 'always-run') 'a digest for a shard the anchor lists as always-run was not refused'
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @((New-TestDigest 'S1' $shaB 501 @{ 'src/X.cs' = @(9, 3) })))) -match 'invalid range') 'a digest with an inverted range was not refused'
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @((New-TestDigest 'S1' $shaB 501 @{})))) -match 'executed no file') 'an empty digest was not refused'
$mustCover = @($entries | ForEach-Object { $copy = Copy-Json $_; if ($copy.name -ceq 'S1') { $copy | Add-Member -NotePropertyName mustCover -NotePropertyValue @('src/Subject/*') }; $copy })
$coveringElsewhere = New-TestDigest 'S1' $shaB 501 -Entry ($mustCover | Where-Object name -ceq 'S1')
Assert-True ((Get-Rejection (New-TestSource 501 $shaB @($coveringElsewhere) -Manifest $mustCover)) -match 'mustCover') 'a digest that never reached its shard''s subject was not refused'
$twice = New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId 77 -Sources @((New-TestSource 501 $shaB @($good, (Copy-Json $good)))) -Now $now
Assert-True (@($twice.Rejected | Where-Object { $_.reason -match 'more than one digest' }).Count -eq 1) 'a second digest for one shard in one run was not refused'

# ---- the map builder: what it builds
$built = New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId 77 -Sources @((New-TestSource 501 $shaB @($good))) -Now $now
$rolling = Copy-Json $built.Map
Assert-RollingShardMap -Map $rolling
Assert-True (($rolling.layers | ForEach-Object sha) -join ',' -ceq "$shaA,$shaB") 'a refreshed shard did not get a layer at its own measured commit after the anchor''s'
Assert-True ((@($rolling.layers[0].map.knownShards) -join ',') -ceq 'S2,S3' -and (@($rolling.layers[1].map.knownShards) -join ',') -ceq 'S1') 'the refreshed shard did not move out of the anchor layer'
Assert-True (($rolling.layers[0].map.files.'src/X.cs' | ConvertTo-Json -Compress) -ceq '{"s":0,"r":[10,12]}' -and
    ($rolling.layers[0].map.files.'src/Y.cs' | ConvertTo-Json -Compress) -ceq '{"s":1,"r":[1,2]}') 'the anchor layer''s occurrences were not re-addressed with their ranges untouched'
Assert-True (($rolling.layers[1].map.files.'src/X.cs' | ConvertTo-Json -Compress) -ceq '{"s":0,"r":[3,9]}') 'the refreshed shard''s layer does not hold its digest''s ranges'
Assert-True ($rolling.shards.S1.source -ceq 'nightly' -and $rolling.shards.S1.sha -ceq $shaB -and [long] $rolling.shards.S1.runId -eq 501 -and
    $rolling.shards.S2.source -ceq 'anchor' -and $rolling.shards.S2.sha -ceq $shaA -and [long] $rolling.shards.S2.runId -eq 77) 'per-shard measured commit and source run are not recorded'
Assert-True ((@($rolling.alwaysRun) -join ',') -ceq 'H' -and $rolling.kind -ceq 'rolling' -and [long] $rolling.anchor.runId -eq 77) 'the rolling map does not carry the anchor''s always-run list and identity'
$newer = New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId 77 -Now $now -Sources @(
    (New-TestSource 501 $shaB @($good, (New-TestDigest 'S2' $shaB 501)) -AgeDays 2),
    (New-TestSource 502 $shaC @((New-TestDigest 'S1' $shaC 502 @{ 'src/X.cs' = @(4, 8) })) -AgeDays 1))
$newerMap = Copy-Json $newer.Map
Assert-True ($newerMap.shards.S1.sha -ceq $shaC -and $newerMap.shards.S2.sha -ceq $shaB -and @($newerMap.layers).Count -eq 3) 'the newest digest per shard did not win across runs'
Assert-ThrowsLike { New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId 77 -Now $now -Sources @(
    (New-TestSource 502 $shaC @() -AgeDays 1), (New-TestSource 501 $shaB @($good) -AgeDays 2)) } 'oldest first' 'sources out of order were accepted'
$sameCommit = Copy-Json (New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId 77 -Sources @((New-TestSource 501 $shaA @((New-TestDigest 'S1' $shaA 501)))) -Now $now).Map
Assert-True (@($sameCommit.layers).Count -eq 1 -and $sameCommit.shards.S1.source -ceq 'nightly') 'a nightly at the anchor''s own commit did not share the anchor''s layer'
$withNew = @($entries) + @([pscustomobject]@{ name = 'S4'; project = 'tests/T/T.csproj'; filter = 'FullyQualifiedName~S4' })
$entered = Copy-Json (New-RollingShardMapObject -AnchorMap $anchorMap -AnchorRunId 77 -Now $now -Sources @(
    (New-TestSource 501 $shaB @((New-TestDigest 'S4' $shaB 501)) @{ S4 = 'success' } -Manifest $withNew))).Map
Assert-True ($entered.shards.S4.sha -ceq $shaB) 'a shard the anchor has never seen could not enter through a valid digest'

# ---- structure: coordinate systems stay apart
foreach ($case in @(
    @{ What = 'two layers at one commit'; Pattern = 'two layers at commit'; Edit = { param($m) $m.layers[1].sha = $shaA; $m.layers[1].map.sha = $shaA } },
    @{ What = 'a layer holding a map measured at another commit'; Pattern = 'holds a map measured at'; Edit = { param($m) $m.layers[1].map.sha = $shaC } },
    @{ What = 'a shard indexed in two layers'; Pattern = 'more than one place'; Edit = { param($m) $m.layers[1].map.knownShards = @('S1', 'S2') } },
    @{ What = 'a layer with its own always-run list'; Pattern = 'always-run list'; Edit = { param($m) $m.layers[1].map.alwaysRun = @('Q') } },
    @{ What = 'a shard recorded at another commit than its layer'; Pattern = 'recorded as measured at'; Edit = { param($m) $m.shards.S1.sha = $shaC } },
    @{ What = 'an anchor measurement claimed at another commit'; Pattern = 'recorded as measured at|claims the anchor'; Edit = { param($m) $m.shards.S1.source = 'anchor' } },
    @{ What = 'a shard with no recorded measurement'; Pattern = 'no recorded measurement'; Edit = { param($m) $m.shards.PSObject.Properties.Remove('S1') } }
)) {
    $broken = Copy-Json $rolling
    & $case.Edit $broken
    Assert-ThrowsLike { Assert-RollingShardMap -Map $broken } $case.Pattern "a rolling map with $($case.What) was accepted"
}

# ---- certification
$hash = '0' * 64
$anchorCertificate = [pscustomobject]@{ certificationRunId = 77; candidateMapSha = $shaA; missCount = 0 }
function New-TestAudit([string[]] $Run, [string[]] $Skip, [string[]] $Failed, [bool] $Escalated = $false) {
    $missed = @($Failed | Where-Object { $_ -cin $Skip })
    return Copy-Json ([ordered]@{
        Escalated = $Escalated; TotalShards = $Run.Count + $Skip.Count; WouldRun = $Run.Count; WouldSkip = $Skip.Count
        WouldRunShards = @($Run); WouldSkipShards = @($Skip); Failed = $Failed.Count; Missed = @($missed); MissCount = $missed.Count
    })
}
function New-TestOutcomes([string[]] $Failed = @()) {
    return @('H', 'S1', 'S2', 'S3' | ForEach-Object { [pscustomobject]@{ shard = $_; outcome = $(if ($_ -cin $Failed) { 'failure' } else { 'success' }) } })
}
$certifyArguments = @{
    RollingMap = $rolling; RollingMapHash = $hash; Refreshed = $built.Refreshed; AnchorMap = $anchorMap; AnchorCertificate = $anchorCertificate
    AnchorRunId = 77; AuditSourceRunId = 501; AuditSourceSha = $shaB; CertificationRunId = 77; Now = $now
}
$certified = Get-RollingCertificationDecision @certifyArguments -Audit (New-TestAudit @('H', 'S1') @('S2', 'S3') @()) -Outcomes (New-TestOutcomes)
Assert-True ($certified.Eligible -and $certified.Disposition -ceq 'Certified' -and $null -ne $certified.Certificate -and $certified.Certificate.mapHash -ceq $hash) 'a clean reducing replay did not certify the rolling map'
$ranFailure = Get-RollingCertificationDecision @certifyArguments -Audit (New-TestAudit @('H', 'S1') @('S2', 'S3') @('S1')) -Outcomes (New-TestOutcomes @('S1'))
Assert-True ($ranFailure.Eligible) 'a failure in a shard the rolling map WOULD run was treated as a miss'
$miss = Get-RollingCertificationDecision @certifyArguments -Audit (New-TestAudit @('H', 'S1') @('S2', 'S3') @('S2')) -Outcomes (New-TestOutcomes @('S2'))
Assert-True (-not $miss.Eligible -and $miss.Disposition -ceq 'RevokedMiss' -and $null -eq $miss.Certificate -and (@($miss.Decision.missed) -join ',') -ceq 'S2') 'a failed shard the rolling map would skip did not refuse the certificate'
$noReduction = Get-RollingCertificationDecision @certifyArguments -Audit (New-TestAudit @('H', 'S1', 'S2', 'S3') @() @() $true) -Outcomes (New-TestOutcomes)
Assert-True (-not $noReduction.Eligible -and $noReduction.Disposition -ceq 'SafeNoReduction') 'an escalated replay certified the rolling map'
$cleanAudit = New-TestAudit @('H', 'S1') @('S2', 'S3') @()
$tampered = New-TestAudit @('H', 'S1') @('S2', 'S3') @('S2'); $tampered.Missed = @(); $tampered.MissCount = 0
$agedArguments = $certifyArguments.Clone(); $agedArguments.Now = $now.AddDays(11)
$wrongRun = $certifyArguments.Clone(); $wrongRun.CertificationRunId = 78
$unverified = $certifyArguments.Clone(); $unverified.Refreshed = @{}
$otherAnchor = $certifyArguments.Clone(); $otherAnchor.AnchorCertificate = [pscustomobject]@{ certificationRunId = 77; candidateMapSha = $shaC; missCount = 0 }
$missedAnchor = $certifyArguments.Clone(); $missedAnchor.AnchorCertificate = [pscustomobject]@{ certificationRunId = 77; candidateMapSha = $shaA; missCount = 1 }
foreach ($case in @(
    @{ What = 'a rolling map certified by another run than its anchor''s'; Pattern = 'run that certified its anchor'; Arguments = $wrongRun; Audit = $cleanAudit; Outcomes = (New-TestOutcomes) },
    @{ What = 'a nightly digest the builder did not verify'; Pattern = 'did not verify'; Arguments = $unverified; Audit = $cleanAudit; Outcomes = (New-TestOutcomes) },
    @{ What = 'an anchor certificate for another map'; Pattern = 'another map'; Arguments = $otherAnchor; Audit = $cleanAudit; Outcomes = (New-TestOutcomes) },
    @{ What = 'an anchor certificate recording a miss'; Pattern = 'records a selection miss'; Arguments = $missedAnchor; Audit = $cleanAudit; Outcomes = (New-TestOutcomes) },
    @{ What = 'a digest older than the limit at certification'; Pattern = 'older than 10 day'; Arguments = $agedArguments; Audit = $cleanAudit; Outcomes = (New-TestOutcomes) },
    @{ What = 'an audit that hides a skipped failure'; Pattern = 'does not equal the failed shards'; Arguments = $certifyArguments; Audit = $tampered; Outcomes = (New-TestOutcomes @('S2')) },
    @{ What = 'a nightly with an unfinished shard'; Pattern = 'unauditable outcome'; Arguments = $certifyArguments; Audit = $cleanAudit
       Outcomes = @((New-TestOutcomes | Where-Object shard -cne 'S3') + [pscustomobject]@{ shard = 'S3'; outcome = 'cancelled' }) }
)) {
    $arguments = $case.Arguments
    Assert-ThrowsLike { Get-RollingCertificationDecision @arguments -Audit $case.Audit -Outcomes $case.Outcomes } $case.Pattern "$($case.What) was certified"
}

# ---- what selection requires of a certificate before a rolling map decides anything
$certificate = Copy-Json $certified.Certificate
$useArguments = @{ RollingMap = $rolling; RollingMapHash = $hash; Certificate = $certificate; AnchorMap = $anchorMap; AnchorRunId = 77; Now = $now }
Assert-CertifiedRollingShardMap @useArguments
$changedAlways = Copy-Json $rolling; $changedAlways.alwaysRun = @()
$lostShard = Copy-Json $anchorMap; $lostShard.knownShards = @('S1', 'S2', 'S3', 'S9')
foreach ($case in @(
    @{ What = 'map bytes that do not match the certificate'; Pattern = 'does not match the rolling map bytes'; Edit = { param($a) $a.RollingMapHash = '1' * 64 } },
    # The resolver returns another run once the anchor is superseded, and none at all once it is revoked.
    @{ What = 'a rolling map built on an anchor that is no longer the one in use'; Pattern = 'not the anchor in use'; Edit = { param($a) $a.AnchorRunId = 78 } },
    @{ What = 'an anchor at another commit'; Pattern = 'another anchor commit'; Edit = { param($a) $other = Copy-Json $anchorMap; $other.sha = $shaC; $a.AnchorMap = $other } },
    @{ What = 'a refused certificate'; Pattern = "disposition is 'RevokedMiss'"; Edit = { param($a) $c = Copy-Json $certificate; $c.disposition = 'RevokedMiss'; $a.Certificate = $c } },
    @{ What = 'a certificate recording a miss'; Pattern = 'selection miss'; Edit = { param($a) $c = Copy-Json $certificate; $c.missCount = 1; $a.Certificate = $c } },
    @{ What = 'a certificate from an escalated replay'; Pattern = 'escalated audit'; Edit = { param($a) $c = Copy-Json $certificate; $c.escalated = $true; $a.Certificate = $c } },
    @{ What = 'a certificate that proves no reduction'; Pattern = 'reduced selection'; Edit = { param($a) $c = Copy-Json $certificate; $c.wouldSkip = 0; $a.Certificate = $c } },
    @{ What = 'a digest that aged past the limit after certification'; Pattern = 'older than 10 day'; Edit = { param($a) $a.Now = $now.AddDays(11) } },
    @{ What = 'a rolling map that dropped the anchor''s always-run list'; Pattern = 'always-run list'; Edit = { param($a) $a.RollingMap = $changedAlways } },
    @{ What = 'a rolling map missing one of the anchor''s shards'; Pattern = "lost the anchor's shard"; Edit = { param($a) $a.AnchorMap = $lostShard } }
)) {
    $arguments = $useArguments.Clone()
    & $case.Edit $arguments
    Assert-ThrowsLike { Assert-CertifiedRollingShardMap @arguments } $case.Pattern "$($case.What) was accepted for selection"
}

# ---- which selection decides
$allShards = @('H', 'S1', 'S2', 'S3')
function New-TestSelection([string[]] $Shards, [bool] $Escalate = $false, [string] $Reason = 'selected', [bool] $RequiresShards = $true) {
    return Copy-Json ([ordered]@{ escalate = $Escalate; requiresValidation = $true; requiresShards = $RequiresShards; reason = $Reason; shards = @($Shards) })
}
$anchorSelection = New-TestSelection @('H', 'S1', 'S2')
$rollingSelection = New-TestSelection @('H', 'S1', 'S3')
$shadow = Get-RollingSelectionDecision -Mode shadow -AnchorSelection $anchorSelection -RollingSelection $rollingSelection -AllShards $allShards
Assert-True ($shadow.decides -ceq 'anchor' -and $shadow.mode -ceq 'shadow') 'shadow mode let the rolling map decide'
Assert-True ((@($shadow.comparison.droppedByRolling) -join ',') -ceq 'S2' -and (@($shadow.comparison.addedByRolling) -join ',') -ceq 'S3' -and $shadow.comparison.inBoth -eq 2 -and
    $shadow.comparison.anchor.shardCount -eq 3 -and $shadow.comparison.rolling.shardCount -eq 3) 'shadow mode did not record both selections and their differences'
$on = Get-RollingSelectionDecision -Mode on -AnchorSelection $anchorSelection -RollingSelection $rollingSelection -AllShards $allShards
Assert-True ($on.decides -ceq 'rolling') 'a certified, usable rolling selection did not decide in on mode'
foreach ($case in @(
    @{ What = 'no certified rolling map'; Selection = $null; Problem = 'the anchor''s map run 77 published no certified rolling map' },
    @{ What = 'no rolling selection'; Selection = $null; Problem = '' },
    @{ What = 'an unreadable rolling map'; Selection = (New-TestSelection @() $true 'map-unreadable'); Problem = '' },
    @{ What = 'a rolling map whose commit is missing'; Selection = (New-TestSelection @() $true 'map-unresolvable'); Problem = '' },
    @{ What = 'a failed rolling selection'; Selection = (New-TestSelection @() $true 'selection-failed'); Problem = '' },
    @{ What = 'a disagreement about needing shards at all'; Selection = (New-TestSelection @() $false 'build-only' $false); Problem = '' }
)) {
    $decision = Get-RollingSelectionDecision -Mode on -AnchorSelection $anchorSelection -RollingSelection $case.Selection -RollingProblem $case.Problem -AllShards $allShards
    Assert-True ($decision.decides -ceq 'anchor' -and $decision.reason -like 'the certified weekly map decides*') "with $($case.What), the anchor did not decide in on mode"
    Assert-True ((@($decision.comparison.anchor.shards) -join ',') -ceq 'H,S1,S2') "with $($case.What), the anchor's selection was not carried through unchanged"
}
$escalatedAnchor = Get-RollingSelectionDecision -Mode shadow -AnchorSelection (New-TestSelection @() $true 'impact-unknown') -RollingSelection $rollingSelection -AllShards $allShards
Assert-True ($escalatedAnchor.comparison.anchor.shardCount -eq 4 -and (@($escalatedAnchor.comparison.droppedByRolling) -join ',') -ceq 'S2') 'an escalated anchor selection was not compared as the full matrix'
Assert-ThrowsLike { Get-RollingSelectionDecision -Mode off -AnchorSelection $anchorSelection -RollingSelection $rollingSelection -AllShards $allShards } 'is off' 'a decision was made with the flag off'
$afterShadow = Get-RollingShadowOutcome -Decision $shadow -Outcomes @{ H = 'success'; S1 = 'success'; S2 = 'failure'; S3 = 'skipped' }
Assert-True ((@($afterShadow.droppedThenFailed) -join ',') -ceq 'S2' -and @($afterShadow.droppedNotRun).Count -eq 0) 'a shard the rolling map would have dropped that then failed was not reported'
$afterOn = Get-RollingShadowOutcome -Decision $on -Outcomes @{ H = 'success'; S1 = 'success'; S3 = 'success' }
Assert-True (@($afterOn.droppedThenFailed).Count -eq 0 -and (@($afterOn.droppedNotRun) -join ',') -ceq 'S2') 'a shard the rolling map did drop was not reported as not run'

# ============================================================ part 2: real commits

$selector = Join-Path $PSScriptRoot 'Select-Shards.ps1'
$digestWriter = Join-Path $PSScriptRoot 'New-CoverageDigest.ps1'
$mapBuilder = Join-Path $PSScriptRoot 'New-RollingShardMap.ps1'
$rollingSelector = Join-Path $PSScriptRoot 'Invoke-RollingSelection.ps1'
$budgetSelector = Join-Path $PSScriptRoot 'Select-RollingCoverageShards.ps1'
$root = Join-Path ([IO.Path]::GetFullPath([IO.Path]::GetTempPath())) ('aidotnet-rolling-e2e-' + [guid]::NewGuid().ToString('N'))
$repo = Join-Path $root 'repo'
$out = Join-Path $root 'out'
$savedSwitch = $env:TEST_IMPACT_PRECISE_FALLBACKS
function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments)] [string[]] $Arguments)
    $output = & git @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "git $($Arguments -join ' ') failed: $(($output | Out-String).Trim())" }
    return $output
}
function Write-Source([string] $Path, [string[]] $Lines) {
    $full = Join-Path $repo $Path
    New-Item -ItemType Directory -Path (Split-Path -Parent $full) -Force | Out-Null
    [IO.File]::WriteAllText($full, (($Lines -join "`n") + "`n"), [System.Text.UTF8Encoding]::new($false))
}
function Get-Head { return ((Invoke-Git rev-parse HEAD) | Out-String).Trim() }
function Get-Names($Selection) { return (@($Selection.shards | Sort-Object) -join ',') }

try {
    New-Item -ItemType Directory -Path $repo, $out -Force | Out-Null
    Push-Location $repo
    try {
        $env:TEST_IMPACT_PRECISE_FALLBACKS = ''
        Invoke-Git init --quiet
        Invoke-Git config user.email 'ci-impact-fixture@example.invalid'
        Invoke-Git config user.name 'CI impact fixture'
        Invoke-Git config commit.gpgSign false
        Invoke-Git config core.autocrlf false
        $hooks = Join-Path $root 'disabled-hooks'
        New-Item -ItemType Directory -Path $hooks -Force | Out-Null
        Invoke-Git config core.hooksPath $hooks

        # ---- commit 1: the anchor's commit. F.cs has one method per line: block P is lines 20-22, Q is 40-42.
        $feature = [string[]] (@('namespace Fixture { public class F {') + @(2..99 | ForEach-Object { '    public int M{0:000}() {{ return {0}; }}' -f $_ }) + @('} }'))
        Write-Source 'src/Area/F.cs' $feature
        Write-Source 'src/Area/Other.cs' @('namespace Fixture { public class Other {', '    public int Get() { return 1; }', '} }')
        Write-Source 'src/Area/Churn.cs' @('namespace Fixture { public class Churn {', '    public int Get() { return 1; }', '} }')
        $shardNames = @('Early', 'Shifted', 'Late', 'Moved', 'Always')
        $manifestYaml = { param([hashtable] $Filters = @{})
            $lines = @('shard:')
            foreach ($name in $shardNames) {
                $filter = if ($Filters.ContainsKey($name)) { $Filters[$name] } else { "FullyQualifiedName~$name" }
                $lines += @("  - name: $name", '    project: tests/T/T.csproj', "    filter: $filter")
            }
            return $lines
        }
        Write-Source '.github/test-shards.yml' (& $manifestYaml)
        Invoke-Git add --all
        Invoke-Git commit --quiet -m 'commit 1'
        $c1 = Get-Head
        $manifestFile = Join-Path $out 'shard-manifest.json'
        $writeManifest = {
            $all = @(yq -o=json -I=0 '.shard' .github/test-shards.yml | ConvertFrom-Json)
            ConvertTo-Json -InputObject @($all) -Depth 5 | Set-Content -LiteralPath $manifestFile -Encoding utf8
        }
        & $writeManifest

        # The certified weekly map, measured at commit 1: Early and Moved execute Q, Shifted and Late execute P.
        $anchorFile = Join-Path $out 'anchor-map.json'
        [ordered]@{
            schemaVersion = 1; sha = $c1; generatedUtc = '2026-11-07T00:00:00Z'
            knownShards = @('Early', 'Shifted', 'Late', 'Moved'); alwaysRun = @('Always')
            files = [ordered]@{ 'src/Area/F.cs' = @(
                [ordered]@{ s = 0; r = @(40, 42) }, [ordered]@{ s = 1; r = @(20, 22) }, [ordered]@{ s = 2; r = @(20, 22) }, [ordered]@{ s = 3; r = @(40, 42) }) }
        } | ConvertTo-Json -Depth 8 -Compress | Set-Content -LiteralPath $anchorFile -Encoding utf8
        $anchorCertificateFile = Join-Path $out 'anchor-certification.json'
        [ordered]@{ certificationRunId = 1001; candidateMapSha = $c1; missCount = 0 } | ConvertTo-Json | Set-Content -LiteralPath $anchorCertificateFile -Encoding utf8

        # ---- commit 2: 20 lines above everything in F.cs (P is now 40-42, Q is 60-62), a new file, one edited file.
        $shifted = [string[]] (@(1..20 | ForEach-Object { "// added on the base branch $_" }) + $feature)
        Write-Source 'src/Area/F.cs' $shifted
        Write-Source 'src/Area/New.cs' @('namespace Fixture { public class New {', '    public int A() { return 1; }', '    public int B() { return 2; }', '    public int C() { return 3; }', '} }')
        Write-Source 'src/Area/Churn.cs' @('namespace Fixture { public class Churn {', '    public int Get() { return 2; }', '} }')
        Invoke-Git add --all
        Invoke-Git commit --quiet -m 'commit 2'
        $c2 = Get-Head

        # A nightly's digest, through the real digest writer: coverage XML in, digest with its provenance out.
        $writeDigest = { param([string] $Shard, [string] $Sha, [long] $RunId, [hashtable] $Lines, [string] $Directory, [string] $Filter = '')
            $entry = @(Invoke-Git show "${Sha}:.github/test-shards.yml" | yq -o=json -I=0 '.shard' '-' | ConvertFrom-Json) | Where-Object name -ceq $Shard
            if ($Filter) { $entry = Copy-Json $entry; $entry.filter = $Filter }
            $xml = [System.Text.StringBuilder]::new('<CoverageSession><Modules><Module><Files>')
            $ids = @{}
            foreach ($path in $Lines.Keys) { $ids[$path] = $ids.Count + 1; [void] $xml.Append("<File uid=`"$($ids[$path])`" fullPath=`"/runner/work/$path`" />") }
            [void] $xml.Append('</Files><Classes><Class><Methods><Method><SequencePoints>')
            foreach ($path in $Lines.Keys) { foreach ($line in $Lines[$path]) { [void] $xml.Append("<SequencePoint vc=`"1`" sl=`"$line`" fileid=`"$($ids[$path])`" />") } }
            [void] $xml.Append('</SequencePoints></Method></Methods></Class></Classes></Module></Modules></CoverageSession>')
            New-Item -ItemType Directory -Path $Directory -Force | Out-Null
            $coverage = Join-Path $Directory "$Shard.coverage.xml"
            Set-Content -LiteralPath $coverage -Value $xml.ToString() -Encoding utf8
            & $digestWriter -CoverageXml $coverage -Shard $Shard -OutFile (Join-Path $Directory "$Shard.digest.json") -MeasuredSha $Sha -RunId "$RunId" `
                -RunAttempt 1 -ShardDefinitionJson ($entry | ConvertTo-Json -Compress) `
                -ManifestBlob ((Invoke-Git rev-parse "${Sha}:.github/test-shards.yml") | Out-String).Trim() 6>$null
            Remove-Item -LiteralPath $coverage
        }
        $nightTwo = Join-Path $out 'digests-2002'
        & $writeDigest 'Late' $c2 2002 @{ 'src/Area/F.cs' = @(40, 41, 42); 'src/Area/Other.cs' = @(2); 'src/Area/Churn.cs' = @(2) } $nightTwo
        & $writeDigest 'Moved' $c2 2002 @{ 'src/Area/F.cs' = @(60, 61, 62); 'src/Area/New.cs' = @(2, 3, 4) } $nightTwo
        $lateDigest = Get-Content -LiteralPath (Join-Path $nightTwo 'Late.digest.json') -Raw | ConvertFrom-Json
        Assert-True ($lateDigest.measuredSha -ceq $c2 -and [long] $lateDigest.runId -eq 2002 -and $lateDigest.definitionHash -cmatch '^[0-9a-f]{64}$' -and
            $lateDigest.manifestBlob -cmatch '^[0-9a-f]{40}$' -and ($lateDigest.files.'src/Area/F.cs' -join ',') -ceq '40,42') 'the digest writer did not record the measured commit, run, definition hash and manifest blob beside the ranges'
        $plainDirectory = Join-Path $out 'digest-plain'
        New-Item -ItemType Directory -Path $plainDirectory -Force | Out-Null
        '<CoverageSession><File uid="1" fullPath="/w/src/Area/F.cs" /><SequencePoint vc="1" sl="3" fileid="1" /></CoverageSession>' | Set-Content -LiteralPath (Join-Path $plainDirectory 'c.xml') -Encoding utf8
        & $digestWriter -CoverageXml (Join-Path $plainDirectory 'c.xml') -Shard 'Late' -OutFile (Join-Path $plainDirectory 'Late.digest.json') 6>$null
        $plain = Get-Content -LiteralPath (Join-Path $plainDirectory 'Late.digest.json') -Raw | ConvertFrom-Json
        Assert-True ((@($plain.PSObject.Properties.Name) -join ',') -ceq 'schemaVersion,shard,generatedUtc,fileCount,files') "a digest written without provenance has other properties than before ('$(@($plain.PSObject.Properties.Name) -join ',')')"

        $successAll = { param([string[]] $Failed = @()) @($shardNames | ForEach-Object { [ordered]@{ shard = $_; outcome = $(if ($_ -cin $Failed) { 'failure' } else { 'success' }) } }) }
        $sourceTwo = [ordered]@{ runId = 2002; sha = $c2; createdAt = [DateTimeOffset]::UtcNow.AddDays(-3).ToString('o'); outcomes = (& $successAll); digestDirectory = $nightTwo }
        $sourcesTwo = Join-Path $out 'sources-2.json'
        ConvertTo-Json -InputObject @($sourceTwo) -Depth 6 | Set-Content -LiteralPath $sourcesTwo -Encoding utf8
        $rollingTwo = Join-Path $out 'rolling-2'
        & $mapBuilder -AnchorMapFile $anchorFile -AnchorCertificateFile $anchorCertificateFile -AnchorRunId 1001 -OutDirectory $rollingTwo `
            -SourcesFile $sourcesTwo -WorkDirectory (Join-Path $out 'work-2') 6>$null
        Assert-True ($LASTEXITCODE -eq 0) "the rolling map builder exited $LASTEXITCODE"
        $rollingTwoFile = Join-Path $rollingTwo 'rolling-shard-map.json'
        $decisionTwo = Get-Content -LiteralPath (Join-Path $rollingTwo 'rolling-decision.json') -Raw | ConvertFrom-Json
        Assert-True ($decisionTwo.disposition -ceq 'Certified' -and (Test-Path -LiteralPath (Join-Path $rollingTwo 'rolling-certification.json'))) "the first rolling map was not certified ($($decisionTwo.disposition): $($decisionTwo.reason))"
        $mapTwo = Get-Content -LiteralPath $rollingTwoFile -Raw | ConvertFrom-Json
        Assert-True ((@($mapTwo.layers | ForEach-Object { "$($_.sha.Substring(0, 7)):$(@($_.map.knownShards) -join '+')" }) -join ' ') -ceq "$($c1.Substring(0, 7)):Early+Shifted $($c2.Substring(0, 7)):Late+Moved") 'the first rolling map does not hold Early and Shifted at commit 1 and Late and Moved at commit 2'
        Assert-True ($decisionTwo.certificate.wouldSkip -ge 1 -and $decisionTwo.certificate.missCount -eq 0 -and [long] $decisionTwo.certificate.auditSourceRunId -eq 2002) 'the first rolling map''s certificate does not record a reducing, miss-free replay of the nightly'
        Assert-True ((Invoke-Git status --porcelain | Out-String).Trim() -ceq '' -and (Get-Head) -ceq $c2) 'building the rolling map moved or dirtied the checkout it ran in'

        # A rolling map with a single layer, by hand: it must select exactly what the anchor selects.
        $oneLayerFile = Join-Path $out 'one-layer.json'
        $anchorObject = Get-Content -LiteralPath $anchorFile -Raw | ConvertFrom-Json
        $provenance = [ordered]@{}
        foreach ($name in $anchorObject.knownShards) { $provenance[$name] = [ordered]@{ sha = $c1; runId = 1001; source = 'anchor' } }
        [ordered]@{
            schemaVersion = 1; kind = 'rolling'; anchor = [ordered]@{ sha = $c1; runId = 1001 }; alwaysRun = @('Always'); shards = $provenance
            layers = @([ordered]@{ sha = $c1; map = [ordered]@{ schemaVersion = 1; sha = $c1; knownShards = $anchorObject.knownShards; alwaysRun = @(); files = $anchorObject.files } })
        } | ConvertTo-Json -Depth 10 -Compress | Set-Content -LiteralPath $oneLayerFile -Encoding utf8

        # One pull request per case: a branch off commit 2 and the merge commit GitHub would test.
        $pullRequest = { param([string] $Name, [string] $Base, [string] $Path, [int] $Line, [string] $Text)
            Invoke-Git checkout --quiet -B "pr-$Name" $Base
            $lines = [string[]] (Get-Content -LiteralPath (Join-Path $repo $Path))
            $lines[$Line - 1] = $Text
            Write-Source $Path $lines
            Invoke-Git add --all
            Invoke-Git commit --quiet -m "change $Name"
            $head = Get-Head
            Invoke-Git checkout --quiet --detach $Base
            Invoke-Git merge --quiet --no-ff -m "merge $Name" "pr-$Name"
            return $head
        }
        $select = { param([string] $Map, [string] $Head, [string] $Name)
            $file = Join-Path $out "selection-$Name.json"
            & $selector -MapFile $Map -PullRequestHeadSha $Head -ShardManifestFile $manifestFile -ExpectedShards $shardNames -OutFile $file 6>$null
            Assert-True ($LASTEXITCODE -eq 0) "$Name`: the selector exited $LASTEXITCODE"
            return Get-Content -LiteralPath $file -Raw | ConvertFrom-Json
        }
        $cases = @(
            # THE MIXED-COORDINATE CASE. Line 41 at commit 2 is block P, which is line 21 at commit 1. Shifted
            # (measured at 1, lines 20-22) and Late (measured at 2, lines 40-42) execute it; Early's 40-42 are
            # commit 1's numbers for other code. Carrying the edit to commit 1 only selects Shifted,Late,Moved;
            # leaving it at commit 2 only selects Early,Late and misses Shifted.
            @{ Name = 'moved'; Path = 'src/Area/F.cs'; Line = 41; Text = '    public int M021() { return 210; }'; Rolling = 'Always,Late,Shifted'; Anchor = 'Always,Late,Shifted' },
            # Added after commit 1: Moved executes it at commit 2; Early and Shifted were measured before it
            # existed, so the unmapped-file rule (the directory's owners) applies to them, and only to them.
            @{ Name = 'added'; Path = 'src/Area/New.cs'; Line = 3; Text = '    public int B() { return 20; }'; Rolling = 'Always,Early,Moved,Shifted'; Anchor = 'Always,Early,Late,Moved,Shifted' },
            # Present and unchanged since commit 1, executed by nothing there, by Late at commit 2.
            @{ Name = 'settled'; Path = 'src/Area/Other.cs'; Line = 2; Text = '    public int Get() { return 10; }'; Rolling = 'Always,Late'; Anchor = 'Always,Early,Late,Moved,Shifted' },
            # Present at commit 1 but edited on the base branch since: what commit 1's shards make of the new
            # content is unknown, so the unmapped-file rule applies to them.
            @{ Name = 'churned'; Path = 'src/Area/Churn.cs'; Line = 2; Text = '    public int Get() { return 20; }'; Rolling = 'Always,Early,Late,Shifted'; Anchor = 'Always,Early,Late,Moved,Shifted' }
        )
        $heads = @{}
        foreach ($case in $cases) {
            $heads[$case.Name] = & $pullRequest $case.Name $c2 $case.Path $case.Line $case.Text
            $viaAnchor = & $select $anchorFile $heads[$case.Name] "$($case.Name)-anchor"
            $viaOneLayer = & $select $oneLayerFile $heads[$case.Name] "$($case.Name)-one-layer"
            $viaRolling = & $select $rollingTwoFile $heads[$case.Name] "$($case.Name)-rolling"
            Assert-True (-not [bool] $viaRolling.escalate -and (Get-Names $viaRolling) -ceq $case.Rolling) "$($case.Name): the rolling map selected '$(Get-Names $viaRolling)' (escalated: $($viaRolling.escalate)), not '$($case.Rolling)'"
            Assert-True (-not [bool] $viaAnchor.escalate -and (Get-Names $viaAnchor) -ceq $case.Anchor) "$($case.Name): the weekly map selected '$(Get-Names $viaAnchor)', not '$($case.Anchor)'"
            Assert-True ([bool] $viaOneLayer.escalate -eq [bool] $viaAnchor.escalate -and (Get-Names $viaOneLayer) -ceq (Get-Names $viaAnchor) -and
                [string] $viaOneLayer.reason -ceq [string] $viaAnchor.reason -and [bool] $viaOneLayer.requiresShards -eq [bool] $viaAnchor.requiresShards -and
                (@($viaOneLayer.routes | Sort-Object) -join '|') -ceq (@($viaAnchor.routes | Sort-Object) -join '|') -and
                (@($viaOneLayer.reasons) -join '|') -ceq (@($viaAnchor.reasons) -join '|')) "$($case.Name): a one-layer rolling map did not select exactly what the map it was built from selects"
            Assert-True (@($viaRolling.rolling.layers).Count -eq 2 -and [string] $viaRolling.map.sha -ceq $c1) "$($case.Name): the rolling selection does not report its two measured commits"
        }

        # ---- shadow: both selections recorded, the weekly map decides, its selection file is not touched.
        Invoke-Git checkout --quiet --detach $c2
        Invoke-Git merge --quiet --no-ff -m 'merge added' 'pr-added'
        $selectionFile = Join-Path $out 'selection.json'
        $rollingSelectionFile = Join-Path $out 'selection-rolling.json'
        $anchorCopyFile = Join-Path $out 'selection-anchor.json'
        $comparisonFile = Join-Path $out 'rolling-comparison.json'
        $decide = { param([string] $Mode, [string] $Directory, [long] $AnchorRunId = 1001)
            Remove-Item -LiteralPath $selectionFile, $rollingSelectionFile, $anchorCopyFile, $comparisonFile -ErrorAction SilentlyContinue
            & $selector -MapFile $anchorFile -PullRequestHeadSha $heads['added'] -ShardManifestFile $manifestFile -ExpectedShards $shardNames -OutFile $selectionFile 6>$null
            $before = (Get-FileHash -LiteralPath $selectionFile -Algorithm SHA256).Hash
            & $rollingSelector -Mode $Mode -AnchorMapFile $anchorFile -AnchorRunId $AnchorRunId -RollingDirectory $Directory -AnchorSelectionFile $selectionFile `
                -PullRequestHeadSha $heads['added'] -ShardManifestFile $manifestFile -ExpectedShards $shardNames `
                -RollingSelectionFile $rollingSelectionFile -AnchorCopyFile $anchorCopyFile -ComparisonFile $comparisonFile 6>$null
            Assert-True ($LASTEXITCODE -eq 0) "Invoke-RollingSelection ($Mode) exited $LASTEXITCODE"
            return [pscustomobject]@{
                Untouched = $before -ceq (Get-FileHash -LiteralPath $selectionFile -Algorithm SHA256).Hash
                Selection = Get-Content -LiteralPath $selectionFile -Raw | ConvertFrom-Json
                Comparison = $(if (Test-Path -LiteralPath $comparisonFile) { Get-Content -LiteralPath $comparisonFile -Raw | ConvertFrom-Json } else { $null })
            }
        }
        foreach ($offValue in '', 'off', 'enabled') {
            $off = & $decide $offValue $rollingTwo
            Assert-True ($off.Untouched -and $null -eq $off.Comparison -and -not (Test-Path -LiteralPath $rollingSelectionFile)) "with TEST_IMPACT_ROLLING_MAP='$offValue' the rolling path did something"
        }
        $shadowRun = & $decide 'shadow' $rollingTwo
        Assert-True ($shadowRun.Untouched -and (Get-Names $shadowRun.Selection) -ceq 'Always,Early,Late,Moved,Shifted') 'shadow mode changed the weekly map''s selection file'
        Assert-True ($shadowRun.Comparison.decides -ceq 'anchor' -and $shadowRun.Comparison.mode -ceq 'shadow' -and
            (@($shadowRun.Comparison.comparison.rolling.shards | Sort-Object) -join ',') -ceq 'Always,Early,Moved,Shifted' -and
            (@($shadowRun.Comparison.comparison.anchor.shards | Sort-Object) -join ',') -ceq 'Always,Early,Late,Moved,Shifted' -and
            (@($shadowRun.Comparison.comparison.droppedByRolling) -join ',') -ceq 'Late' -and @($shadowRun.Comparison.comparison.addedByRolling).Count -eq 0) 'shadow mode did not record both selections and the shard the rolling map would drop'
        Assert-True (Test-Path -LiteralPath $rollingSelectionFile) 'shadow mode did not keep the rolling selection'
        $summary = Join-Path $out 'summary.md'
        & (Join-Path $PSScriptRoot 'Write-SelectionReport.ps1') -SelectionFile $selectionFile -TotalShards 5 -RollingComparisonFile $comparisonFile -SummaryFile $summary -ReportFile (Join-Path $out 'report.json')
        Assert-True ($LASTEXITCODE -eq 0 -and (Get-Content -LiteralPath $summary -Raw) -match 'Rolling coverage map \(shadow mode\)' -and (Get-Content -LiteralPath $summary -Raw) -match 'would not run \(1\)</summary>\s+- Late') 'the selection report did not render the shadow comparison'
        # After the matrix: the shard the rolling map would have dropped failed.
        $jobsFile = Join-Path $out 'jobs.json'
        ConvertTo-Json -InputObject @($shardNames | ForEach-Object { [ordered]@{ name = "Tests (net10.0) - $_"; conclusion = $(if ($_ -ceq 'Late') { 'failure' } else { 'success' }) } }) | Set-Content -LiteralPath $jobsFile -Encoding utf8
        $outcomeFile = Join-Path $out 'rolling-shadow-outcome.json'
        & $rollingSelector -OutcomeComparisonFile $comparisonFile -JobsFile $jobsFile -OutcomeFile $outcomeFile -SummaryFile $summary 6>$null
        Assert-True ((@((Get-Content -LiteralPath $outcomeFile -Raw | ConvertFrom-Json).droppedThenFailed) -join ',') -ceq 'Late' -and (Get-Content -LiteralPath $summary -Raw) -match 'would NOT have run failed') 'a shard the rolling map would have dropped that then failed was not reported after the matrix'

        # ---- on: a certified rolling map decides; the weekly map's selection is kept beside it.
        $onRun = & $decide 'on' $rollingTwo
        Assert-True ($onRun.Comparison.decides -ceq 'rolling' -and (Get-Names $onRun.Selection) -ceq 'Always,Early,Moved,Shifted' -and
            (Get-Names (Get-Content -LiteralPath $anchorCopyFile -Raw | ConvertFrom-Json)) -ceq 'Always,Early,Late,Moved,Shifted') 'on mode with a certified rolling map did not hand the matrix the rolling selection and keep the weekly one'

        # ---- every certification failure falls back to the weekly map's selection, byte for byte.
        $broken = { param([string] $Name, [scriptblock] $Break)
            $directory = Join-Path $out "rolling-broken-$Name"
            Copy-Item -LiteralPath $rollingTwo -Destination $directory -Recurse
            & $Break $directory
            return $directory
        }
        foreach ($case in @(
            @{ What = 'no certificate'; Pattern = 'published no certified rolling map'; Directory = (& $broken 'no-certificate' { param($d) Remove-Item -LiteralPath (Join-Path $d 'rolling-certification.json') }) },
            @{ What = 'map bytes changed after certification'; Pattern = 'does not match the rolling map bytes'; Directory = (& $broken 'bytes' { param($d) Add-Content -LiteralPath (Join-Path $d 'rolling-shard-map.json') -Value ' ' }) },
            @{ What = 'a certificate recording a miss'; Pattern = 'selection miss|disposition'; Directory = (& $broken 'miss' { param($d)
                $file = Join-Path $d 'rolling-certification.json'; $c = Get-Content -LiteralPath $file -Raw | ConvertFrom-Json; $c.missCount = 1; $c | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $file -Encoding utf8 }) },
            @{ What = 'no rolling artifact at all'; Pattern = 'published no certified rolling map'; Directory = (Join-Path $out 'rolling-absent') },
            # The anchor in use is another map run: the one this rolling map was built on was superseded or revoked.
            @{ What = 'an anchor that is no longer the one in use'; Pattern = 'not the anchor in use'; Directory = $rollingTwo; AnchorRunId = 1002 }
        )) {
            $anchorRun = if ($case.ContainsKey('AnchorRunId')) { $case.AnchorRunId } else { 1001 }
            $fallback = & $decide 'on' $case.Directory $anchorRun
            Assert-True ($fallback.Untouched -and $fallback.Comparison.decides -ceq 'anchor' -and (Get-Names $fallback.Selection) -ceq 'Always,Early,Late,Moved,Shifted') "on mode with $($case.What) did not leave the weekly map's selection exactly as it was"
            Assert-True ([string] $fallback.Comparison.rollingProblem -match $case.Pattern) "on mode with $($case.What) fell back for another reason: $($fallback.Comparison.rollingProblem)"
            Assert-True (-not (Test-Path -LiteralPath $anchorCopyFile)) "on mode with $($case.What) behaved as if the rolling map had decided"
        }

        # ---- commit 3 and a second nightly: one failed shard job with a digest, one digest that ran another definition.
        Invoke-Git checkout --quiet --detach $c2
        $third = [string[]] $shifted.Clone()
        $third[60] = '    public int M041() { return 410; }'
        Write-Source 'src/Area/F.cs' $third
        Invoke-Git add --all
        Invoke-Git commit --quiet -m 'commit 3'
        $c3 = Get-Head
        $nightThree = Join-Path $out 'digests-2003'
        & $writeDigest 'Shifted' $c3 2003 @{ 'src/Area/F.cs' = @(40, 41, 42) } $nightThree
        & $writeDigest 'Early' $c3 2003 @{ 'src/Area/F.cs' = @(60, 61, 62) } $nightThree
        & $writeDigest 'Moved' $c3 2003 @{ 'src/Area/F.cs' = @(60, 61, 62) } $nightThree 'FullyQualifiedName~SomethingElse'
        $buildThree = { param([string] $Name, [string[]] $Failed)
            $sources = Join-Path $out "sources-$Name.json"
            ConvertTo-Json -Depth 6 -InputObject @($sourceTwo, [ordered]@{
                runId = 2003; sha = $c3; createdAt = [DateTimeOffset]::UtcNow.AddDays(-2).ToString('o'); outcomes = (& $successAll $Failed); digestDirectory = $nightThree
            }) | Set-Content -LiteralPath $sources -Encoding utf8
            $directory = Join-Path $out "rolling-$Name"
            & $mapBuilder -AnchorMapFile $anchorFile -AnchorCertificateFile $anchorCertificateFile -AnchorRunId 1001 -OutDirectory $directory `
                -SourcesFile $sources -WorkDirectory (Join-Path $out "work-$Name") 6>$null
            Assert-True ($LASTEXITCODE -eq 0) "the rolling map builder ($Name) exited $LASTEXITCODE"
            return $directory
        }
        $rollingThree = & $buildThree '3' @('Early')
        $decisionThree = Get-Content -LiteralPath (Join-Path $rollingThree 'rolling-decision.json') -Raw | ConvertFrom-Json
        $mapThree = Get-Content -LiteralPath (Join-Path $rollingThree 'rolling-shard-map.json') -Raw | ConvertFrom-Json
        Assert-True ((@($mapThree.layers | ForEach-Object { @($_.map.knownShards) -join '+' }) -join ' ') -ceq 'Early Late+Moved Shifted') "the second rolling map's layers are '$(@($mapThree.layers | ForEach-Object { @($_.map.knownShards) -join '+' }) -join ' ')': the failed job's digest or the redefined one got in, or the valid one did not"
        Assert-True (@($decisionThree.rejected | Where-Object { $_.shard -ceq 'Early' -and $_.reason -match 'did not complete successfully \(failure\)' }).Count -eq 1) 'the digest from the failed shard job was not recorded as refused for that reason'
        Assert-True (@($decisionThree.rejected | Where-Object { $_.shard -ceq 'Moved' -and $_.reason -match 'differs from the manifest at its commit' }).Count -eq 1) 'the digest measured with another definition was not recorded as refused for that reason'
        Assert-True ($decisionThree.disposition -ceq 'Certified' -and [long] $decisionThree.certificate.auditSourceRunId -eq 2003 -and $decisionThree.certificate.failedShards -eq 1) "the second rolling map was not certified against the newest complete nightly ($($decisionThree.disposition): $($decisionThree.reason))"
        # The same nightly, but a shard the rolling map would skip there failed: no certificate, the weekly map decides.
        $rollingMiss = & $buildThree 'miss' @('Early', 'Late')
        $decisionMiss = Get-Content -LiteralPath (Join-Path $rollingMiss 'rolling-decision.json') -Raw | ConvertFrom-Json
        Assert-True ($decisionMiss.disposition -ceq 'RevokedMiss' -and (@($decisionMiss.certificate.missed) -join ',') -ceq 'Late' -and
            -not (Test-Path -LiteralPath (Join-Path $rollingMiss 'rolling-certification.json'))) "a rolling map that would have skipped a failed shard was not refused ($($decisionMiss.disposition))"
        Invoke-Git checkout --quiet --detach $c2
        Invoke-Git merge --quiet --no-ff -m 'merge added' 'pr-added'
        $refused = & $decide 'on' $rollingMiss
        Assert-True ($refused.Untouched -and $refused.Comparison.decides -ceq 'anchor' -and [string] $refused.Comparison.rollingProblem -match 'RevokedMiss') 'a rolling map refused for a miss was used in on mode'
        # A nightly that did not finish every shard job cannot be replayed against.
        $unfinished = Join-Path $out 'sources-unfinished.json'
        $cancelled = Copy-Json $sourceTwo
        ($cancelled.outcomes | Where-Object shard -ceq 'Always').outcome = 'cancelled'
        ConvertTo-Json -InputObject @($cancelled) -Depth 6 | Set-Content -LiteralPath $unfinished -Encoding utf8
        & $mapBuilder -AnchorMapFile $anchorFile -AnchorCertificateFile $anchorCertificateFile -AnchorRunId 1001 -OutDirectory (Join-Path $out 'rolling-unfinished') `
            -SourcesFile $unfinished -WorkDirectory (Join-Path $out 'work-unfinished') 6>$null
        Assert-True ((Get-Content -LiteralPath (Join-Path $out 'rolling-unfinished/rolling-decision.json') -Raw | ConvertFrom-Json).disposition -ceq 'NoCompleteNightly' -and
            -not (Test-Path -LiteralPath (Join-Path $out 'rolling-unfinished/rolling-certification.json'))) 'a rolling map was certified against a nightly with an unfinished shard job'

        # ---- commit 4 redefines Moved after its digest: it runs until measured again, whatever its index says.
        Invoke-Git checkout --quiet --detach $c3
        Write-Source '.github/test-shards.yml' (& $manifestYaml @{ Moved = 'FullyQualifiedName~Moved|FullyQualifiedName~Extra' })
        Invoke-Git add --all
        Invoke-Git commit --quiet -m 'commit 4'
        $c4 = Get-Head
        & $writeManifest
        $headRedefined = & $pullRequest 'redefined' $c4 'src/Area/F.cs' 41 '    public int M021() { return 211; }'
        $viaThree = & $select (Join-Path $rollingThree 'rolling-shard-map.json') $headRedefined 'redefined-rolling'
        Assert-True (-not [bool] $viaThree.escalate -and @($viaThree.shards) -ccontains 'Moved' -and @($viaThree.shards) -ccontains 'Late' -and @($viaThree.shards) -ccontains 'Shifted' -and
            @($viaThree.routes | Where-Object { $_ -like 'Moved <= changed its manifest definition*' }).Count -eq 1) "a shard redefined after its digest was not run ('$(Get-Names $viaThree)')"
        Assert-True (@($viaThree.rolling.layers).Count -eq 3) 'the selection against three measured commits does not report three'

        # ---- the nightly's budget selection, from this repository's state.
        $shardScriptCopy = Join-Path $out 'Invoke-Shard.ps1'
        @('$heavyShards = @(', "  'Early',", "  'Shifted',", "  'Late',", "  'Moved',", "  'Always'", ')', '$performanceShards = @(', "  'Timing'", ')') | Set-Content -LiteralPath $shardScriptCopy -Encoding utf8
        $planFile = Join-Path $out 'rolling-source.json'
        & $budgetSelector -ShardManifestFile $manifestFile -AnchorMapFile $anchorFile -RollingMapFile (Join-Path $rollingThree 'rolling-shard-map.json') `
            -ShardScriptPath $shardScriptCopy -BudgetMinutes '50' -OutFile $planFile 6>$null
        $nightPlan = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
        # Moved is redefined (first); then the stalest: Early is still at commit 1, Late at commit 2, Shifted at 3.
        Assert-True ((@($nightPlan.instrument) -join ',') -ceq 'Moved,Early' -and $nightPlan.predictedExtraMinutes -eq 50 -and
            @($nightPlan.excluded | Where-Object name -ceq 'Always').Count -eq 1) "tonight's plan was '$(@($nightPlan.instrument) -join ',')', not the redefined shard and then the one measured longest ago, within 50 minutes"
    }
    finally { Pop-Location }
}
catch { [void] $failures.Add("the real-commit cases could not run: $($_.Exception.Message) at line $($_.InvocationInfo.ScriptLineNumber)") }
finally {
    $env:TEST_IMPACT_PRECISE_FALLBACKS = $savedSwitch
    if ((Split-Path -Leaf $root) -like 'aidotnet-rolling-e2e-*') { Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue }
}

if ($failures.Count -gt 0) {
    Write-Host 'Rolling shard map proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}
Write-Host 'Rolling shard map proof passed: per-commit translation selected exactly the executing shards; a one-layer rolling map equals its anchor; refused digests, a missed failure, an unfinished nightly and a superseded anchor all left the weekly map deciding.'
exit 0
