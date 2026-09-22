<#
.SYNOPSIS
    Ratchet on how much CI a certified shard map can actually skip.

.DESCRIPTION
    Every other proof here checks that selection is CORRECT - that it never skips a shard it should
    have run. None checks that it skips anything at all, and that is the gap this closes. The map
    certified on 2026-09-21 (run 35628474741) was correct and still left a one-file model change
    selecting 52 of 164 shards, because 51 of them were unmapped and therefore always-run. A
    correctness proof cannot see that; it looks identical to a healthy map.

    So the number that matters is the size of the always-run set, and it is pinned here. It is the
    direct cause of the residual cost: a change the map implicates in ONE shard still pays for
    every always-run entry, whatever the selector does.

    This ratchet only ever moves DOWN. Raising a baseline is not a fix - it is the record of a
    regression, and the reason the previous attempts at this problem each looked green and changed
    nothing.

.PARAMETER MapFile
    A shard-map.json to measure. Normally the freshly built candidate in the map workflow.

.PARAMETER SelfTest
    Runs the built-in checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Measure')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Measure')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

<#
    Always-run baseline history. Append, never overwrite: the trail is the evidence.

    51  2026-09-21  first measurement, from the map certified by run 35628474741.
                    46 are the mustCover Conformance/Sweep shards whose worker was never linked
                    (Connect-WorkerCoverage.ps1 existed but nothing called it, fixed in efdb960edd);
                    5 are heavy model shards held by the no-coverage latch (escape hatch added in
                    0eb5ef3ed6). Both fixes are expected to drop this sharply once a
                    coverage-everywhere run feeds a new map - at which point this constant comes
                    down to whatever that run actually proves, not to whatever was hoped for.
#>
$script:AlwaysRunBaseline = 51

function Measure-MapReduction {
    <# Pure: given a parsed map, report its selectable/always-run split. #>
    param([Parameter(Mandatory)] $Map)

    foreach ($name in 'knownShards', 'alwaysRun') {
        if (-not $Map.PSObject.Properties[$name]) { throw "map is missing $name" }
    }
    if ($Map.knownShards -isnot [array] -or $Map.alwaysRun -isnot [array]) {
        throw 'map knownShards and alwaysRun must be arrays'
    }
    $retired = @(if ($Map.PSObject.Properties['retiredShards']) { @($Map.retiredShards) } else { @() })
    # A retired slot is not a live shard and must not flatter the measurement.
    $indexed = @(@($Map.knownShards) | Where-Object { $_ -cnotin $retired })
    $always = @(@($Map.alwaysRun) | Where-Object { $_ -cnotin $retired })
    $total = $indexed.Count + $always.Count
    if ($total -lt 1) { throw 'map contains no live shards' }
    return [pscustomobject]@{
        Mapped    = $indexed.Count
        AlwaysRun = $always.Count
        Total     = $total
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

    $m = Measure-MapReduction ([pscustomobject]@{
        knownShards = @('A', 'B', 'C'); alwaysRun = @('Heavy')
    })
    Assert-True ($m.Mapped -eq 3 -and $m.AlwaysRun -eq 1 -and $m.Total -eq 4) `
        'the mapped/always-run split was not reported correctly'

    # A retired slot is dead weight, not an always-run cost, and not a mapped shard either.
    $m = Measure-MapReduction ([pscustomobject]@{
        knownShards = @('A', 'B', 'Retired'); alwaysRun = @('Heavy'); retiredShards = @('Retired')
    })
    Assert-True ($m.Mapped -eq 2 -and $m.AlwaysRun -eq 1 -and $m.Total -eq 3) `
        'a retired slot was counted as a live shard'

    Assert-Rejected { Measure-MapReduction ([pscustomobject]@{ knownShards = @('A') }) } `
        'a map without alwaysRun was measured instead of rejected'
    Assert-Rejected { Measure-MapReduction ([pscustomobject]@{ knownShards = 'A'; alwaysRun = @() }) } `
        'a non-array knownShards was accepted'
    Assert-Rejected { Measure-MapReduction ([pscustomobject]@{
        knownShards = @('X'); alwaysRun = @(); retiredShards = @('X') }) } `
        'a map whose only shard is retired was measured instead of rejected'

    # The ratchet must actually bite. A map one over the baseline has to be rejected, or this
    # proof is decoration - which is exactly what the existing correctness proofs were for this.
    $over = $script:AlwaysRunBaseline + 1
    $m = Measure-MapReduction ([pscustomobject]@{
        knownShards = @('A')
        alwaysRun   = @(0..($over - 1) | ForEach-Object { "Heavy$_" })
    })
    Assert-True ($m.AlwaysRun -gt $script:AlwaysRunBaseline) `
        'a map exceeding the always-run baseline did not measure as exceeding it'

    if ($failures.Count -gt 0) {
        Write-Host 'Selection reduction self-test FAILED:'
        foreach ($f in $failures) { Write-Host "  - $f" }
        exit 1
    }
    Write-Host 'Selection reduction self-test passed.'
    exit 0
}

$map = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
$measured = Measure-MapReduction -Map $map
Write-Host ("selection reduction: {0} mapped, {1} always-run, {2} live shard(s)" -f
    $measured.Mapped, $measured.AlwaysRun, $measured.Total)

if ($measured.AlwaysRun -gt $script:AlwaysRunBaseline) {
    # One string, then -f. Concatenating first and formatting after binds -f to the LAST segment
    # only, which printed a literal "{0}" in the very message meant to name the regression.
    $message = '::error::always-run shards rose to {0}, above the baseline of {1}. Every one of ' +
        'them runs on every pull request whatever the change touches. Fix the coverage gap that ' +
        'made them unmappable; do not raise the baseline.'
    Write-Host ($message -f $measured.AlwaysRun, $script:AlwaysRunBaseline)
    exit 1
}
if ($measured.AlwaysRun -lt $script:AlwaysRunBaseline) {
    $message = '::notice::always-run shards fell to {0}, below the baseline of {1} - lower ' +
        'AlwaysRunBaseline in Test-SelectionReduction.ps1 to hold the gain.'
    Write-Host ($message -f $measured.AlwaysRun, $script:AlwaysRunBaseline)
}
exit 0
