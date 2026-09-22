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

.PARAMETER ShardManifestFile
    The shard manifest, so the always-run set can be costed in TESTS rather than in shards. Shard
    count understates the damage: the always-run shards are not evenly sized, and it is the tests
    inside them that a pull request actually waits for. Omit to check the shard count only.

.PARAMETER TestsRoot
    Where to count [Fact]/[Theory] methods. Defaults to tests/.

.PARAMETER SelfTest
    Runs the built-in checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Measure')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Measure')] [string] $MapFile,
    [Parameter(ParameterSetName = 'Measure')] [string] $ShardManifestFile,
    [Parameter(ParameterSetName = 'Measure')] [string] $TestsRoot = 'tests',
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

<#
    Tests every pull request executes regardless of what it changed - the sum over the always-run
    shards. This is the number a one-line docs fix actually waits for, and shard count hides it
    because the always-run shards are wildly uneven in size.

    6109  2026-09-22  WRONG, superseded same day. Derived from a hand-rolled filter matcher that
                      required every FullyQualifiedName term to match, which treats the '|' in 83
                      of the 164 filters as AND, and which ignored each shard's project.
    5412  2026-09-22  Measured with the repo's own ConvertTo-TestFilter and Test-TestFilter plus a
                      project match - the same evaluator selection uses - against the map certified
                      by run 35719826020. 15.7% of the suite's 34,421 [Fact]/[Theory] methods, from
                      51 always-run shards. 46 of those are the mustCover worker shards whose
                      instrumented binary was never linked (efdb960edd) and 5 are held by the
                      no-coverage latch (0eb5ef3ed6); both fixes should cut this once a coverage
                      run re-bases the map, at which point this comes down to what that run proves.
#>
$script:AlwaysRunTestBaseline = 5412

# Load Select-Shards' declarations - ConvertTo-TestFilter, Test-TestFilter and the $script:Filter*
# constants - without its parameter block or self-test, so this measures with exactly the parser
# and evaluator selection uses rather than a second implementation of the same grammar. Dot-sourced
# at script scope on purpose: doing it inside a function would scope the definitions to that call.
$script:SelectorLines = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'Select-Shards.ps1')
$script:SelectorFrom = ($script:SelectorLines | Select-String -Pattern '^Set-StrictMode' | Select-Object -First 1).LineNumber
$script:SelectorTo = ($script:SelectorLines | Select-String -Pattern '^if \(\$SelfTest\)' | Select-Object -First 1).LineNumber
$script:SelectorFragment = Join-Path ([IO.Path]::GetTempPath()) ("selection-reduction-$([guid]::NewGuid().ToString('N')).ps1")
Set-Content -LiteralPath $script:SelectorFragment `
    -Value $script:SelectorLines[($script:SelectorFrom - 1)..($script:SelectorTo - 2)] -Encoding utf8
. $script:SelectorFragment
Remove-Item -LiteralPath $script:SelectorFragment -Force -ErrorAction SilentlyContinue

function Measure-AlwaysRunTests {
    <# Tests reachable from the always-run shards, via the FullyQualifiedName terms VSTest uses. #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $AlwaysRun,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Manifest,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Units
    )

    # Evaluate each shard's real filter with the parser and evaluator selection itself uses, and
    # require the test's project to be the one the shard runs. A hand-rolled matcher got this wrong
    # three times: matching ANY FullyQualifiedName term over-counts; requiring ALL of them treats
    # the '|' in 83 of the 164 filters as AND and drops 74% of the suite; and skipping the project
    # check lets AiDotNet.Serving.Tests, whose filter carries no name condition, claim all 34,421.
    $counted = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $total = 0
    foreach ($name in $AlwaysRun) {
        $shard = @($Manifest | Where-Object { [string] $_.name -ceq [string] $name })
        if ($shard.Count -eq 0) { continue }
        $tree = $null
        try { $tree = ConvertTo-TestFilter -Text ([string] $shard[0].filter) }
        catch {
            # An unparseable filter must not silently contribute nothing: count the whole project,
            # which is the conservative reading and keeps the ratchet from reporting a false gain.
            $tree = $null
        }
        $projectDirectory = ((([string] $shard[0].project).Replace('\', '/')) -split '/')[1]
        foreach ($u in $Units) {
            if ([string] $u.Project -cne $projectDirectory) { continue }
            if ($null -ne $tree) {
                $candidate = [pscustomobject]@{
                    Fqn = [string] $u.Fqn; PrefixOnly = $false; AnySuffix = $false; Categories = @()
                }
                if ((Test-TestFilter -Node $tree -Candidate $candidate) -eq $script:FilterFalse) { continue }
            }
            # A file two shards both claim is paid for once, not twice.
            if ($counted.Add([string] $u.Fqn)) { $total += [int] $u.Tests }
        }
    }
    return $total
}

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

    # Costing the always-run set in TESTS, which is what a pull request actually waits for.
    $manifest = @(
        [pscustomobject]@{ name = 'Heavy'; project = 'tests/Acme.Tests/Acme.Tests.csproj'; filter = 'FullyQualifiedName~Acme.Heavy' },
        [pscustomobject]@{ name = 'Light'; project = 'tests/Acme.Tests/Acme.Tests.csproj'; filter = 'FullyQualifiedName~Acme.Light' },
        # An OR filter: the hand-rolled matcher this replaced required every term, so it dropped
        # everything a '|' filter selects - 83 of the repo's 164 filters contain one.
        [pscustomobject]@{ name = 'Either'; project = 'tests/Acme.Tests/Acme.Tests.csproj'
                           filter = 'FullyQualifiedName~Acme.Heavy|FullyQualifiedName~Acme.Light' },
        # No name condition at all: without the project check this claims every test everywhere.
        [pscustomobject]@{ name = 'Other'; project = 'tests/Other.Tests/Other.Tests.csproj'; filter = 'Category!=GPU' },
        # A filter the parser rejects. It must cost the whole project, not nothing - a shard that
        # silently contributes zero would hand the ratchet a reduction it has not earned.
        [pscustomobject]@{ name = 'Broken'; project = 'tests/Acme.Tests/Acme.Tests.csproj'; filter = 'Nonsense' }
    )
    $units = @(
        [pscustomobject]@{ Fqn = 'Acme.Heavy.BigTests'; Tests = 100; Project = 'Acme.Tests' },
        [pscustomobject]@{ Fqn = 'Acme.Light.SmallTests'; Tests = 3; Project = 'Acme.Tests' }
    )
    Assert-True ((Measure-AlwaysRunTests -AlwaysRun @('Heavy') -Manifest $manifest -Units $units) -eq 100) `
        'the always-run test cost did not follow the shard filter'
    Assert-True ((Measure-AlwaysRunTests -AlwaysRun @() -Manifest $manifest -Units $units) -eq 0) `
        'an empty always-run set must cost nothing'
    # Shard count cannot stand in for this: two always-run shards can differ by 30x.
    Assert-True ((Measure-AlwaysRunTests -AlwaysRun @('Light') -Manifest $manifest -Units $units) -eq 3) `
        'a small always-run shard was costed as if it were a large one'
    # A file two shards both claim is paid for once. 'Either' overlaps 'Heavy' by construction. The
    # fixture here used to pair 'Heavy' with 'Light' over an FQN only ONE of the two filters
    # matched - 'Acme.Heavy.Light.SharedTests' contains 'Acme.Heavy' but not 'Acme.Light', since
    # VSTest '~' is a substring test - so it passed whether or not the dedup existed.
    $shared = @([pscustomobject]@{ Fqn = 'Acme.Heavy.SharedTests'; Tests = 7; Project = 'Acme.Tests' })
    Assert-True ((Measure-AlwaysRunTests -AlwaysRun @('Heavy', 'Either') -Manifest $manifest -Units $shared) -eq 7) `
        'a test file claimed by two always-run shards was counted twice'
    # An OR filter must select both sides. Requiring every term instead is what lost 74% of the
    # suite when this was measured with a hand-rolled matcher.
    Assert-True ((Measure-AlwaysRunTests -AlwaysRun @('Either') -Manifest $manifest -Units $units) -eq 103) `
        'an OR filter did not select both of its alternatives'
    # A shard for another project must claim nothing here, however permissive its filter.
    Assert-True ((Measure-AlwaysRunTests -AlwaysRun @('Other') -Manifest $manifest -Units $units) -eq 0) `
        'a shard running a different project claimed these tests'
    Assert-True ((Measure-AlwaysRunTests -AlwaysRun @('Broken') -Manifest $manifest -Units $units) -eq 103) `
        'an unparseable filter contributed nothing instead of its whole project'

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
$retiredForReport = @(if ($map.PSObject.Properties['retiredShards']) { @($map.retiredShards) } else { @() })
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

# The shard count is a proxy; this is the cost itself.
if ($ShardManifestFile -and (Test-Path -LiteralPath $ShardManifestFile) -and (Test-Path -LiteralPath $TestsRoot)) {
    $manifest = @(Get-Content -LiteralPath $ShardManifestFile -Raw | ConvertFrom-Json)
    $testsRootFull = (Resolve-Path -LiteralPath $TestsRoot).Path
    $units = @(foreach ($f in @(Get-ChildItem -LiteralPath $TestsRoot -Recurse -Filter *.cs -File)) {
        $text = [IO.File]::ReadAllText($f.FullName)
        $n = ([regex]::Matches($text, '\[\s*(Fact|Theory)\b')).Count
        if ($n -eq 0) { continue }
        $ns = ''
        $m = [regex]::Match($text, '(?m)^\s*namespace\s+([A-Za-z0-9_.]+)')
        if ($m.Success) { $ns = $m.Groups[1].Value }
        $relative = $f.FullName.Substring($testsRootFull.Length).TrimStart('\', '/').Replace('\', '/')
        [pscustomobject]@{
            Fqn = "$ns.$([IO.Path]::GetFileNameWithoutExtension($f.Name))"
            Tests = $n
            Project = ($relative -split '/')[0]
        }
    })
    $always = @(@($map.alwaysRun) | Where-Object { $_ -cnotin $retiredForReport })
    $alwaysTests = Measure-AlwaysRunTests -AlwaysRun $always -Manifest $manifest -Units $units
    Write-Host ("tests every pull request runs regardless of its change: {0}" -f $alwaysTests)

    if ($alwaysTests -gt $script:AlwaysRunTestBaseline) {
        $message = '::error::every pull request now runs {0} tests regardless of what it changed, ' +
            'above the baseline of {1}. Give the always-run shards coverage so they become ' +
            'selectable; do not raise the baseline.'
        Write-Host ($message -f $alwaysTests, $script:AlwaysRunTestBaseline)
        exit 1
    }
    if ($alwaysTests -lt $script:AlwaysRunTestBaseline) {
        $message = '::notice::unconditional tests fell to {0}, below the baseline of {1} - lower ' +
            'AlwaysRunTestBaseline in Test-SelectionReduction.ps1 to hold the gain.'
        Write-Host ($message -f $alwaysTests, $script:AlwaysRunTestBaseline)
    }
}
exit 0
