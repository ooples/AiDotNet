<#
.SYNOPSIS
    Narrows the shard selection to the test classes a change can reach, from the Build job's output.

.DESCRIPTION
    select-shards chooses whole shards from line coverage, and every shard executes the shared base
    classes, so in practice it ran 50-164 of 164 shards on every pull request. This runs in the Build
    job, where the compiled assemblies already are, and asks TypeImpact (tools/TestImpact/TypeImpact)
    which test classes the change can reach through the type-reference graph. Each shard is narrowed
    to the selected classes it owns and a shard owning none is skipped.

    It only ever narrows a decision select-shards has already made to run tests. It passes that
    decision through unchanged when:
      - the event is not a pull request or a push (schedules, dispatches and the coverage run
        feed the nightly and the map, which need everything);
      - a post-merge push is re-running a delta select-shards computed from the pull request run;
      - TypeImpact cannot map every changed file (a non-C# file, a generator input, a source with
        no compiled type) or fails in any way.
    The nightly full matrix on master is the net for what static reachability cannot see
    (reflection over a type name).

    Writes matrix, ledger_matrix, skipped and impact_mode to GITHUB_OUTPUT.
#>
[CmdletBinding()]
param(
    [string] $Repository = (Get-Location).Path,
    # Where the built test output lives; the checkout itself unless testing against another build.
    [string] $BuildRoot = '',
    [string] $PlanFile = 'type-impact-plan.json',
    # What actually runs after this script's post-processing (for the selection report). The plan file lists
    # candidates for every manifest shard; this records the chosen intersection, passthrough, and any narrowing
    # undone to fit the output limit.
    [string] $EffectiveFile = 'type-impact-effective.json',
    # Job outputs are capped at 1 MB in total, and matrix and ledger_matrix carry the same shards;
    # narrowed filters are un-narrowed, largest first, until both together fit under this.
    [int] $MaxMatrixCharacters = 800000
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Output-Value([string] $Name, [string] $Value) {
    "$Name=$Value" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
}

function Write-Effective([string] $Mode, [string] $Why, $Shards) {
    # Report-only and never fatal: the matrix outputs above are what the workflow consumes.
    try {
        $rows = @(foreach ($shard in @($Shards)) {
            $narrowed = [bool] ($shard.PSObject.Properties['narrowed'] -and $shard.narrowed)
            [pscustomobject]@{
                name = [string] $shard.name
                narrowed = $narrowed
                classes = $(if ($narrowed -and $shard.PSObject.Properties['narrowedClasses']) { [int] $shard.narrowedClasses } else { $null })
            }
        })
        ConvertTo-Json -InputObject ([pscustomobject]@{ schemaVersion = 1; mode = $Mode; reason = $Why; shards = $rows }) -Depth 4 |
            Set-Content -LiteralPath (Join-Path $Repository $EffectiveFile) -Encoding utf8
    }
    catch { Write-Host "::warning::effective test-level selection not recorded: $($_.Exception.Message)" }
}

function Write-Passthrough([string] $Why) {
    # Passing through is a success. A native command this script tolerated (TypeImpact, yq) leaves its exit code in
    # $LASTEXITCODE, and the Actions pwsh wrapper ends the step with `exit $LASTEXITCODE`, so without this reset
    # the step failed after deliberately falling back.
    $global:LASTEXITCODE = 0
    Write-Host "test-level selection not applied: $Why - keeping the shard selection as chosen"
    Write-Output-Value 'matrix' $env:SELECTED_MATRIX
    Write-Output-Value 'ledger_matrix' $env:SELECTED_LEDGER_MATRIX
    Write-Output-Value 'skipped' $env:SELECTED_SKIPPED
    Write-Output-Value 'impact_mode' 'passthrough'
    $passed = @(try { @($env:SELECTED_MATRIX | ConvertFrom-Json) } catch { @() })
    Write-Effective 'passthrough' $Why $passed
    "### Test-level selection`n`nNot applied: $Why." | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8
}

function Get-InertChangedPaths([string] $Base) {
    # Select-Shards' path classifier, the same one the Select shards job applies. NonRuntime and BuildOnly paths
    # cannot change what any test executes; every other category (and any failure) keeps the path.
    $classifier = Join-Path $Repository 'tools/TestImpact/Select-Shards.ps1'
    if (-not (Test-Path -LiteralPath $classifier)) { return @() }
    $out = Join-Path ([IO.Path]::GetTempPath()) 'type-impact-path-classification.json'
    try {
        & pwsh -NoProfile -File $classifier -ClassifyOnly -BaseSha $Base -OutFile $out | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $out)) { return @() }
        $result = Get-Content -LiteralPath $out -Raw | ConvertFrom-Json
        $impacts = $result.PSObject.Properties['pathImpacts']
        if ($null -eq $impacts -or $null -eq $impacts.Value) { return @() }
        return @($impacts.Value.PSObject.Properties |
            Where-Object { [string] $_.Value -cin @('NonRuntime', 'BuildOnly') } | ForEach-Object Name)
    }
    catch {
        Write-Host "path classification unavailable ($($_.Exception.Message)); every changed path stays a type-impact input"
        return @()
    }
    finally { $global:LASTEXITCODE = 0 }
}

if (-not $BuildRoot) { $BuildRoot = $Repository }
$eventName = [string] $env:GITHUB_EVENT_NAME
if ($env:COLLECT_COVERAGE_EVERYWHERE -eq 'true') { Write-Passthrough 'coverage-everywhere run'; return }
if ($eventName -notin @('pull_request', 'push')) { Write-Passthrough "event '$eventName' runs the full selection"; return }
if ($eventName -eq 'push' -and $env:DELTA_MODE -ceq 'Partial') { Write-Passthrough 'post-merge delta re-run'; return }
if ($env:REQUIRES_SHARDS -ne 'true') { Write-Passthrough 'no test shards were required'; return }

try {
    # The change is the merge commit against its first parent: for a pull request, refs/pull/N/merge
    # against the base actually tested; for a push, the landed commit against the previous tip.
    $parents = @(((& git rev-list --parents -n 1 HEAD) -join ' ').Split(' ', [StringSplitOptions]::RemoveEmptyEntries))
    if ($LASTEXITCODE -ne 0 -or $parents.Count -lt 2) { throw 'HEAD has no parent to diff against' }
    if ($eventName -eq 'pull_request' -and ($parents.Count -ne 3 -or $parents[2] -cne $env:PR_HEAD_SHA)) {
        throw 'the checkout is not the pull request merge commit'
    }

    # Written with WriteAllLines, not piped to Set-Content: a pipeline with no input never runs Set-Content, so a
    # change with no C# edits (a workflow-only PR) left no diff file at all and TypeImpact died opening it.
    $changes = Join-Path ([IO.Path]::GetTempPath()) 'type-impact-changes.txt'
    $changeLines = @(& git -c core.quotepath=false diff --no-renames --name-status $parents[1] HEAD)
    if ($LASTEXITCODE -ne 0) { throw 'git diff failed' }
    # A path the shard selector proves cannot affect a test (documentation, an unreferenced tool or test project,
    # a .github file only GitHub reads, .editorconfig) is not a type-impact input: TypeImpact cannot map a non-C#
    # file, so one such file used to pass the whole decision through. Classification that fails removes nothing.
    $inert = @(Get-InertChangedPaths -Base $parents[1])
    if ($inert.Count -gt 0) {
        $inertSet = [Collections.Generic.HashSet[string]]::new([string[]] $inert, [StringComparer]::Ordinal)
        $changeLines = @($changeLines | Where-Object { -not $inertSet.Contains(([string] $_).Split("`t")[-1]) })
        Write-Host "left out $($inert.Count) path(s) that cannot affect a test: $(@($inert | Select-Object -First 5) -join ', ')"
    }
    [IO.File]::WriteAllLines($changes, [string[]] $changeLines)
    # Line-level diff, for the const edits the type graph cannot follow.
    $diff = Join-Path ([IO.Path]::GetTempPath()) 'type-impact-changes.diff'
    $diffLines = @(& git -c core.quotepath=false diff --no-renames -U0 $parents[1] HEAD -- '*.cs')
    if ($LASTEXITCODE -ne 0) { throw 'git diff -U0 failed' }
    [IO.File]::WriteAllLines($diff, [string[]] $diffLines)

    $all = @(yq -o=json -I=0 '.shard' (Join-Path $Repository '.github/test-shards.yml') | ConvertFrom-Json)
    if ($all.Count -eq 0) { throw 'test-shards.yml yielded no shards' }
    $manifest = Join-Path ([IO.Path]::GetTempPath()) 'type-impact-shards.json'
    ConvertTo-Json -InputObject @($all) -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8

    # Known-answer cases over a compiled fixture; a selector that fails them must not narrow anything.
    & pwsh -NoProfile -File (Join-Path $Repository 'tools/TestImpact/TypeImpact/Test-TypeImpact.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'TypeImpact failed its self-test' }
    # A generator change maps through its generated output when Build-GeneratorBase.ps1 built the merge base;
    # without that build it stays unmappable and the coverage selection is kept.
    $generatorArguments = @('--unmappable', 'src/AiDotNet.Generators/')
    if ($env:TYPE_IMPACT_BASE_BIN -and (Test-Path -LiteralPath $env:TYPE_IMPACT_BASE_BIN)) {
        $generatorArguments = @('--base-bin', $env:TYPE_IMPACT_BASE_BIN, '--base-repo', $env:TYPE_IMPACT_BASE_REPO,
            '--generator-root', 'src/AiDotNet.Generators/', '--generator-tests', 'AiDotNet.Tests.Generators.')
    }
    $tool = Join-Path $Repository 'tools/TestImpact/TypeImpact/TypeImpact.csproj'
    & dotnet run --project $tool -c Release --no-build -- `
        --repo $BuildRoot `
        --bin (Join-Path $BuildRoot 'tests/AiDotNet.Tests/bin/Release/net10.0') `
        --bin (Join-Path $BuildRoot 'tests/AiDotNet.Serving.Tests/bin/Release/net10.0') `
        --project 'tests/AiDotNet.Tests/AiDotNetTests.csproj=AiDotNetTests' `
        --project 'tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj=AiDotNet.Serving.Tests' `
        @generatorArguments `
        --unmappable "tools/" `
        --changes $changes --diff $diff --shards $manifest --out $PlanFile
    if ($LASTEXITCODE -ne 0) { throw "TypeImpact exited $LASTEXITCODE" }
    $plan = Get-Content -LiteralPath $PlanFile -Raw | ConvertFrom-Json
}
catch {
    Write-Host "::warning::test-level selection unavailable ($($_.Exception.Message))"
    Write-Passthrough 'TypeImpact could not produce a plan'
    return
}

if (-not $plan.resolved) {
    $first = @($plan.unresolved | Select-Object -First 3 | ForEach-Object { "$($_.path) ($($_.reason))" }) -join '; '
    Write-Passthrough "$(@($plan.unresolved).Count) changed file(s) cannot be mapped to types, e.g. $first"
    return
}

# Only ever narrow: a shard select-shards did not choose (a deferred nightly sweep, one outside the
# coverage selection) never runs here, however TypeImpact reaches it.
try {
    $chosen = [Collections.Generic.HashSet[string]]::new(
        [string[]] @($env:SELECTED_MATRIX | ConvertFrom-Json | ForEach-Object { [string] $_.name } | Where-Object { $_ }),
        [StringComparer]::Ordinal)
}
catch { $chosen = $null }
if ($null -eq $chosen -or $chosen.Count -eq 0) { Write-Passthrough 'the chosen shard matrix is empty or unreadable'; return }

$byName = @{}
foreach ($entry in @($plan.shards)) { $byName[[string] $entry.name] = $entry }
$narrowedShards = [Collections.Generic.List[object]]::new()
foreach ($shard in $all) {
    if (-not $chosen.Contains([string] $shard.name)) { continue }
    $entry = $byName[[string] $shard.name]
    if ($null -eq $entry -or -not $entry.run) { continue }
    $copy = $shard.PSObject.Copy()
    if ($entry.narrowed) {
        $copy.filter = [string] $entry.filter
        $copy | Add-Member -NotePropertyName narrowed -NotePropertyValue $true -Force
        $copy | Add-Member -NotePropertyName narrowedClasses -NotePropertyValue ([int] $entry.classes) -Force
    }
    $narrowedShards.Add($copy)
}

if ($narrowedShards.Count -eq 0) {
    # Nothing the change touches is reachable from a test. Rare enough (dead code, an unused
    # option) that running the chosen selection is cheaper than proving the empty plan right.
    Write-Passthrough 'no test class reaches the changed types'
    return
}

. (Join-Path $Repository 'tools/TestImpact/CiWorkloadKinds.ps1')
$runnable = Complete-CiWorkloadSelection -All $all -Selected @($narrowedShards) -RequiresValidation $true -Escalated $false
if ($runnable.Escalated) { Write-Passthrough 'the narrowed selection has no ordinary ledger-producing shard'; return }
$workloads = Split-CiWorkloads -Shards @($runnable.Shards)

function ConvertTo-MatrixJson($Shards) { ConvertTo-Json -InputObject @($Shards) -Depth 6 -Compress }
function Measure-Outputs { (ConvertTo-MatrixJson $tests).Length + (ConvertTo-MatrixJson $runnable.LedgerShards).Length }
$tests = @($workloads.Tests)
while ((Measure-Outputs) -gt $MaxMatrixCharacters) {
    $widest = $tests | Where-Object { $_.PSObject.Properties['narrowed'] -and $_.narrowed } |
        Sort-Object { $_.narrowedClasses } -Descending | Select-Object -First 1
    if ($null -eq $widest) { break }
    $original = $all | Where-Object { $_.name -ceq $widest.name } | Select-Object -First 1
    $widest.filter = $original.filter
    $widest.narrowed = $false
}
$json = ConvertTo-MatrixJson $tests
if ((Measure-Outputs) -gt $MaxMatrixCharacters) { Write-Passthrough 'the narrowed matrix does not fit a job output'; return }

$running = [Collections.Generic.HashSet[string]]::new([string[]] @($runnable.Shards.name), [StringComparer]::Ordinal)
$skipped = @($all | Where-Object { -not $running.Contains([string] $_.name) } | ForEach-Object { $_.name -replace '[\\/:*?"<>|\s-]+', '_' })

Write-Output-Value 'matrix' $json
Write-Output-Value 'ledger_matrix' (ConvertTo-Json -InputObject @($runnable.LedgerShards) -Depth 6 -Compress)
Write-Output-Value 'skipped' (ConvertTo-Json -InputObject @($skipped) -Depth 3 -Compress)
Write-Output-Value 'impact_mode' 'narrowed'
Write-Effective 'narrowed' '' $tests

$narrowedCount = @($tests | Where-Object { $_.PSObject.Properties['narrowed'] -and $_.narrowed }).Count
$lines = @(
    '### Test-level selection', '',
    "**$($tests.Count)** of $($all.Count) shards run ($narrowedCount narrowed to the affected classes); " +
        "$($plan.selectedTestClasses) of $($plan.totalTestClasses) test classes selected from $($plan.changedTypes) changed type(s).", '',
    '| Shard | Classes |', '|---|---|'
)
foreach ($shard in $tests) {
    $classes = if ($shard.PSObject.Properties['narrowed'] -and $shard.narrowed) { [string] $shard.narrowedClasses } else { 'whole shard' }
    $lines += "| $($shard.name) | $classes |"
}
($lines -join "`n") | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8
Write-Host "test-level selection: $($tests.Count) of $($all.Count) shard(s), $narrowedCount narrowed"
