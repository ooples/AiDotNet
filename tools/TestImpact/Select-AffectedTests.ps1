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
    # Job outputs are capped at 1 MB in total; narrowed filters are un-narrowed, largest first,
    # until the matrix fits under this.
    [int] $MaxMatrixCharacters = 700000
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Output-Value([string] $Name, [string] $Value) {
    "$Name=$Value" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
}

function Write-Passthrough([string] $Why) {
    Write-Host "test-level selection not applied: $Why - keeping the shard selection as chosen"
    Write-Output-Value 'matrix' $env:SELECTED_MATRIX
    Write-Output-Value 'ledger_matrix' $env:SELECTED_LEDGER_MATRIX
    Write-Output-Value 'skipped' $env:SELECTED_SKIPPED
    Write-Output-Value 'impact_mode' 'passthrough'
    "### Test-level selection`n`nNot applied: $Why." | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8
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

    $changes = Join-Path ([IO.Path]::GetTempPath()) 'type-impact-changes.txt'
    & git -c core.quotepath=false diff --no-renames --name-status $parents[1] HEAD | Set-Content -LiteralPath $changes -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'git diff failed' }

    $all = @(yq -o=json -I=0 '.shard' (Join-Path $Repository '.github/test-shards.yml') | ConvertFrom-Json)
    if ($all.Count -eq 0) { throw 'test-shards.yml yielded no shards' }
    $manifest = Join-Path ([IO.Path]::GetTempPath()) 'type-impact-shards.json'
    ConvertTo-Json -InputObject @($all) -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8

    # Known-answer cases over a compiled fixture; a selector that fails them must not narrow anything.
    & pwsh -NoProfile -File (Join-Path $Repository 'tools/TestImpact/TypeImpact/Test-TypeImpact.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'TypeImpact failed its self-test' }
    $tool = Join-Path $Repository 'tools/TestImpact/TypeImpact/TypeImpact.csproj'
    & dotnet run --project $tool -c Release --no-build -- `
        --repo $BuildRoot `
        --bin (Join-Path $BuildRoot 'tests/AiDotNet.Tests/bin/Release/net10.0') `
        --bin (Join-Path $BuildRoot 'tests/AiDotNet.Serving.Tests/bin/Release/net10.0') `
        --project 'tests/AiDotNet.Tests/AiDotNetTests.csproj=AiDotNetTests' `
        --project 'tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj=AiDotNet.Serving.Tests' `
        --unmappable 'src/AiDotNet.Generators/' `
        --unmappable "tools/" `
        --changes $changes --shards $manifest --out $PlanFile
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

$byName = @{}
foreach ($entry in @($plan.shards)) { $byName[[string] $entry.name] = $entry }
$narrowedShards = [Collections.Generic.List[object]]::new()
foreach ($shard in $all) {
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
$tests = @($workloads.Tests)
$json = ConvertTo-MatrixJson $tests
while ($json.Length -gt $MaxMatrixCharacters) {
    $widest = $tests | Where-Object { $_.PSObject.Properties['narrowed'] -and $_.narrowed } |
        Sort-Object { $_.narrowedClasses } -Descending | Select-Object -First 1
    if ($null -eq $widest) { break }
    $original = $all | Where-Object { $_.name -ceq $widest.name } | Select-Object -First 1
    $widest.filter = $original.filter
    $widest.narrowed = $false
    $json = ConvertTo-MatrixJson $tests
}
if ($json.Length -gt $MaxMatrixCharacters) { Write-Passthrough 'the narrowed matrix does not fit a job output'; return }

$running = [Collections.Generic.HashSet[string]]::new([string[]] @($runnable.Shards.name), [StringComparer]::Ordinal)
$skipped = @($all | Where-Object { -not $running.Contains([string] $_.name) } | ForEach-Object { $_.name -replace '[\\/:*?"<>|\s-]+', '_' })

Write-Output-Value 'matrix' $json
Write-Output-Value 'ledger_matrix' (ConvertTo-Json -InputObject @($runnable.LedgerShards) -Depth 6 -Compress)
Write-Output-Value 'skipped' (ConvertTo-Json -InputObject @($skipped) -Depth 3 -Compress)
Write-Output-Value 'impact_mode' 'narrowed'

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
