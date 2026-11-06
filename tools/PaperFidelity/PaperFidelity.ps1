<#
.SYNOPSIS
    Shared definitions for the paper-fidelity lanes. Dot-source it.

.DESCRIPTION
    Every paper-fidelity test (the generated option-sensitivity theories, the LearnsToyTask_* facts on the
    family bases, GeneratedPaperDefaultsTests) carries [Trait("Category", "PaperFidelity")]. The ordinary PR
    shards exclude that category (.github/scripts/Invoke-Shard.ps1 appends Category!=PaperFidelity to every
    shard filter, and tools/TestImpact/Assert-ShardInventory.ps1 names it as a deliberate exclusion). Two lanes
    run it instead:

      - pull requests: the "Paper fidelity" jobs in sonarcloud.yml run it for the test classes the change can
        reach, read from the Build job's TypeImpact plan (Select-AffectedTests.ps1 adds the entry below to the
        manifest it plans over), and fall back to the full sweep exactly where that plan cannot narrow;
      - nightly: .github/workflows/paper-fidelity-nightly.yml runs the full sweep across parallel slices.

    Both lanes run slices with Invoke-PaperFidelitySlice.ps1, so a pull request's fallback sweep and the
    nightly sweep are the same work.
#>

Set-StrictMode -Version Latest

$script:PaperFidelityCategory = 'PaperFidelity'
$script:PaperFidelityProject = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
$script:PaperFidelityFramework = 'net10.0'
# The slice count of a full sweep, on pull requests and nightly alike.
$script:PaperFidelityFullSweepSlices = 20
# Classes one pull-request slice runs before another slice is added.
$script:PaperFidelityClassesPerSlice = 20

function Get-PaperFidelityImpactEntry {
    <#
    .SYNOPSIS
        The manifest entry TypeImpact plans the paper-fidelity lane with, alongside the test-shards.yml shards.
    .DESCRIPTION
        Planned like any shard: TypeImpact lists the reachable classes with a method this filter can select, and
        narrows to them unless more than its --max-classes (400) are reachable, in which case the entry runs
        whole. It is not a test-shards.yml shard, so the ordinary test matrix never runs it.
    #>
    return [pscustomobject]@{
        name = 'Paper fidelity'
        project = $script:PaperFidelityProject
        framework = $script:PaperFidelityFramework
        filter = "Category=$($script:PaperFidelityCategory)"
    }
}

function Resolve-PaperFidelitySelection {
    <#
    .SYNOPSIS
        What a pull request's paper-fidelity lane runs: nothing, the reachable classes, or the full sweep.
    .DESCRIPTION
        Mirrors the rule the ordinary shards follow, so shared code is handled the same way:
          - no test shards required (a change that cannot reach a test)            -> None
          - no plan, an unreadable plan, or one that did not resolve every changed
            file (a generator input without a base build, a non-C# runtime file)    -> Full
          - the plan's paper-fidelity entry does not run (nothing reaches it)       -> None
          - the entry runs whole: more classes are reachable than TypeImpact
            narrows to (a shared family base, a widely used layer)                  -> Full
          - otherwise                                                              -> Impact, with its classes
    .OUTPUTS
        [pscustomobject] with Mode ('None' | 'Impact' | 'Full'), Classes (string[]) and Reason.
    #>
    param(
        [AllowEmptyString()] [string] $PlanFile,
        [Parameter(Mandatory)] [bool] $RequiresShards
    )
    $none = { param($why) [pscustomobject]@{ Mode = 'None'; Classes = @(); Reason = $why } }
    $full = { param($why) [pscustomobject]@{ Mode = 'Full'; Classes = @(); Reason = $why } }
    if (-not $RequiresShards) { return & $none 'the change cannot reach a test' }
    if (-not $PlanFile -or -not (Test-Path -LiteralPath $PlanFile)) {
        return & $full 'the Build job produced no test-level plan'
    }
    try { $plan = Get-Content -LiteralPath $PlanFile -Raw | ConvertFrom-Json }
    catch { return & $full "the test-level plan is unreadable ($($_.Exception.Message))" }

    $resolved = $plan.PSObject.Properties['resolved']
    if ($null -eq $resolved -or $resolved.Value -ne $true) {
        return & $full 'the test-level plan could not map every changed file'
    }
    $name = (Get-PaperFidelityImpactEntry).name
    $entry = @($plan.shards | Where-Object { $_ -and [string] $_.name -ceq $name }) | Select-Object -First 1
    if ($null -eq $entry) { return & $full 'the test-level plan has no paper-fidelity entry' }
    if ($entry.run -ne $true) { return & $none 'no paper-fidelity test class reaches the changed types' }
    $classesProperty = $entry.PSObject.Properties['testClasses']
    if ($entry.narrowed -ne $true -or $null -eq $classesProperty) {
        return & $full "$($entry.classes) paper-fidelity classes are reachable, more than the plan narrows to"
    }
    $classes = @($classesProperty.Value | ForEach-Object { [string] $_ } | Where-Object { $_ } | Sort-Object -Unique)
    if ($classes.Count -eq 0) { return & $none 'no paper-fidelity test class reaches the changed types' }
    return [pscustomobject]@{ Mode = 'Impact'; Classes = $classes; Reason = "$($classes.Count) reachable class(es)" }
}

function Get-PaperFidelitySliceCount {
    <# The number of parallel slices for a selection. #>
    param([Parameter(Mandatory)] [object] $Selection)
    switch ($Selection.Mode) {
        'None' { return 0 }
        'Full' { return $script:PaperFidelityFullSweepSlices }
        default {
            return [Math]::Min($script:PaperFidelityFullSweepSlices,
                [Math]::Max(1, [int][Math]::Ceiling($Selection.Classes.Count / [double] $script:PaperFidelityClassesPerSlice)))
        }
    }
}

function Select-PaperFidelitySlice {
    <#
    .SYNOPSIS
        The classes slice <Slice> of <SliceCount> owns: every SliceCount-th class of the sorted list, so the slices
        cover every class exactly once.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Classes,
        [Parameter(Mandatory)] [int] $Slice,
        [Parameter(Mandatory)] [int] $SliceCount
    )
    if ($SliceCount -lt 1 -or $Slice -lt 0 -or $Slice -ge $SliceCount) { throw "slice $Slice of $SliceCount is out of range" }
    $sorted = @($Classes | Sort-Object -Unique -CaseSensitive)
    return @(for ($i = 0; $i -lt $sorted.Count; $i++) { if ($i % $SliceCount -eq $Slice) { $sorted[$i] } })
}

function Find-PaperFidelityClasses {
    <#
    .SYNOPSIS
        Every test class carrying the paper-fidelity category, discovered from the built test assembly.
    .DESCRIPTION
        Throws when discovery fails or finds nothing: discovering nothing would "pass" by running nothing.
    #>
    $listed = @(& dotnet test $script:PaperFidelityProject -c Release --framework $script:PaperFidelityFramework `
        --no-build --no-restore --filter "Category=$($script:PaperFidelityCategory)" --list-tests 2>&1 |
        ForEach-Object { [string] $_ })
    if ($LASTEXITCODE -ne 0) {
        $listed | Select-Object -Last 40 | ForEach-Object { Write-Host $_ }
        throw "dotnet test --list-tests exited $LASTEXITCODE"
    }
    $classes = @(ConvertFrom-PaperFidelityTestListing -Lines $listed)
    if ($classes.Count -eq 0) { throw "no $($script:PaperFidelityCategory) test classes were discovered" }
    return $classes
}

function ConvertFrom-PaperFidelityTestListing {
    <#
    .SYNOPSIS
        The distinct test classes in `dotnet test --list-tests` output.
    .DESCRIPTION
        Listed tests are indented four spaces, `Namespace.Class.Method` optionally followed by theory arguments.
    #>
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Lines)
    return @($Lines |
        ForEach-Object { $_ -replace '\r', '' } |
        Where-Object { $_ -match '^\s{4}\S+\.\S+' } |
        ForEach-Object { ($_.Trim() -split '\(')[0] } |
        Where-Object { $_.LastIndexOf('.') -gt 0 } |
        ForEach-Object { $_.Substring(0, $_.LastIndexOf('.')) } |
        Sort-Object -Unique -CaseSensitive)
}
