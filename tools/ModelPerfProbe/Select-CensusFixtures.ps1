# Copyright (c) AiDotNet. All rights reserved.

<#
.SYNOPSIS
    Narrows the census fixture inventory to the fixtures a pull request can affect, and sizes the
    census matrix to them.

.DESCRIPTION
    A caller (a manual dispatch, or a future pipeline caller) hands the census a scope: 'full', 'none', or 'selected' with the
    test shards the coverage selection chose. A census fixture is the ModelPerformanceCensus test
    of a model-family test class, and the correctness tests of that same class run in the shard
    whose filter matches it - so the shards the change reaches name the models it can affect.

    Ownership is decided on the class's real test names (its correctness tests, from the complete
    test listing), because a filter such as FullyQualifiedName~UnitTests could match an unknown
    method name and would otherwise claim every class. A fixture is kept when any selected shard's
    filter can match one of its class's tests, and also when NO shard in the manifest can: an
    unowned class has no coverage to prove it unaffected. Filters are evaluated with
    Select-Shards.ps1's own three-valued evaluator; categories are unknown, so a category condition
    never hides a class from the shard that names it.

    The matrix keeps the full census's density: about FullShardCount shards for the whole
    inventory, so a subset gets proportionally fewer runners.

.PARAMETER Inventory
    The vstest --ListTests output (test-list.txt).

.PARAMETER AllTests
    The complete vstest --ListTests output for the same assembly. Required for a 'selected' scope.

.PARAMETER Scope
    JSON: { "mode": "full" | "selected" | "none", "shards": [ ... ] }. Empty means 'full'.

.PARAMETER ShardManifestFile
    JSON array of { name, project, filter } converted from .github/test-shards.yml.

.PARAMETER OutFile
    Where the narrowed inventory is written, in the same format as the input.

.PARAMETER SelfTest
    Runs the built-in checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $Inventory,
    [Parameter(ParameterSetName = 'Select')] [string] $AllTests,
    [Parameter(ParameterSetName = 'Select')] [AllowEmptyString()] [string] $Scope = '',
    [Parameter(ParameterSetName = 'Select')] [string] $ShardManifestFile,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $OutFile,
    [Parameter(ParameterSetName = 'Select')] [ValidateRange(1, 256)] [int] $FullShardCount = 32,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The filter evaluator is Select-Shards.ps1's, loaded as declarations only: that script's body is
# the PR selector and must not run here. One evaluator means the census can never read a shard
# filter differently from the selection that chose the shard.
$selectorPath = Join-Path $PSScriptRoot '../TestImpact/Select-Shards.ps1'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path -LiteralPath $selectorPath).Path, [ref] $tokens, [ref] $parseErrors)
if ($parseErrors.Count -gt 0) { throw 'Select-Shards.ps1 does not parse; the filter evaluator is unavailable.' }
$needed = @('ConvertTo-TestFilter', 'Invert-FilterResult', 'Test-OpenSuffixCanComplete', 'Test-FilterStringCondition', 'Test-TestFilter')
foreach ($name in $needed) {
    $definition = @($ast.EndBlock.Statements | Where-Object {
        $_ -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $_.Name -ceq $name })
    if ($definition.Count -ne 1) { throw "Select-Shards.ps1 no longer defines $name." }
    . ([scriptblock]::Create($definition[0].Extent.Text))
}
$script:FilterFalse = 0
$script:FilterTrue = 1
$script:FilterUnknown = 2

function Get-CensusFixtureClass {
    param([Parameter(Mandatory)] [string] $TestName)
    return ($TestName.Trim() -replace '\.ModelPerformanceCensus$', '')
}

function Get-TestFullyQualifiedName {
    <# A listed test name without its theory arguments: what a FullyQualifiedName filter sees. #>
    param([Parameter(Mandatory)] [string] $Listed)
    $name = $Listed.Trim()
    $open = $name.IndexOf('(')
    if ($open -ge 0) { $name = $name.Substring(0, $open) }
    return $name
}

function Get-ClassTestIndex {
    <# Class -> its listed tests other than the census, from the complete listing. #>
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [AllowEmptyString()] [string[]] $Listing)
    $index = [Collections.Generic.Dictionary[string, Collections.Generic.List[string]]]::new([StringComparer]::Ordinal)
    foreach ($line in $Listing) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $name = Get-TestFullyQualifiedName $line
        $dot = $name.LastIndexOf('.')
        if ($dot -le 0 -or $name.EndsWith('.ModelPerformanceCensus', [StringComparison]::Ordinal)) { continue }
        $class = $name.Substring(0, $dot)
        if (-not $index.ContainsKey($class)) { $index[$class] = [Collections.Generic.List[string]]::new() }
        [void] $index[$class].Add($name)
    }
    return $index
}

function Test-FilterCanMatchAny {
    <#
        Whether the filter can select any of the class's tests. Categories are unknown per test.
        The class prefix is tried first: false for "Class.<any method>" is false for every real
        test of the class, and it settles most filters without reading the tests one by one.
    #>
    param(
        [Parameter(Mandatory)] $Filter,
        [Parameter(Mandatory)] [string] $Class,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Names
    )
    $prefix = [pscustomobject]@{ Fqn = "$Class."; PrefixOnly = $true; AnySuffix = $false; Categories = $null }
    if ((Test-TestFilter -Node $Filter -Candidate $prefix) -eq $script:FilterFalse) { return $false }
    foreach ($name in $Names) {
        $candidate = [pscustomobject]@{ Fqn = $name; PrefixOnly = $false; AnySuffix = $false; Categories = $null }
        if ((Test-TestFilter -Node $Filter -Candidate $candidate) -ne $script:FilterFalse) { return $true }
    }
    return $false
}

function Select-CensusTests {
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Tests,
        [Parameter(Mandatory)] [string] $Mode,
        [AllowEmptyCollection()] [string[]] $Shards = @(),
        [AllowEmptyCollection()] [object[]] $Manifest = @(),
        [AllowEmptyCollection()] [AllowEmptyString()] [string[]] $Listing = @()
    )
    if ($Mode -ceq 'none') { return @() }
    if ($Mode -ceq 'full') { return @($Tests) }
    if ($Mode -cne 'selected') { throw "unknown census scope '$Mode'" }

    $byName = @{}
    foreach ($shard in $Manifest) { $byName[[string] $shard.name] = $shard }
    $missing = @($Shards | Where-Object { -not $byName.ContainsKey($_) })
    if ($missing.Count -gt 0) { throw "selected shard(s) absent from the manifest: $($missing -join ', ')" }

    # Parse every filter up front: one this evaluator cannot read must not silently own nothing.
    $parsed = @($Manifest | ForEach-Object {
        [pscustomobject]@{ Name = [string] $_.name; Filter = ConvertTo-TestFilter -Text ([string] $_.filter) }
    })
    $selectedNames = [Collections.Generic.HashSet[string]]::new([string[]] @($Shards), [StringComparer]::Ordinal)
    $selectedFilters = @($parsed | Where-Object { $selectedNames.Contains($_.Name) })

    $index = Get-ClassTestIndex -Listing $Listing
    $kept = [Collections.Generic.List[string]]::new()
    foreach ($test in $Tests) {
        $class = Get-CensusFixtureClass $test
        [string[]] $siblings = @(if ($index.ContainsKey($class)) { $index[$class] })
        $owned = $false
        foreach ($shard in $selectedFilters) {
            if (Test-FilterCanMatchAny -Filter $shard.Filter -Class $class -Names $siblings) { $owned = $true; break }
        }
        if (-not $owned) {
            $anyOwner = $false
            foreach ($shard in $parsed) {
                if (Test-FilterCanMatchAny -Filter $shard.Filter -Class $class -Names $siblings) { $anyOwner = $true; break }
            }
            # No shard runs this class's correctness tests, so no coverage says it is unaffected.
            if (-not $anyOwner) { $owned = $true }
        }
        if ($owned) { [void] $kept.Add($test) }
    }
    return @($kept)
}

function Get-CensusShardCount {
    param([int] $Selected, [int] $Total, [int] $FullShards)
    if ($Selected -le 0) { return 0 }
    return [int] [Math]::Max(1, [Math]::Min($FullShards, [Math]::Ceiling($Selected * $FullShards / [double] $Total)))
}

if ($SelfTest) {
    $failures = [Collections.Generic.List[string]]::new()
    function Assert-Census([bool] $Condition, [string] $Message) { if (-not $Condition) { [void] $failures.Add($Message) } }
    $prefix = 'AiDotNet.Tests.ModelFamilyTests'
    $tests = @(
        "$prefix.NeuralNetworks.AlexNetTests.ModelPerformanceCensus",
        "$prefix.NeuralNetworks.ResNetTests.ModelPerformanceCensus",
        "$prefix.NeuralNetworks.SqueezeNetTests.ModelPerformanceCensus",
        "$prefix.Diffusion.DdpmModelTests.ModelPerformanceCensus",
        "$prefix.Orphans.LonelyModelTests.ModelPerformanceCensus"
    )
    $listing = @(
        'The following Tests are available:',
        "    $prefix.NeuralNetworks.AlexNetTests.ForwardPass_ShouldBeFinite",
        "    $prefix.NeuralNetworks.AlexNetTests.Serialize_RoundTrips(seed: 3)",
        "    $prefix.NeuralNetworks.ResNetTests.ForwardPass_ShouldBeFinite",
        "    $prefix.NeuralNetworks.SqueezeNetTests.ForwardPass_ShouldBeFinite",
        "    $prefix.Diffusion.DdpmModelTests.Sample_IsFinite",
        "    $prefix.Orphans.LonelyModelTests.Forward_IsFinite"
    ) + @($tests | ForEach-Object { "    $_" })
    $manifest = @(
        [pscustomobject]@{ name = 'NN A-F'; project = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
            filter = "FullyQualifiedName~ModelFamilyTests.NeuralNetworks&(FullyQualifiedName~NeuralNetworks.A|FullyQualifiedName~NeuralNetworks.B)" },
        [pscustomobject]@{ name = 'NN S'; project = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
            filter = 'FullyQualifiedName~ModelFamilyTests.NeuralNetworks&FullyQualifiedName~NeuralNetworks.S' },
        [pscustomobject]@{ name = 'NN R'; project = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
            filter = 'FullyQualifiedName~ModelFamilyTests.NeuralNetworks&FullyQualifiedName~NeuralNetworks.R&Category!=ModelPerformanceCensus' },
        [pscustomobject]@{ name = 'Diffusion'; project = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
            filter = 'FullyQualifiedName~ModelFamilyTests.Diffusion' },
        [pscustomobject]@{ name = 'Unit'; project = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
            filter = 'FullyQualifiedName~UnitTests' }
    )

    $kept = @(Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('NN A-F') -Manifest $manifest -Listing $listing)
    Assert-Census ($kept -ccontains $tests[0]) 'a fixture owned by a selected shard was dropped'
    Assert-Census (-not ($kept -ccontains $tests[1]) -and -not ($kept -ccontains $tests[2]) -and -not ($kept -ccontains $tests[3])) `
        'a fixture owned only by an unselected shard was kept'
    Assert-Census ($kept -ccontains $tests[4]) 'a fixture no shard owns was dropped'

    $kept = @(Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('NN R') -Manifest $manifest -Listing $listing)
    Assert-Census ($kept -ccontains $tests[1]) 'a category exclusion on the correctness shard hid the class from the census'

    $kept = @(Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('Unit') -Manifest $manifest -Listing $listing)
    Assert-Census ($kept.Count -eq 1 -and $kept[0] -ceq $tests[4]) 'a selection owning no fixture kept more than the unowned ones'

    $kept = @(Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('NN S') -Manifest $manifest -Listing @($tests))
    Assert-Census ($kept.Count -eq $tests.Count) 'a class with no listed correctness test was not kept as unowned'

    $byMethod = @($manifest) + @([pscustomobject]@{ name = 'ByMethod'; project = 'x'; filter = 'FullyQualifiedName~Serialize_RoundTrips' })
    $kept = @(Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('ByMethod') -Manifest $byMethod -Listing $listing)
    Assert-Census (($kept -ccontains $tests[0]) -and -not ($kept -ccontains $tests[1])) 'a method-name filter was not matched against the real test names'

    # A theory's listed name carries its arguments; FullyQualifiedName does not.
    $exact = @($manifest) + @([pscustomobject]@{ name = 'Exact'; project = 'x'; filter = "FullyQualifiedName=$prefix.NeuralNetworks.AlexNetTests.Serialize_RoundTrips" })
    $kept = @(Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('Exact') -Manifest $exact -Listing $listing)
    Assert-Census ($kept -ccontains $tests[0]) 'a theory was compared with its arguments instead of its fully qualified name'

    Assert-Census (@(Select-CensusTests -Tests $tests -Mode 'full').Count -eq $tests.Count) 'full scope dropped fixtures'
    Assert-Census (@(Select-CensusTests -Tests $tests -Mode 'none').Count -eq 0) 'none scope kept fixtures'

    $threw = $false
    try { $null = Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('Renamed') -Manifest $manifest } catch { $threw = $true }
    Assert-Census $threw 'a selected shard missing from the manifest did not fail'
    $threw = $false
    $broken = @($manifest) + @([pscustomobject]@{ name = 'Broken'; project = 'x'; filter = 'FullyQualifiedName~A&(' })
    try { $null = Select-CensusTests -Tests $tests -Mode 'selected' -Shards @('NN S') -Manifest $broken -Listing $listing } catch { $threw = $true }
    Assert-Census $threw 'an unreadable filter did not fail'

    Assert-Census ((Get-CensusShardCount -Selected 0 -Total 400 -FullShards 32) -eq 0) 'an empty selection was given shards'
    Assert-Census ((Get-CensusShardCount -Selected 1 -Total 400 -FullShards 32) -eq 1) 'one fixture did not get one shard'
    Assert-Census ((Get-CensusShardCount -Selected 25 -Total 400 -FullShards 32) -eq 2) 'a subset was not sized at the full census density'
    Assert-Census ((Get-CensusShardCount -Selected 400 -Total 400 -FullShards 32) -eq 32) 'the full inventory did not get the full matrix'

    if ($failures.Count -gt 0) {
        $failures | ForEach-Object { Write-Host "::error::$_" }
        exit 1
    }
    Write-Host 'Census fixture selection passed: selected, unselected, unowned, category-excluded, method-name, unlisted-class, empty, full, none, missing-shard, unreadable-filter and sizing cases.'
    exit 0
}

$scopeObject = if ([string]::IsNullOrWhiteSpace($Scope)) { [pscustomobject]@{ mode = 'full'; shards = @() } }
               else { $Scope | ConvertFrom-Json }
$mode = [string] $scopeObject.mode
$shards = @(if ($scopeObject.PSObject.Properties['shards']) { @($scopeObject.shards | ForEach-Object { [string] $_ }) })
$manifest = @()
$listing = @()
if ($mode -ceq 'selected') {
    if (-not $ShardManifestFile) { throw 'a selected census scope needs the shard manifest' }
    if (-not $AllTests) { throw 'a selected census scope needs the complete test listing' }
    $manifest = @(Get-Content -LiteralPath $ShardManifestFile -Raw | ConvertFrom-Json)
    $listing = @(Get-Content -LiteralPath $AllTests)
}

$lines = @(Get-Content -LiteralPath $Inventory)
$tests = @($lines | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.ModelPerformanceCensus$' })
if ($tests.Count -eq 0) { throw "No ModelPerformanceCensus fixtures were found in $Inventory." }
$kept = @(Select-CensusTests -Tests $tests -Mode $mode -Shards $shards -Manifest $manifest -Listing $listing)
$shardCount = Get-CensusShardCount -Selected $kept.Count -Total $tests.Count -FullShards $FullShardCount

$keptSet = [Collections.Generic.HashSet[string]]::new([string[]] $kept, [StringComparer]::Ordinal)
@($lines | Where-Object { $_ -notmatch '\.ModelPerformanceCensus$' -or $keptSet.Contains($_.Trim()) }) |
    Set-Content -LiteralPath $OutFile -Encoding utf8

Write-Host "census scope '$mode': $($kept.Count) of $($tests.Count) fixture(s) on $shardCount shard(s)"
if ($mode -ceq 'selected') { Write-Host "  from shard(s): $($shards -join ', ')" }
if ($env:GITHUB_OUTPUT) {
    "count=$($kept.Count)" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    "total=$($tests.Count)" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    "shard_count=$shardCount" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    "matrix=$(ConvertTo-Json -InputObject @(0..([Math]::Max(0, $shardCount - 1)) | Select-Object -First $shardCount) -Compress)" |
        Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
}
exit 0
