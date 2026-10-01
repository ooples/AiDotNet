<#
.SYNOPSIS
    Known-answer tests for TypeImpact, run against a small compiled fixture (SelfTest/).

.DESCRIPTION
    Each case names a changed file and the exact test classes and shards the plan must select. The
    fixture is built from source every time, so the cases read real portable PDBs and real IL.
    Exits non-zero on the first failing case.
#>
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$here = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $here '../../..')).Path
$work = Join-Path ([IO.Path]::GetTempPath()) "type-impact-selftest-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $work | Out-Null
$fixture = 'tools/TestImpact/TypeImpact/SelfTest'
try {
    & dotnet build (Join-Path $here 'TypeImpact.csproj') -c Release --nologo -v q
    if ($LASTEXITCODE -ne 0) { throw 'TypeImpact did not build' }
    $bin = Join-Path $work 'bin'
    & dotnet build (Join-Path $here 'SelfTest/Tests/FixtureTests.csproj') -c Release --nologo -v q -o $bin
    if ($LASTEXITCODE -ne 0) { throw 'the self-test fixture did not build' }

    $shards = Join-Path $work 'shards.json'
    ConvertTo-Json -Depth 4 -InputObject @(
        [ordered]@{ name = 'Fast'; project = 'Fixture.csproj'; framework = 'net10.0'; filter = 'Category!=Slow' },
        [ordered]@{ name = 'Slow'; project = 'Fixture.csproj'; framework = 'net10.0'; filter = 'Category=Slow' },
        [ordered]@{ name = 'Inventory sweep'; project = 'Fixture.csproj'; framework = 'net10.0'; filter = 'FullyQualifiedName~InventoryTests'; nightlyOnly = $true },
        [ordered]@{ name = 'Other project'; project = 'Other.csproj'; framework = 'net10.0'; filter = '' }
    ) | Set-Content -LiteralPath $shards -Encoding utf8

    $cases = @(
        @{ Name = 'a leaf model selects its own tests, the inherited contract and the inventory, not its sibling or the catalog'
           Change = "M`t$fixture/Lib/Alpha.cs"
           Classes = @('AlphaContractTests', 'AlphaTests', 'HelperSweepTests', 'InventoryTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a base class reaches every model that derives from it and every test that names it'
           Change = "M`t$fixture/Lib/ModelBase.cs"
           Classes = @('AlphaContractTests', 'AlphaTests', 'BetaTests', 'CatalogTests', 'HelperSweepTests', 'InventoryTests')
           Runs = @('Fast', 'Slow', 'Other project'); Resolved = $true },
        @{ Name = 'a changed catalog selects the tests that call it'
           Change = "M`t$fixture/Lib/Catalog.cs"
           Classes = @('CatalogTests', 'HelperSweepTests', 'InventoryTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a body-less interface maps through the type-document record'
           Change = "M`t$fixture/Lib/ISurface.cs"
           Classes = @('HelperSweepTests', 'InventoryTests', 'SurfaceTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'an edited sweep test runs its nightly-only shard'
           Change = "M`t$fixture/Tests/InventoryTests.cs"
           Classes = @('InventoryTests')
           Runs = @('Fast', 'Inventory sweep', 'Other project'); Resolved = $true },
        @{ Name = 'an abstract test base selects its concrete subclasses'
           Change = "M`t$fixture/Tests/ModelContractTests.cs"
           Classes = @('AlphaContractTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a test-side helper that enumerates types selects its callers, and only for production changes'
           Change = "M`t$fixture/Tests/TypeSweep.cs"
           Classes = @('HelperSweepTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a body edit in a type that exposes a const stays mapped'
           Change = "M`t$fixture/Lib/Limits.cs"
           Diff = "--- a/$fixture/Lib/Limits.cs`n+++ b/$fixture/Lib/Limits.cs`n@@ -8 +8 @@`n-    public static int Describe() => MaxDepth;`n+    public static int Describe() => MaxDepth + 0;"
           Classes = @('HelperSweepTests', 'InventoryTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'an edited const reaches the files that name it, though the IL holds no reference'
           Change = "M`t$fixture/Lib/Limits.cs"
           Diff = "--- a/$fixture/Lib/Limits.cs`n+++ b/$fixture/Lib/Limits.cs`n@@ -6 +6 @@`n-    public const int MaxDepth = 3;`n+    public const int MaxDepth = 4;"
           Classes = @('HelperSweepTests', 'InventoryTests', 'LimitsTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a const line whose declaration cannot be read stays unresolved'
           Change = "M`t$fixture/Lib/Limits.cs"
           Diff = "--- a/$fixture/Lib/Limits.cs`n+++ b/$fixture/Lib/Limits.cs`n@@ -5 +5 @@`n-    // limits`n+    // const values below"
           Classes = @(); Runs = @('Other project'); Resolved = $false },
        @{ Name = 'a deleted C# source is unresolved: its former dependents are not in the new build'
           Change = "D`t$fixture/Lib/Gamma.cs"
           Classes = @(); Runs = @('Other project'); Resolved = $false },
        @{ Name = 'a build file cannot be mapped, so the plan is unresolved'
           Change = "M`tDirectory.Build.props"
           Classes = @(); Runs = @('Other project'); Resolved = $false },
        @{ Name = 'a generator input is never mapped'
           Change = "M`tsrc/AiDotNet.Generators/SomeGenerator.cs"
           Classes = @(); Runs = @('Other project'); Resolved = $false },
        @{ Name = 'a source generator maps to the types it emitted, read from the PDB'
           Change = "M`t$fixture/Gen/GreetingGenerator.cs"
           Classes = @('GreetingTests', 'HelperSweepTests', 'InventoryTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a shared generator helper maps to everything the project generated'
           Change = "M`t$fixture/Gen/GenHelpers.cs"
           Classes = @('GreetingTests', 'HelperSweepTests', 'InventoryTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'an analyzer changes diagnostics, not tests'
           Change = "M`t$fixture/Gen/QuietAnalyzer.cs"
           Classes = @(); Runs = @('Other project'); Resolved = $true },
        @{ Name = 'a deleted generator source is unresolved'
           Change = "D`t$fixture/Gen/OldGenerator.cs"
           Classes = @(); Runs = @('Other project'); Resolved = $false }
    )

    $failures = 0
    foreach ($case in $cases) {
        $changes = Join-Path $work 'changes.txt'
        Set-Content -LiteralPath $changes -Value $case.Change -Encoding utf8
        $out = Join-Path $work 'plan.json'
        $diffArguments = @()
        if ($case.ContainsKey('Diff')) {
            $diffFile = Join-Path $work 'changes.diff'
            Set-Content -LiteralPath $diffFile -Value $case.Diff -Encoding utf8
            $diffArguments = @('--diff', $diffFile)
        }
        & dotnet (Join-Path $here 'bin/Release/net10.0/TypeImpact.dll') --repo $repo --bin $bin @diffArguments `
            --project 'Fixture.csproj=FixtureTests' --unmappable 'src/AiDotNet.Generators/' `
            --generator-project "$fixture/Gen/=FixtureGen" `
            --catalog-threshold 3 --catalog-max-entry-points 1 `
            --changes $changes --shards $shards --out $out | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "TypeImpact exited $LASTEXITCODE on '$($case.Name)'" }
        $plan = Get-Content -LiteralPath $out -Raw | ConvertFrom-Json
        $classes = @($plan.shards | Where-Object { $_.PSObject.Properties['testClasses'] } |
            ForEach-Object { $_.testClasses } | ForEach-Object { ($_ -split '\.')[-1] } | Sort-Object -Unique)
        $runs = @($plan.shards | Where-Object run | ForEach-Object name | Sort-Object)
        $problems = @()
        if ([bool] $plan.resolved -ne $case.Resolved) { $problems += "resolved=$($plan.resolved), expected $($case.Resolved)" }
        if (($classes -join ',') -cne (@($case.Classes | Sort-Object) -join ',')) { $problems += "classes [$($classes -join ', ')], expected [$($case.Classes -join ', ')]" }
        if (($runs -join ',') -cne (@($case.Runs | Sort-Object) -join ',')) { $problems += "shards [$($runs -join ', ')], expected [$($case.Runs -join ', ')]" }
        if ($problems.Count -gt 0) {
            $failures++
            Write-Host "FAIL: $($case.Name)"
            $problems | ForEach-Object { Write-Host "  $_" }
        }
        else {
            Write-Host "pass: $($case.Name)"
        }
    }

    # A narrowed filter must stay a valid conjunction of the shard's own filter and the classes.
    Set-Content -LiteralPath (Join-Path $work 'changes.txt') -Value "M`t$fixture/Lib/Beta.cs" -Encoding utf8
    & dotnet (Join-Path $here 'bin/Release/net10.0/TypeImpact.dll') --repo $repo --bin $bin --project 'Fixture.csproj=FixtureTests' `
        --catalog-threshold 3 --catalog-max-entry-points 1 --changes (Join-Path $work 'changes.txt') --shards $shards --out (Join-Path $work 'plan.json') | Out-Null
    $slow = (Get-Content -LiteralPath (Join-Path $work 'plan.json') -Raw | ConvertFrom-Json).shards | Where-Object name -eq 'Slow'
    if ($slow.filter -cne '(Category=Slow)&(FullyQualifiedName~Fixture.Tests.BetaTests.)') {
        $failures++
        Write-Host "FAIL: narrowed filter is '$($slow.filter)'"
    }
    else {
        Write-Host 'pass: a narrowed filter conjoins the shard filter with the selected classes'
    }

    # --check-ownership: a class no shard of its project selects fails the check by name; full coverage passes.
    # BetaTests is the class the 'Slow' filter alone selects, so dropping that shard leaves exactly it unowned.
    $partialShards = Join-Path $work 'partial-shards.json'
    ConvertTo-Json -Depth 4 -InputObject @(
        [ordered]@{ name = 'Fast'; project = 'Fixture.csproj'; framework = 'net10.0'; filter = 'Category!=Slow' }
    ) | Set-Content -LiteralPath $partialShards -Encoding utf8
    $ownership = Join-Path $work 'ownership.json'
    & dotnet (Join-Path $here 'bin/Release/net10.0/TypeImpact.dll') --repo $repo --bin $bin --project 'Fixture.csproj=FixtureTests' `
        --shards $partialShards --check-ownership --out $ownership | Out-Null
    $partialExit = $LASTEXITCODE
    $unowned = @((Get-Content -LiteralPath $ownership -Raw | ConvertFrom-Json).unownedTestClasses)
    & dotnet (Join-Path $here 'bin/Release/net10.0/TypeImpact.dll') --repo $repo --bin $bin --project 'Fixture.csproj=FixtureTests' `
        --shards $shards --check-ownership --out $ownership | Out-Null
    $fullExit = $LASTEXITCODE
    if ($partialExit -ne 1 -or ($unowned -join ',') -cne 'Fixture.Tests.BetaTests' -or $fullExit -ne 0) {
        $failures++
        Write-Host "FAIL: ownership check (partial exit $partialExit, unowned '$($unowned -join ',')', full exit $fullExit)"
    }
    else {
        Write-Host 'pass: the ownership check names a class no shard selects, and passes full coverage'
    }

    if ($failures -gt 0) {
        Write-Host "TypeImpact self-test FAILED ($failures case(s))"
        exit 1
    }

    Write-Host "TypeImpact self-test passed ($($cases.Count + 2) cases)."
    exit 0
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $work -ErrorAction SilentlyContinue
}
