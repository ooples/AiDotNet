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
           Classes = @('AlphaContractTests', 'AlphaTests', 'InventoryTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a base class reaches every model that derives from it and every test that names it'
           Change = "M`t$fixture/Lib/ModelBase.cs"
           Classes = @('AlphaContractTests', 'AlphaTests', 'BetaTests', 'CatalogTests', 'InventoryTests')
           Runs = @('Fast', 'Slow', 'Other project'); Resolved = $true },
        @{ Name = 'a changed catalog selects the tests that call it'
           Change = "M`t$fixture/Lib/Catalog.cs"
           Classes = @('CatalogTests', 'InventoryTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a body-less interface maps through the type-document record'
           Change = "M`t$fixture/Lib/ISurface.cs"
           Classes = @('InventoryTests', 'SurfaceTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'an edited sweep test runs its nightly-only shard'
           Change = "M`t$fixture/Tests/InventoryTests.cs"
           Classes = @('InventoryTests')
           Runs = @('Fast', 'Inventory sweep', 'Other project'); Resolved = $true },
        @{ Name = 'an abstract test base selects its concrete subclasses'
           Change = "M`t$fixture/Tests/ModelContractTests.cs"
           Classes = @('AlphaContractTests')
           Runs = @('Fast', 'Other project'); Resolved = $true },
        @{ Name = 'a build file cannot be mapped, so the plan is unresolved'
           Change = "M`tDirectory.Build.props"
           Classes = @(); Runs = @('Other project'); Resolved = $false },
        @{ Name = 'a generator input is never mapped'
           Change = "M`tsrc/AiDotNet.Generators/SomeGenerator.cs"
           Classes = @(); Runs = @('Other project'); Resolved = $false }
    )

    $failures = 0
    foreach ($case in $cases) {
        $changes = Join-Path $work 'changes.txt'
        Set-Content -LiteralPath $changes -Value $case.Change -Encoding utf8
        $out = Join-Path $work 'plan.json'
        & dotnet (Join-Path $here 'bin/Release/net10.0/TypeImpact.dll') --repo $repo --bin $bin `
            --project 'Fixture.csproj=FixtureTests' --unmappable 'src/AiDotNet.Generators/' `
            --catalog-threshold 3 --catalog-max-entry-points 1 `
            --changes $changes --shards $shards --out $out | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "TypeImpact exited $LASTEXITCODE on '$($case.Name)'" }
        $plan = Get-Content -LiteralPath $out -Raw | ConvertFrom-Json
        $classes = @($plan.shards | Where-Object { $_.PSObject.Properties['testClasses'] } |
            ForEach-Object { $_.testClasses } | ForEach-Object { ($_ -split '\.')[-1] } | Sort-Object -Unique)
        if (-not $case.Resolved) { $classes = @() }
        $runs = @($plan.shards | Where-Object run | ForEach-Object name | Sort-Object)
        $problems = @()
        if ([bool] $plan.resolved -ne $case.Resolved) { $problems += "resolved=$($plan.resolved), expected $($case.Resolved)" }
        if (($classes -join ',') -cne (@($case.Classes | Sort-Object) -join ',')) { $problems += "classes [$($classes -join ', ')], expected [$($case.Classes -join ', ')]" }
        if ($case.Resolved -and ($runs -join ',') -cne (@($case.Runs | Sort-Object) -join ',')) { $problems += "shards [$($runs -join ', ')], expected [$($case.Runs -join ', ')]" }
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

    if ($failures -gt 0) {
        Write-Host "TypeImpact self-test FAILED ($failures case(s))"
        exit 1
    }

    Write-Host "TypeImpact self-test passed ($($cases.Count + 1) cases)."
    exit 0
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $work -ErrorAction SilentlyContinue
}
