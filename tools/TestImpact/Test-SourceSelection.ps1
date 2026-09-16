[CmdletBinding()]
param([Parameter(Mandatory)][string] $EvidenceDirectory, [switch] $CrossAssembly)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $root) { throw 'Source-selection evidence must be new.' }
$checkout = Join-Path $root 'checkout'
[void](New-Item -ItemType Directory -Path $checkout)
$cli = Join-Path $PSScriptRoot 'Attribution.Cli/bin/Release/net10.0/Attribution.Cli.dll'
$mapper = Join-Path $PSScriptRoot 'fixtures/TestAttribution/Instrumenter/bin/Release/net10.0/AttributionInstrumenter.dll'
$adapter = [Security.SecurityElement]::Escape((Join-Path $PSScriptRoot 'Attribution.Xunit/bin/Release/net10.0'))
$project = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net10.0</TargetFramework><IsTestProject>true</IsTestProject><IsPackable>false</IsPackable><ManagePackageVersionsCentrally>false</ManagePackageVersionsCentrally><IncludeSourceRevisionInInformationalVersion>false</IncludeSourceRevisionInInformationalVersion></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="18.10.0" />
    <PackageReference Include="xunit" Version="2.9.3" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.8.2" />
    <Reference Include="Attribution.Xunit"><HintPath>$adapter/Attribution.Xunit.dll</HintPath></Reference>
    <Reference Include="AttributionRuntime"><HintPath>$adapter/AttributionRuntime.dll</HintPath></Reference>
    <Reference Include="Attribution.Protocol"><HintPath>$adapter/Attribution.Protocol.dll</HintPath></Reference>
  </ItemGroup>
</Project>
"@
if ($CrossAssembly) {
    $libraryDirectory = Join-Path $checkout 'Library'
    [void](New-Item -ItemType Directory -Path $libraryDirectory)
    [IO.File]::WriteAllText((Join-Path $libraryDirectory 'SourceLibrary.csproj'), @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net10.0</TargetFramework><IncludeSourceRevisionInInformationalVersion>false</IncludeSourceRevisionInInformationalVersion></PropertyGroup>
</Project>
'@)
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fixtures/TestAttribution/SourceCases/Library.cs') -Destination $libraryDirectory
    $project = $project.Replace('</Project>', '<ItemGroup><Compile Remove="Library/**/*.cs" /><ProjectReference Include="Library/SourceLibrary.csproj" /></ItemGroup></Project>')
}
[IO.File]::WriteAllText((Join-Path $checkout 'SourceCases.csproj'), $project)
[IO.File]::WriteAllText((Join-Path $checkout '.gitignore'), "obj/`nbin/`n")
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fixtures/TestAttribution/SourceCases/Cases.cs') -Destination $checkout
if ($CrossAssembly) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fixtures/TestAttribution/SourceCases/CrossAssemblyCases.cs') -Destination (Join-Path $checkout 'Cases.cs')
}
function Check([bool] $Condition, [string] $Message) { if (-not $Condition) { throw $Message } }
function Invoke-Git([string[]] $Arguments) {
    $result = & git -C $checkout @Arguments
    Check ($LASTEXITCODE -eq 0) 'Fixture git operation failed.'
    return ($result -join "`n")
}
function Commit([string] $Message) {
    $null = Invoke-Git @('add','.')
    $null = Invoke-Git @('-c','user.name=Attribution proof','-c','user.email=attribution-proof@example.invalid',
        '-c','commit.gpgsign=false','commit','--quiet','-m',$Message)
    return Invoke-Git @('rev-parse','HEAD')
}
function Build([string] $Name, [string] $Define = '') {
    $bundle = Join-Path $root "$Name-bundle"
    $buildOptions = @()
    if ($Define) { $buildOptions += "-p:DefineConstants=$Define" }
    dotnet build (Join-Path $checkout 'SourceCases.csproj') -c Release -o $bundle --nologo -v:quiet @buildOptions | Out-Host
    Check ($LASTEXITCODE -eq 0) "$Name build failed."
    return $bundle
}
function Snapshot([string] $Name, [string] $Bundle) {
    $path = Join-Path $root "$Name-snapshot.json"
    if ($CrossAssembly) {
        dotnet $mapper (Join-Path $Bundle 'SourceCases.dll') $path SourceBundle $checkout (Join-Path $Bundle 'SourceLibrary.dll') | Out-Host
    } else {
        dotnet $mapper (Join-Path $Bundle 'SourceCases.dll') $path SourceSnapshot $checkout | Out-Host
    }
    Check ($LASTEXITCODE -eq 0) "$Name source snapshot failed."
    return $path
}
function Discover([string] $Name, [string] $Bundle, [string] $Source) {
    $env:ATTRIBUTION_MODE = 'Discover'
    $env:ATTRIBUTION_SOURCE_TREE = $Source
    $env:ATTRIBUTION_PLAN = ''
    $env:ATTRIBUTION_INVENTORY = Join-Path $root "$Name-inventory.json"
    Run $Name $Bundle | Out-Null
    $inventory = Get-Content $env:ATTRIBUTION_INVENTORY -Raw | ConvertFrom-Json
    Check ($inventory.Cases.Count -eq 3) 'Source proof discovery must contain three cases.'
    return $env:ATTRIBUTION_INVENTORY
}
function Run([string] $Name, [string] $Bundle) {
    $env:ATTRIBUTION_OUTPUT = Join-Path $root "$Name/reports"
    $env:ATTRIBUTION_RUN = [guid]::NewGuid().ToString('N')
    dotnet vstest (Join-Path $Bundle 'SourceCases.dll') '/TestCaseFilter:FullyQualifiedName~SourceCases.Cases' `
        "/ResultsDirectory:$root/$Name" '/Logger:trx;LogFileName=results.trx' | Out-Host
    Check ($LASTEXITCODE -eq 0) "$Name execution failed."
    return $env:ATTRIBUTION_RUN
}
$saved = @{}
$names = @('ATTRIBUTION_MODE','ATTRIBUTION_PLAN','ATTRIBUTION_INVENTORY','ATTRIBUTION_OUTPUT','ATTRIBUTION_RUN',
    'ATTRIBUTION_OWNER','ATTRIBUTION_TOKEN','ATTRIBUTION_SOURCE_TREE','ATTRIBUTION_PROFILE_HASH','ATTRIBUTION_WORKLOAD')
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name); [Environment]::SetEnvironmentVariable($name, $null) }
try {
    $null = Invoke-Git @('init','--quiet')
    $before = Commit 'test: baseline source selection proof'
    $env:ATTRIBUTION_PROFILE_HASH = 'source-selection-net10-fixed-workload'
    $env:ATTRIBUTION_WORKLOAD = 'source-cases/net10.0'
    $oldBundle = Build before
    $oldSnapshot = Snapshot before $oldBundle
    $oldInventory = Discover before $oldBundle $before
    dotnet $cli Prepare $oldInventory "$root/baseline-plan.json" FullWorkload | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Cannot prepare real full baseline.'
    $env:ATTRIBUTION_MODE = 'ExecutePlan'
    $env:ATTRIBUTION_PLAN = "$root/baseline-plan.json"
    $baselineRun = Run baseline-full $oldBundle
    dotnet $cli Verify $oldInventory "$root/baseline-plan.json" "$root/baseline-full/reports" "$root/baseline-full/results.trx" `
        $baselineRun local/source-proof 1 1 "$root/baseline-verified.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Full baseline did not pass independent verification.'
    $baselineInput = [ordered]@{ Plan="$root/baseline-plan.json"; Reports="$root/baseline-full/reports";
        Trx="$root/baseline-full/results.trx"; CollectionRun=$baselineRun; Origin=@{Repository='local/source-proof';RunId=1;Attempt=1} }
    $baselineInput | ConvertTo-Json -Depth 8 | Set-Content "$root/baseline-input.json"
    dotnet $cli Prepare $oldInventory "$root/partial-baseline-plan.json" SelectedMethods 'SourceCases:SourceCases.Cases.Left' | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Cannot prepare partial baseline rejection control.'
    $env:ATTRIBUTION_PLAN = "$root/partial-baseline-plan.json"
    $partialBaselineRun = Run baseline-partial $oldBundle
    $partialBaselineInput = [ordered]@{ Plan="$root/partial-baseline-plan.json"; Reports="$root/baseline-partial/reports";
        Trx="$root/baseline-partial/results.trx"; CollectionRun=$partialBaselineRun; Origin=@{Repository='local/source-proof';RunId=1;Attempt=1} }
    $sourcePath = Join-Path $checkout $(if ($CrossAssembly) { 'Library/Library.cs' } else { 'Cases.cs' })
    $source = [IO.File]::ReadAllText($sourcePath)
    $oldBody = if ($CrossAssembly) { 'return input + 1;' } else { 'Equal(2, 1 + 1);' }
    $newBody = if ($CrossAssembly) { 'return 1 + input;' } else { 'Equal(3, 1 + 2);' }
    Check ($source.Contains($oldBody)) 'Missing expected source edit anchor.'
    [IO.File]::WriteAllText($sourcePath, $source.Replace($oldBody, $newBody))
    $after = Commit 'test: change one executable test body'

    # Actual stale PDB/binary control: checkout changed but old assembly did not.
    $staleSnapshot = Snapshot stale $oldBundle
    $staleSource = Get-Content $staleSnapshot -Raw | ConvertFrom-Json
    $staleAssemblies = if ($CrossAssembly) { @($staleSource.Assemblies) } else { @($staleSource) }
    Check (@($staleAssemblies | Where-Object Status -eq 'Unverifiable').Count -gt 0) 'Stale PDB source was trusted.'
    $staleInventory = Discover stale $oldBundle $after
    dotnet $cli SelectChanges $checkout $oldSnapshot $staleSnapshot $oldInventory $staleInventory $oldBundle $oldBundle `
        "$root/forbidden-stale-plan.json" "$root/forbidden-stale-selection.json" 2>&1 | Tee-Object -FilePath "$root/stale-rejection.log" | Out-Host
    Check ($LASTEXITCODE -ne 0 -and -not (Test-Path "$root/forbidden-stale-plan.json")) 'Stale binaries authorized execution.'
    Check ((Get-Content "$root/stale-rejection.log" -Raw).Contains('Source snapshot differs from the discovered binary bundle.')) 'Stale control failed for another reason.'

    $newBundle = Build after
    $newSnapshot = Snapshot after $newBundle
    $newInventory = Discover after $newBundle $after
    $plan = Join-Path $root 'selected-plan.json'
    dotnet $cli SelectChanges $checkout $oldSnapshot $newSnapshot $oldInventory $newInventory $oldBundle $newBundle $plan "$root/selection.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Actual diff selection failed.'
    $selection = Get-Content "$root/selection.json" -Raw | ConvertFrom-Json
    Check (-not $selection.Selection.FullFallback -and $selection.Selection.Methods.Count -eq 1 -and
        $selection.Selection.Methods[0].MethodId -eq 'SourceCases:SourceCases.Cases.Left') 'Actual body edit did not select exactly its test.'
    $env:ATTRIBUTION_MODE = 'ExecutePlan'
    $env:ATTRIBUTION_PLAN = $plan
    $run = Run selected $newBundle
    dotnet $cli Verify $newInventory $plan "$root/selected/reports" "$root/selected/results.trx" $run local/source-proof 1 1 "$root/verified.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Actual source-selected execution did not verify.'
    [xml]$trx = Get-Content "$root/selected/results.trx" -Raw
    $rows = @($trx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    Check ($rows.Count -eq 1 -and $rows[0].testName -eq 'SourceCases.Cases.Left' -and $rows[0].outcome -eq 'Passed') 'Wrong tests ran after source selection.'

    if ($CrossAssembly) {
        dotnet $mapper (Join-Path $newBundle 'SourceCases.dll') "$root/missing-library-snapshot.json" SourceBundle $checkout | Out-Host
        Check ($LASTEXITCODE -eq 0) 'Could not construct the missing-dependency control.'
        dotnet $cli SelectChanges $checkout $oldSnapshot "$root/missing-library-snapshot.json" $oldInventory $newInventory $oldBundle $newBundle `
            "$root/missing-library-plan.json" "$root/missing-library-selection.json" | Out-Host
        Check ($LASTEXITCODE -eq 0) 'Missing dependency did not fall back safely.'
        $missingLibrary = Get-Content "$root/missing-library-plan.json" -Raw | ConvertFrom-Json
        Check ($missingLibrary.Scope -eq 'FullWorkload' -and $missingLibrary.RequiredCases.Count -eq 3) 'Missing dependency allowed partial execution.'
    }

    $beforeInput = [ordered]@{ Snapshot=$oldSnapshot; Inventory=$oldInventory; Bundle=$oldBundle }
    $afterInput = [ordered]@{ Snapshot=$newSnapshot; Inventory=$newInventory; Bundle=$newBundle }
    $request = [ordered]@{ Repository=$checkout; Before=$beforeInput; After=$afterInput; Baseline=$baselineInput }
    $requestPath = "$root/reuse-request.json"
    $request | ConvertTo-Json -Depth 12 | Set-Content $requestPath
    dotnet $cli PrepareReuse $requestPath "$root/reuse-plan.json" "$root/reuse-partition.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Could not prepare changed-source reuse from the actual full baseline.'
    $reusePlan = Get-Content "$root/reuse-plan.json" -Raw | ConvertFrom-Json
    Check ($reusePlan.PlanHash -eq (Get-Content $plan -Raw | ConvertFrom-Json).PlanHash) 'Reuse and actual source selection disagreed.'
    $executionInput = [ordered]@{ Plan=$plan; Reports="$root/selected/reports"; Trx="$root/selected/results.trx";
        CollectionRun=$run; Origin=@{Repository='local/source-proof';RunId=1;Attempt=1} }
    $executionInput | ConvertTo-Json -Depth 8 | Set-Content "$root/execution-input.json"
    dotnet $cli CompleteReuse $requestPath "$root/reuse-completed.json" "$root/execution-input.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Could not complete the actual changed-source partition.'
    $completed = Get-Content "$root/reuse-completed.json" -Raw | ConvertFrom-Json
    Check ($completed.Cases.Count -eq 3 -and $completed.ReusedCases.Count -eq 2 -and $completed.ExecutedCases.Count -eq 1 -and
        -not $completed.CanReplaceFullBaseline -and $completed.BaselineContext.SourceTree -eq $before -and $completed.Context.SourceTree -eq $after) `
        'Mixed source execution lost provenance or was promoted to a fresh full baseline.'
    dotnet $cli CompleteReuse $requestPath "$root/forbidden-missing-execution.json" 2>&1 | Tee-Object -FilePath "$root/missing-execution.log" | Out-Host
    Check ($LASTEXITCODE -ne 0 -and -not (Test-Path "$root/forbidden-missing-execution.json")) 'Missing fresh execution was accepted.'
    Check ((Get-Content "$root/missing-execution.log" -Raw).Contains('Current execution does not satisfy the exact partition.')) 'Missing-execution control failed for another reason.'
    $request.Baseline = $partialBaselineInput
    $request | ConvertTo-Json -Depth 12 | Set-Content "$root/partial-reuse-request.json"
    dotnet $cli PrepareReuse "$root/partial-reuse-request.json" "$root/forbidden-partial-reuse-plan.json" "$root/forbidden-partial-reuse.json" `
        2>&1 | Tee-Object -FilePath "$root/partial-baseline.log" | Out-Host
    Check ($LASTEXITCODE -ne 0 -and -not (Test-Path "$root/forbidden-partial-reuse.json")) 'Actual partial baseline authorized reuse.'
    Check ((Get-Content "$root/partial-baseline.log" -Raw).Contains('A partial run cannot seed changed-base reuse.')) 'Partial-baseline control failed for another reason.'
    $request.Baseline = $baselineInput
    $request.After = $beforeInput
    $request | ConvertTo-Json -Depth 12 | Set-Content "$root/unchanged-reuse-request.json"
    dotnet $cli PrepareReuse "$root/unchanged-reuse-request.json" "$root/forbidden-empty-plan.json" "$root/unchanged-partition.json" | Out-Host
    Check ($LASTEXITCODE -eq 0 -and -not (Test-Path "$root/forbidden-empty-plan.json")) 'Reuse-only preparation manufactured an empty test execution.'
    dotnet $cli CompleteReuse "$root/unchanged-reuse-request.json" "$root/unchanged-completed.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Verified unchanged baseline could not be reused.'
    $unchanged = Get-Content "$root/unchanged-completed.json" -Raw | ConvertFrom-Json
    Check ($unchanged.ReusedCases.Count -eq 3 -and $null -eq $unchanged.ExecutedCases -and -not $unchanged.CanReplaceFullBaseline) 'Reuse-only evidence was promoted to fresh execution.'

    # Same commit and same PDB source bytes, but different compiler output. A
    # source-diff-only implementation would incorrectly select no tests here.
    foreach ($variant in @([pscustomobject]@{ Name='binary'; Define='SOURCE_ALTERNATIVE'; Expected=1 },
        [pscustomobject]@{ Name='metadata'; Define='METADATA_ALTERNATIVE'; Expected=3 })) {
        $variantBundle = Build $variant.Name $variant.Define
        $variantSnapshot = Snapshot $variant.Name $variantBundle
        $variantInventory = Discover $variant.Name $variantBundle $after
        $variantPlan = Join-Path $root "$($variant.Name)-plan.json"
        $variantSelection = Join-Path $root "$($variant.Name)-selection.json"
        dotnet $cli SelectChanges $checkout $newSnapshot $variantSnapshot $newInventory $variantInventory $newBundle $variantBundle `
            $variantPlan $variantSelection | Out-Host
        Check ($LASTEXITCODE -eq 0) "$($variant.Name) identical-source selection failed."
        $variantResult = Get-Content $variantSelection -Raw | ConvertFrom-Json
        Check ($variantResult.Delta.Before.Count -eq 0 -and $variantResult.Delta.After.Count -eq 0) 'Identical-source control unexpectedly contained a source diff.'
        $variantExecution = Get-Content $variantPlan -Raw | ConvertFrom-Json
        Check ($variantExecution.RequiredCases.Count -eq $variant.Expected) "$($variant.Name) output change lost required cases."
        if ($variant.Expected -eq 1) {
            Check ($variantExecution.RequiredCases[0].MethodId -eq 'SourceCases:SourceCases.Cases.Left') 'Binary change selected the wrong method.'
        } else { Check ($variantExecution.Scope -eq 'FullWorkload') 'Metadata change did not broaden execution.' }
        $env:ATTRIBUTION_MODE = 'ExecutePlan'
        $env:ATTRIBUTION_PLAN = $variantPlan
        $variantRun = Run "$($variant.Name)-selected" $variantBundle
        dotnet $cli Verify $variantInventory $variantPlan "$root/$($variant.Name)-selected/reports" "$root/$($variant.Name)-selected/results.trx" `
            $variantRun local/source-proof 1 1 "$root/$($variant.Name)-verified.json" | Out-Host
        Check ($LASTEXITCODE -eq 0) "$($variant.Name) selected execution did not verify."
    }

    [IO.File]::WriteAllText((Join-Path $checkout 'settings.json'), '{"unknownInput":true}')
    $configRevision = Commit 'test: add an unmapped configuration input'
    $configBundle = Build config
    $configSnapshot = Snapshot config $configBundle
    $configInventory = Discover config $configBundle $configRevision
    dotnet $cli SelectChanges $checkout $newSnapshot $configSnapshot $newInventory $configInventory $newBundle $configBundle `
        "$root/config-plan.json" "$root/config-selection.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Configuration fallback failed.'
    $config = Get-Content "$root/config-plan.json" -Raw | ConvertFrom-Json
    Check ($config.Scope -eq 'FullWorkload' -and $config.RequiredCases.Count -eq 3) 'Unmapped configuration skipped tests.'
    $env:ATTRIBUTION_MODE = 'ExecutePlan'
    $env:ATTRIBUTION_PLAN = "$root/config-plan.json"
    $configRun = Run config-full $configBundle
    dotnet $cli Verify $configInventory "$root/config-plan.json" "$root/config-full/reports" "$root/config-full/results.trx" `
        $configRun local/source-proof 1 1 "$root/config-verified.json" | Out-Host
    Check ($LASTEXITCODE -eq 0) 'Full fallback execution did not verify.'
    [ordered]@{ before=$before; after=$after; config=$configRevision; crossAssembly=[bool]$CrossAssembly; discovered=3; baselinePassed=3; sourceSelected=1;
        configurationFallback=3; identicalSourceBinarySelected=1; identicalSourceMetadataFallback=3;
        changedSourceReused=2; changedSourceExecuted=1; mixedCannotReplaceBaseline=$true;
        partialBaselineRejected=$true; missingFreshExecutionRejected=$true; reuseOnlyCases=3;
        staleBinaryRejected=$true; productionSelectionEnabled=$false } |
        ConvertTo-Json | Set-Content "$root/proof.json"
    Write-Host "Actual source selection passed: 1/3 tests, configuration fallback 3/3, stale binary rejected. Evidence: $root"
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
