[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/CiWorkloadKinds.ps1"
function Assert-True([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw $Message }
}
$normal = [pscustomobject]@{ name = 'Ordinary'; filter = 'Category!=Sweep' }
$sweep = [pscustomobject]@{ name = 'Count'; workload = 'ParameterSweep' }
$shape = [pscustomobject]@{ name = 'Shape'; workload = 'ModelShape' }
$split = Split-CiWorkloads @($normal, $sweep, $shape)
Assert-True ($split.Tests.Count -eq 1 -and $split.Tests[0].name -ceq 'Ordinary') 'Ordinary partition is wrong.'
Assert-True ($split.ParameterSweep.Count -eq 1 -and $split.ParameterSweep[0].name -ceq 'Count') 'Sweep partition is wrong.'
Assert-True ($split.ModelShape.Count -eq 1 -and $split.ModelShape[0].name -ceq 'Shape') 'Shape partition is wrong.'
$split = Split-CiWorkloads @($shape)
Assert-True ($split.Tests.Count -eq 0 -and $split.ParameterSweep.Count -eq 0 -and $split.ModelShape.Count -eq 1) `
    'An isolated shape selection widened to other families.'
$split = Split-CiWorkloads @()
Assert-True ($split.Tests.Count + $split.ParameterSweep.Count + $split.ModelShape.Count -eq 0) `
    'An authorized empty selection became full.'
foreach ($bad in @('', 'Unknown', 'parametersweep', '1', 1, $null)) {
    $rejected = $false
    try { $null = Split-CiWorkloads @([pscustomobject]@{ name = 'Invalid'; workload = $bad }) }
    catch { $rejected = $true }
    Assert-True $rejected 'Malformed workload kind was silently omitted.'
}
$workflow = Get-Content (Join-Path $PSScriptRoot '../../.github/workflows/sonarcloud.yml') -Raw
$emitter = [regex]::Match($workflow,
    '(?ms)^          \. ./tools/TestImpact/CiWorkloadKinds\.ps1\r?\n.*?(?=^          "skipped=)')
Assert-True $emitter.Success 'Shipping workload emitter was not found.'
$code = [scriptblock]::Create([regex]::Replace($emitter.Value, '(?m)^          ', ''))
$all = @($normal, $sweep, $shape)
$legacyMap = [pscustomobject]@{ knownShards = @('Ordinary'); alwaysRun = @(); files = [pscustomobject]@{} }
$extension = Complete-CiMapWorkloads -Map $legacyMap -Manifest $all
Assert-True ($extension.Added.Count -eq 2 -and ($extension.Map.alwaysRun -join ',') -ceq 'Count,Shape') `
    'Legacy ordinary coverage did not retain every new auxiliary workload as mandatory.'
Assert-True ($legacyMap.alwaysRun.Count -eq 0 -and $extension.Map.knownShards[0] -ceq 'Ordinary') `
    'Legacy extension mutated its source or invented indexed coverage.'
$complete = Complete-CiMapWorkloads -Map $extension.Map -Manifest $all
Assert-True ($complete.Added.Count -eq 0) 'A complete map was extended twice.'
$newOrdinary = Complete-CiMapWorkloads -Map $legacyMap -Manifest @($all + [pscustomobject]@{ name = 'New ordinary' })
Assert-True (($newOrdinary.Added -join ',') -ceq 'Count,Shape,New ordinary' -and
    ($newOrdinary.Map.alwaysRun -join ',') -ceq 'Count,Shape,New ordinary') `
    'A new ordinary shard the map has not measured was not made mandatory.'
Assert-True (($newOrdinary.Map.knownShards -join ',') -ceq 'Ordinary') `
    'A new ordinary shard was given indexed coverage it never measured.'
$rejected = $false
try { $null = Complete-CiMapWorkloads -Map $legacyMap -Manifest @($normal, [pscustomobject]@{ name = 'Bad'; workload = 'tests' }) }
catch { $rejected = $true }
Assert-True $rejected 'A new workload with a malformed kind was accepted.'
$rejected = $false
try { $null = Complete-CiMapWorkloads -Map $legacyMap -Manifest @($sweep) }
catch { $rejected = $true }
Assert-True $rejected 'A map shard the manifest no longer has was accepted.'
$runnable = Complete-CiWorkloadSelection -All $all -Selected @($normal, $sweep) -RequiresValidation $true -Escalated $false
Assert-True (-not $runnable.Escalated -and $runnable.Shards.Count -eq 2) 'A valid mixed selection widened.'
$runnable = Complete-CiWorkloadSelection -All $all -Selected @($shape) -RequiresValidation $true -Escalated $false
Assert-True ($runnable.Escalated -and $runnable.Shards.Count -eq 3) 'Auxiliary-only selection would publish an empty ordinary ledger.'
$runnable = Complete-CiWorkloadSelection -All $all -Selected @($shape) -ImportedNames @('Ordinary') -RequiresValidation $true -Escalated $false
Assert-True (-not $runnable.Escalated -and $runnable.Shards.Count -eq 1 -and $runnable.Shards[0].name -ceq 'Shape' -and
    $runnable.LedgerShards.Count -eq 1 -and $runnable.LedgerShards[0].name -ceq 'Ordinary') 'Imported ordinary evidence did not preserve an auxiliary-only rerun.'
$runnable = Complete-CiWorkloadSelection -All $all -Selected @($shape) -ImportedNames @('Count') -RequiresValidation $true -Escalated $false
Assert-True $runnable.Escalated 'Auxiliary imports falsely satisfied the ordinary ledger requirement.'
foreach ($imports in @(@('Unknown'), @('Ordinary', 'Ordinary'), @('Shape'))) {
    $rejected = $false
    try { $null = Complete-CiWorkloadSelection -All $all -Selected @($shape) -ImportedNames $imports -RequiresValidation $true -Escalated $false }
    catch { $rejected = $true }
    Assert-True $rejected 'An unknown, duplicate, or rerun-overlapping import was accepted.'
}
$runnable = Complete-CiWorkloadSelection -All $all -Selected @() -RequiresValidation $false -Escalated $false
Assert-True (-not $runnable.Escalated -and $runnable.Shards.Count -eq 0) 'Non-runtime selection was widened.'
$completion = [regex]::Match($workflow,
    '(?ms)^          \. ''\./tools/TestImpact/CiWorkloadKinds\.ps1''\r?\n.*?(?=^          # Slug exactly)')
Assert-True $completion.Success 'Shipping workload completion block was not found.'
$completeCode = [scriptblock]::Create([regex]::Replace($completion.Value, '(?m)^          ', ''))
Push-Location (Join-Path $PSScriptRoot '../..')
try {
    foreach ($partial in @($true, $false)) {
        $deltaPartial = $partial
        $escalate = $false
        $requiresValidation = $true
        $matrixShards = @($shape)
        $importShards = @('Ordinary')
        . $completeCode
        Assert-True ($escalate -eq (-not $partial)) 'Shipping completion ignored valid imports or trusted imports outside a partial plan.'
        Assert-True ($ledgerShards.Count -eq 1 -and $ledgerShards[0].name -ceq 'Ordinary') 'Shipping completion lost ordinary evidence.'
        Assert-True ($matrixShards.Count -eq $(if ($partial) { 1 } else { 3 })) 'Shipping completion widened a valid partial plan.'
    }
}
finally { Pop-Location }
Assert-True ($workflow -match '(?m)^          EXPECTED_SHARD_MATRIX: \$\{\{ needs\.select-shards\.outputs\.ledger_matrix \}\}\r?$') `
    'Regression inventory does not require imported ordinary shards.'
$outputPath = Join-Path ([IO.Path]::GetTempPath()) "ci-workloads-$([Guid]::NewGuid().ToString('N')).txt"
$previousOutput = $env:GITHUB_OUTPUT
try {
    $env:GITHUB_OUTPUT = $outputPath
    foreach ($case in @(
        @{ Shards = @($normal); Tests = 1; Sweeps = 0; Shapes = 0 },
        @{ Shards = @($sweep); Tests = 0; Sweeps = 1; Shapes = 0 },
        @{ Shards = @($shape); Tests = 0; Sweeps = 0; Shapes = 1 },
        @{ Shards = @($normal, $sweep, $shape); Tests = 1; Sweeps = 1; Shapes = 1 },
        @{ Shards = @(); Tests = 0; Sweeps = 0; Shapes = 0 }
    )) {
        if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath }
        $matrixShards = @($case.Shards)
        $ledgerShards = @((Split-CiWorkloads $matrixShards).Tests)
        if ($case.Shapes -eq 1 -and $case.Tests -eq 0) { $ledgerShards = @($normal) }
        & $code
        $values = @{}
        foreach ($line in Get-Content -LiteralPath $outputPath) {
            $parts = $line.Split('=', 2)
            Assert-True (-not $values.ContainsKey($parts[0])) 'Duplicate workflow output.'
            $values[$parts[0]] = $parts[1]
        }
        $actualLedger = @($values['ledger_matrix'] | ConvertFrom-Json)
        Assert-True (($actualLedger | ConvertTo-Json -Compress) -ceq ($ledgerShards | ConvertTo-Json -Compress)) 'Shipping emitter lost the imported ordinary ledger inventory.'
        if ($case.Shapes -eq 1 -and $case.Tests -eq 0) {
            $expected = @(& "$PSScriptRoot/../../.github/scripts/get-selected-shard-names.ps1" -MatrixJson $values['ledger_matrix'])
            Assert-True ($expected.Count -eq 1 -and $expected[0] -ceq 'Ordinary') 'Production ledger reader cannot accept an auxiliary-only rerun with imported evidence.'
        }
        foreach ($partition in @(
            @{ Matrix = 'matrix'; Flag = 'requires_tests'; Count = $case.Tests },
            @{ Matrix = 'parameter_matrix'; Flag = 'requires_sweeps'; Count = $case.Sweeps },
            @{ Matrix = 'shape_matrix'; Flag = 'requires_shapes'; Count = $case.Shapes }
        )) {
            $rows = @($values[$partition.Matrix] | ConvertFrom-Json)
            Assert-True ($rows.Count -eq $partition.Count) "Wrong emitted $($partition.Matrix) count."
            Assert-True ($values[$partition.Flag] -ceq ($partition.Count -gt 0).ToString().ToLowerInvariant()) `
                "Wrong emitted $($partition.Flag) decision."
        }
        $totalEmitted = @($values.matrix | ConvertFrom-Json).Count +
            @($values.parameter_matrix | ConvertFrom-Json).Count + @($values.shape_matrix | ConvertFrom-Json).Count
        Assert-True ($totalEmitted -eq $matrixShards.Count) 'Selected auxiliary workloads were omitted from job emission.'
    }
}
finally {
    $env:GITHUB_OUTPUT = $previousOutput
    if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath }
}
Write-Host 'Workload partitions passed, including five executions of the shipping workflow emitter and six malformed kinds.'
exit 0
