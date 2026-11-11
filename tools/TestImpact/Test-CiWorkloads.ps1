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
# Get-DeferredNightlyShards: which selected nightlyOnly shards a pull request or push may leave to the
# nightly run. Each case is one reason the selector can give, taken from real runs.
$unit = [pscustomobject]@{ name = 'Unit - Finance' }
$swA = [pscustomobject]@{ name = 'Sweep - Count 0/8'; nightlyOnly = $true }
$swB = [pscustomobject]@{ name = 'Sweep - Count 1/8'; nightlyOnly = $true }
$conf = [pscustomobject]@{ name = 'Conformance - offset 0'; nightlyOnly = $true }
$picked = @($unit, $swA, $swB, $conf)
function Get-Deferred([string[]] $Routes) { @(Get-DeferredNightlyShards -Shards $picked -Routes $Routes) | Sort-Object }
Assert-True (@(Get-DeferredNightlyShards -Shards $picked -Routes $null).Count -eq 0) `
    'Nightly shards were deferred without selector routes, where no reason for them is known.'
Assert-True (((Get-Deferred @('Sweep - Count 0/8 <= is always run', 'Sweep - Count 1/8 <= is always run',
        'Conformance - offset 0 <= is always run', 'Unit - Finance <= is always run')) -join '|') -ceq
        'Conformance - offset 0|Sweep - Count 0/8|Sweep - Count 1/8') `
    'Always-run nightly shards were not all deferred, or a non-nightly shard was.'
Assert-True (-not ((Get-Deferred @('Sweep - Count 0/8 <= its manifest or execution policy changed',
        'Sweep - Count 1/8 <= is always run')) -contains 'Sweep - Count 0/8')) `
    'A sweep whose own definition changed was deferred, so the change redefining it would never run it.'
Assert-True ((Get-Deferred @('Conformance - offset 0 <= runs tests affected by tests/Finance/MaskTests.cs (its filter selects tests in this file)',
        'Unit - Finance <= runs tests affected by tests/Finance/MaskTests.cs (its filter selects tests in this file)')) -contains 'Conformance - offset 0') `
    'A broad sweep filter kept a Conformance window although another running shard runs the changed test file.'
$onlyNightly = Get-Deferred @('Sweep - Count 0/8 <= runs tests affected by tests/Sweeps/CountTests.cs (its filter selects tests in this file)',
    'Sweep - Count 1/8 <= runs tests affected by tests/Sweeps/CountTests.cs (its filter selects tests in this file)')
Assert-True (-not ($onlyNightly -contains 'Sweep - Count 0/8') -and -not ($onlyNightly -contains 'Sweep - Count 1/8')) `
    'A changed test file only sweeps run was deferred, or only one of the shards that split it was kept.'
Assert-True (@(Get-DeferredNightlyShards -Shards @($unit) -Routes @('Unit - Finance <= is always run')).Count -eq 0) `
    'A shard that is not nightlyOnly was deferred.'

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
$retiring = [pscustomobject]@{ knownShards = @('Ordinary', 'Gone'); alwaysRun = @('Count', 'Dropped'); files = [pscustomobject]@{} }
$retired = Complete-CiMapWorkloads -Map $retiring -Manifest $all
Assert-True (($retired.Retired -join ',') -ceq 'Dropped,Gone' -and ($retired.RetiredIndexed -join ',') -ceq 'Gone') `
    'Retired workloads were not reported, or an always-run one was reported as indexed.'
Assert-True (($retired.Map.alwaysRun -join ',') -ceq 'Count,Shape') `
    'A retired always-run workload was kept, or a new one was not made mandatory.'
Assert-True (($retired.Map.knownShards -join ',') -ceq 'Ordinary,Gone') `
    'A retired indexed workload lost its position, which would misattribute every later file index entry.'
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
# The Build job's narrowed ledger comes first; select-shards' is the fallback when it published none.
# #2369 moved this value out of the environment to avoid Linux's per-string size limit.
$matrixWriter = [regex]::Match($workflow,
    '(?ms)^      - name: Write expected shard matrix\r?\n        shell: pwsh\r?\n        run: \|\r?\n(?<code>.*?)(?=^      - name:)')
Assert-True $matrixWriter.Success 'The file-based expected shard matrix writer is missing.'
$matrixExpression = '${{ needs.build.outputs.ledger_matrix || needs.select-shards.outputs.ledger_matrix }}'
Assert-True ($matrixWriter.Groups['code'].Value.Contains($matrixExpression)) `
    'Regression inventory does not require imported ordinary shards.'
Assert-True ($workflow -notmatch '(?m)^ +EXPECTED_SHARD_MATRIX:') `
    'The large shard matrix must not be transported through the environment.'
$matrixReader = [regex]::Match($workflow,
    '(?ms)^          \$matrixJson = Get-Content -LiteralPath expected-shard-matrix\.json -Raw\r?\n.*?(?=^          "inventory_valid=true")')
Assert-True $matrixReader.Success 'Regression analysis no longer reads the expected matrix file.'
$readerCode = [regex]::Replace($matrixReader.Value, '(?m)^          ', '').Replace(
    './.github/scripts/get-selected-shard-names.ps1',
    "'$PSScriptRoot/../../.github/scripts/get-selected-shard-names.ps1'")
$transportRoot = Join-Path ([IO.Path]::GetTempPath()) "ci-shard-transport-$([Guid]::NewGuid().ToString('N'))"
$null = New-Item -ItemType Directory -Path $transportRoot
Push-Location $transportRoot
try {
    # Execute the shipping writer and reader with an imported ledger large enough to exceed 128 KiB.
    foreach ($matrix in @('[]', (ConvertTo-Json -InputObject @([pscustomobject]@{ name = 'Ordinary'; filter = ('x' * 140000) }) -Compress))) {
        $writerCode = [regex]::Replace($matrixWriter.Groups['code'].Value, '(?m)^          ', '').Replace($matrixExpression, $matrix)
        & ([scriptblock]::Create($writerCode))
        $emptyRejected = $false
        try { . ([scriptblock]::Create($readerCode)) }
        catch { if ($matrix -eq '[]') { $emptyRejected = $true } else { throw } }
        Assert-True ($matrixJson.Trim() -ceq $matrix) 'The expected ledger was truncated during file transport.'
        if ($matrix -eq '[]') { Assert-True $emptyRejected 'An empty ordinary inventory was silently accepted.' }
        else { Assert-True ($expectedShardNames.Count -eq 1 -and $expectedShardNames[0] -ceq 'Ordinary') `
            'File transport lost imported ordinary evidence.' }
    }
}
finally {
    Pop-Location
    Remove-Item -LiteralPath (Join-Path $transportRoot 'expected-shard-matrix.json') -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $transportRoot
}
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
# Set-CiShardRunners: a few ordinary shards go to the self-hosted label; heavy, custom-timeout and
# the hosted majority stay put, and the caller's objects are not modified.
$plain = 1..6 | ForEach-Object { [pscustomobject]@{ name = "Plain $_" } }
$heavy = [pscustomobject]@{ name = 'Heavy'; heavy = $true }
$slow = [pscustomobject]@{ name = 'Slow'; timeout = 90 }
$shards = @($heavy, $slow) + $plain
$routed = @(Set-CiShardRunners -Shards $shards -Count 3)
Assert-True ($routed.Count -eq 8) 'Routing changed the number of shards.'
Assert-True ((@($routed | Where-Object { $_.PSObject.Properties['runner'] } | ForEach-Object name) -join '|') -ceq
    'Plain 1|Plain 2|Plain 3') 'Routing picked the wrong shards.'
Assert-True (@($shards | Where-Object { $_.PSObject.Properties['runner'] }).Count -eq 0) 'Routing modified the input shards.'
Assert-True (@((Set-CiShardRunners -Shards $shards -Count 99) | Where-Object { $_.PSObject.Properties['runner'] }).Count -eq 4) `
    'Routing sent more than half the matrix to the self-hosted runner.'
Assert-True (@((Set-CiShardRunners -Shards $shards -Count 0) | Where-Object { $_.PSObject.Properties['runner'] }).Count -eq 0) `
    'Routing with a zero count still routed shards.'
Assert-True (@(Set-CiShardRunners -Shards @() -Count 3).Count -eq 0) 'Routing an empty selection produced shards.'
# Get-SelfHostedShardPlan: only online runners carrying the label take shards; anything unusable routes none.
function New-Runner([string] $Status, [string[]] $Labels) {
    [pscustomobject]@{ status = $Status; labels = @($Labels | ForEach-Object { [pscustomobject]@{ name = $_ } }) }
}
$fleet = @((New-Runner 'online' @('self-hosted', 'cpu-a')), (New-Runner 'online' @('self-hosted', 'cpu-a')),
    (New-Runner 'offline' @('self-hosted', 'cpu-a')), (New-Runner 'online' @('self-hosted', 'gpu')))
$plan = Get-SelfHostedShardPlan -Config '{"label":"cpu-a","shardsPerRunner":5}' -Runners $fleet
Assert-True ($plan.Count -eq 10 -and $plan.Label -ceq 'cpu-a' -and -not $plan.Warning) `
    'Two online runners at five shards each should take ten.'
Assert-True ((Get-SelfHostedShardPlan -Config '{"label":"cpu-a","shardsPerRunner":5}' -Runners $fleet -Pending 4).Count -eq 6) `
    'Jobs already queued for the label did not reduce the routed count.'
Assert-True ((Get-SelfHostedShardPlan -Config '{"label":"cpu-a","shardsPerRunner":5}' -Runners $fleet -Pending 25).Count -eq 0) `
    'A backlog larger than the capacity still routed shards.'
Assert-True ((Get-SelfHostedShardPlan -Config '{"label":"cpu-a","shardsPerRunner":5}' -Runners $fleet -Pending -3).Count -eq 10) `
    'A negative pending count raised the routed count.'
Assert-True ((Get-SelfHostedPendingJobs -Repository 'o/r' -Token '' -Label 'cpu-a') -eq 0) 'A missing token produced a pending count.'
Assert-True ((Get-SelfHostedShardPlan -Config '{"label":"CPU-A","shardsPerRunner":5}' -Runners $fleet).Count -eq 0) `
    'A label matched a runner label of another case.'
Assert-True ((Get-SelfHostedShardPlan -Config '{"label":"cpu-a","shardsPerRunner":5}' -Runners @()).Count -eq 0) `
    'Shards went to a label with no runners.'
$unset = Get-SelfHostedShardPlan -Config '' -Runners $fleet
Assert-True ($unset.Count -eq 0 -and -not $unset.Warning) 'An unset configuration should route nothing, quietly.'
foreach ($bad in @('not json', '{"shardsPerRunner":5}', '{"label":"a,b","shardsPerRunner":5}', '{"label":"cpu-a"}',
        '{"label":"cpu-a","shardsPerRunner":0}', '{"label":"cpu-a","shardsPerRunner":"many"}')) {
    $plan = Get-SelfHostedShardPlan -Config $bad -Runners $fleet
    Assert-True ($plan.Count -eq 0 -and $plan.Warning) "Malformed configuration $bad routed shards or said nothing."
}
$noList = Get-SelfHostedShardPlan -Config '{"label":"cpu-a","shardsPerRunner":5}' -Runners $null
Assert-True ($noList.Count -eq 0 -and $noList.Warning) 'An unreadable runner list routed shards or said nothing.'
Assert-True ($null -eq (Get-SelfHostedRunners -Repository 'o/r' -Token '')) 'A missing token produced a runner list.'
Write-Host 'Workload partitions passed, including five executions of the shipping workflow emitter, six malformed kinds and six nightly-deferral cases, shard runner routing and self-hosted planning.'
exit 0
