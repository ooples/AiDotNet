[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/CiPolicyImpact.ps1"
function Assert-Impact([bool] $Condition, [string] $Message) { if (-not $Condition) { throw $Message } }
function Copy-Policy($Value) { Get-CiCanonicalJson $Value | ConvertFrom-Json -AsHashtable }
$manifest = @(
    [pscustomobject]@{ name = 'Ordinary' },
    [pscustomobject]@{ name = 'Sweep'; workload = 'ParameterSweep' },
    [pscustomobject]@{ name = 'Shape'; workload = 'ModelShape' })
$baseline = @{
    env = @{ DOTNET_gcServer = '0' }
    jobs = @{
        build = @{ steps = @(@{ run = 'dotnet build' }) }
        'test-net10-sharded' = @{ if = 'runtime'; steps = @(@{ run = 'dotnet test' }); strategy = @{ matrix = 'selected' } }
        'parameter-enumeration-sweep' = @{ steps = @(@{ run = 'dotnet test --filter Sweep' }) }
        'model-shape-conformance-windows' = @{ steps = @(@{ run = 'dotnet test --filter Shape' }) }
        'select-shards' = @{ steps = @(@{ run = 'Select-Shards' }) }
    }
}
$after = Copy-Policy $baseline
$after.jobs['select-shards'].steps[0].run = 'Select-Shards-with-more-contracts'
$after.jobs['test-net10-sharded']['if'] = 'runtime && selected-tests'
$result = Get-CiExecutionImpact $baseline $after $manifest
Assert-Impact (-not $result.Full -and $result.Shards.Count -eq 0) 'Selection-only changes widened model execution.'
foreach ($case in @(
    @{ Job = 'test-net10-sharded'; Expected = 'Ordinary' },
    @{ Job = 'parameter-enumeration-sweep'; Expected = 'Sweep' },
    @{ Job = 'model-shape-conformance-windows'; Expected = 'Shape' }
)) {
    foreach ($property in @('steps', 'env', 'runs-on', 'strategy', 'container', 'defaults', 'timeout-minutes')) {
        $after = Copy-Policy $baseline
        $after.jobs[$case.Job][$property] = 'changed runtime boundary'
        $result = Get-CiExecutionImpact $baseline $after $manifest
        Assert-Impact (-not $result.Full -and ($result.Shards -join ',') -ceq $case.Expected) "Incorrect execution impact for $($case.Job)/$property."
    }
}
foreach ($mutation in @('global-env', 'global-defaults', 'build', 'new-job', 'removed-job', 'unknown-global')) {
    $after = Copy-Policy $baseline
    switch ($mutation) {
        'global-env' { $after.env.DOTNET_gcServer = '1' }
        'global-defaults' { $after.defaults = @{ run = @{ shell = 'bash' } } }
        'build' { $after.jobs.build.steps[0].run = 'dotnet build -p:Changed=true' }
        'new-job' { $after.jobs['unknown'] = @{ steps = @(@{ run = 'anything' }) } }
        'removed-job' { $after.jobs.Remove('test-net10-sharded') }
        'unknown-global' { $after['future-runtime-setting'] = $true }
    }
    $result = Get-CiExecutionImpact $baseline $after $manifest
    Assert-Impact $result.Full "Unsafe workflow mutation was allowed to reduce: $mutation."
}
# Dictionary-backed YAML and object-backed JSON must have identical typed partitions.
$dictionary = @($manifest | ForEach-Object { Copy-Policy ($_ | ConvertTo-Json | ConvertFrom-Json -AsHashtable) })
$result = Get-CiExecutionImpact $baseline $after $dictionary
Assert-Impact $result.Full 'Dictionary YAML lost the full-validation boundary.'
Assert-Impact ((Get-CiWorkloadKind $dictionary[1]) -eq [CiWorkloadKind]::ParameterSweep) 'Dictionary workload kinds were lost.'
# Exercise the path-policy orchestrator with controlled Git/YAML input; production uses
# the exact same comparison and fallback logic after reading the two Git revisions.
function Read-CiGitYaml {
    param([string] $Revision, [string] $Path)
    if ($Path -eq '.github/test-shards.yml') {
        if ($Revision -eq 'HEAD') { return $currentManifest }
        return $oldManifest
    }
    throw 'unexpected YAML input'
}
$oldManifest = @{ shard = @(@{ name = 'Ordinary'; filter = 'old' }, @{ name = 'Other'; filter = 'unchanged' }) }
$currentManifest = Copy-Policy $oldManifest
$currentManifest.shard[0].filter = 'corrected'
$policy = Get-ReviewedCiPolicyImpact -BaseSha 'base' -Paths @('.github/test-shards.yml')
Assert-Impact (($policy.Paths -join ',') -ceq '.github/test-shards.yml' -and ($policy.Shards -join ',') -ceq 'Ordinary') `
    'Changed manifest filter did not require exactly its workload.'
$currentManifest.shard += @{ name = 'New'; filter = 'new' }
$policy = Get-ReviewedCiPolicyImpact -BaseSha 'base' -Paths @('.github/test-shards.yml')
Assert-Impact (($policy.Shards -join ',') -ceq 'New,Ordinary') 'New manifest workload was omitted.'
$currentManifest = @{ shard = @(@{ name = 'Ordinary'; filter = 'old' }) }
$policy = Get-ReviewedCiPolicyImpact -BaseSha 'base' -Paths @('.github/test-shards.yml')
Assert-Impact ($policy.Paths.Count -eq 0) 'Removing a workload bypassed full validation.'
$currentManifest = Copy-Policy $oldManifest
$currentManifest['unknown-runtime-setting'] = $true
$policy = Get-ReviewedCiPolicyImpact -BaseSha 'base' -Paths @('.github/test-shards.yml')
Assert-Impact ($policy.Paths.Count -eq 0) 'Unknown manifest semantics were ignored.'
$policy = Get-ReviewedCiPolicyImpact -BaseSha 'base' -Paths @('tools/TestImpact/UnknownHelper.ps1', 'Directory.Build.props', 'src/AiDotNet.Generators/Generator.cs')
Assert-Impact ($policy.Paths.Count -eq 0) 'Unknown tooling or build-time code bypassed full validation.'
$policy = Get-ReviewedCiPolicyImpact -BaseSha 'base' -Paths @('tools/TestImpact/Select-Shards.ps1', 'tools/TestImpact/Test-CiPolicyImpact.ps1')
Assert-Impact ($policy.Paths.Count -eq 2 -and $policy.Shards.Count -eq 0) 'Explicitly validated tooling required model execution.'
$currentManifest = @{ shard = @(@{ name = 'Ordinary'; filter = 'old' }, @{ name = 'Ordinary'; filter = 'other' }) }
$rejected = $false
try { $null = Get-ReviewedCiPolicyImpact -BaseSha 'base' -Paths @('.github/test-shards.yml') } catch { $rejected = $true }
Assert-Impact $rejected 'Duplicate manifest workloads were accepted.'
Write-Host 'CI execution impact passed: control-only selection, 21 family execution boundaries, six unsafe workflow mutations, typed YAML kinds.'
exit 0
