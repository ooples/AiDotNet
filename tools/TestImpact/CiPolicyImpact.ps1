# CI orchestration is validated by its own mandatory contracts. Runtime execution
# changes require the workloads they can affect, not an unconditional repository run.
. "$PSScriptRoot/CiWorkloadKinds.ps1"

function ConvertTo-CiCanonicalValue {
    param($Value)
    if ($Value -is [Collections.IDictionary]) {
        $ordered = [ordered]@{}
        foreach ($key in @($Value.Keys | Sort-Object)) { $ordered[$key] = ConvertTo-CiCanonicalValue $Value[$key] }
        return $ordered
    }
    if ($Value -is [array]) { return ,@($Value | ForEach-Object { ConvertTo-CiCanonicalValue $_ }) }
    return $Value
}

function Get-CiCanonicalJson {
    param($Value)
    ConvertTo-Json -InputObject (ConvertTo-CiCanonicalValue $Value) -Depth 100 -Compress
}

function Get-CiExecutionImpact {
    param(
        [Parameter(Mandatory)] [Collections.IDictionary] $Before,
        [Parameter(Mandatory)] [Collections.IDictionary] $After,
        [Parameter(Mandatory)] [object[]] $Manifest
    )
    $required = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    # Scheduling and reporting metadata do not change compiled code. Everything else
    # at workflow scope (notably env/defaults) applies across execution jobs.
    $metadata = @('name', 'on', 'permissions', 'concurrency', 'run-name', 'jobs')
    foreach ($key in @(@($Before.Keys) + @($After.Keys) | Sort-Object -Unique)) {
        if ($key -notin $metadata -and (Get-CiCanonicalJson $Before[$key]) -cne (Get-CiCanonicalJson $After[$key])) {
            return [pscustomobject]@{ Full = $true; Shards = @(); Reason = "workflow-wide execution setting changed: $key" }
        }
    }
    $controlJobs = @('validation-source', 'codeql', 'cancel-after-build-failure', 'cancel-after-compat-build-failure',
        'select-shards', 'test-regression-analysis', 'sonarcloud', 'size-check', 'ci-test-analysis',
        'validation-gate', 'certify-expensive-validation', 'promote-ci-test-analysis', 'ci-gate', 'certify-validation')
    $runtimeJobs = @{
        'test-net10-sharded' = [CiWorkloadKind]::Tests
        'parameter-enumeration-sweep' = [CiWorkloadKind]::ParameterSweep
        'model-shape-conformance-windows' = [CiWorkloadKind]::ModelShape
    }
    if (-not $Before.Contains('jobs') -or -not $After.Contains('jobs')) { throw 'Workflow has no jobs.' }
    foreach ($job in @(@($Before.jobs.Keys) + @($After.jobs.Keys) | Sort-Object -Unique)) {
        if ($job -in $controlJobs) { continue }
        $old = $Before.jobs[$job]
        $new = $After.jobs[$job]
        if ((Get-CiCanonicalJson $old) -ceq (Get-CiCanonicalJson $new)) { continue }
        if (-not $runtimeJobs.ContainsKey($job) -or $null -eq $old -or $null -eq $new) {
            return [pscustomobject]@{ Full = $true; Shards = @(); Reason = "build or unknown execution job changed: $job" }
        }
        # Conditions/dependencies select when a job runs; their correctness belongs to
        # the workflow contracts. Keep strategy, env, steps, runner and timeouts in the
        # execution fingerprint: a changed filter, SDK or runtime flag must rerun its family.
        $oldExecution = @{}; $newExecution = @{}
        foreach ($key in $old.Keys) { if ($key -notin @('name', 'if', 'needs', 'permissions')) { $oldExecution[$key] = $old[$key] } }
        foreach ($key in $new.Keys) { if ($key -notin @('name', 'if', 'needs', 'permissions')) { $newExecution[$key] = $new[$key] } }
        if ((Get-CiCanonicalJson $oldExecution) -cne (Get-CiCanonicalJson $newExecution)) {
            foreach ($shard in $Manifest) {
                if ((Get-CiWorkloadKind $shard) -eq $runtimeJobs[$job]) { [void] $required.Add([string] $shard.name) }
            }
        }
    }
    return [pscustomobject]@{ Full = $false; Shards = @($required | Sort-Object); Reason = 'execution boundaries compared' }
}

function Read-CiGitYaml {
    param([string] $Revision, [string] $Path)
    $source = @(& git show "${Revision}:$Path" 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "Cannot read $Path at $Revision" }
    $json = $source -join "`n" | & yq -o=json -I=0 '.' '-'
    if ($LASTEXITCODE -ne 0) { throw "Cannot parse $Path at $Revision" }
    return ($json | ConvertFrom-Json -AsHashtable)
}

function Get-ReviewedCiPolicyImpact {
    param([Parameter(Mandatory)] [string] $BaseSha, [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Paths)
    $handled = [Collections.Generic.List[string]]::new()
    $required = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $manifest = $null
    $tooling = @('Select-Shards.ps1', 'CiPolicyImpact.ps1', 'CiWorkloadKinds.ps1', 'AuxiliaryInventory.ps1',
        'Assert-CiGate.ps1', 'Resolve-CiValidationReuse.ps1', 'Connect-WorkerCoverage.ps1', 'Write-AuxiliaryEvidence.ps1')
    foreach ($path in $Paths) {
        $normalized = $path.Replace('\', '/')
        if ($normalized -cmatch '^tools/TestImpact/Test-[A-Za-z0-9-]+\.ps1$' -or
            $normalized -cin @($tooling | ForEach-Object { "tools/TestImpact/$_" }) -or
            $normalized -cmatch '^tools/TestImpact/fixtures/WorkerCoverage/(?:Directory\.(?:Build|Packages)\.props|(?:Parent|Shared|Worker)/[A-Za-z]+\.(?:cs|csproj))$' -or
            $normalized -ceq '.github/workflows/test-impact-map.yml') {
            $handled.Add($path)
            # Changes to child instrumentation/evidence affect every auxiliary profile.
            if ($normalized -cin @('tools/TestImpact/Connect-WorkerCoverage.ps1', 'tools/TestImpact/Write-AuxiliaryEvidence.ps1')) {
                if ($null -eq $manifest) { $manifest = (Read-CiGitYaml HEAD '.github/test-shards.yml').shard }
                foreach ($shard in $manifest) { if ((Get-CiWorkloadKind $shard) -ne [CiWorkloadKind]::Tests) { [void] $required.Add([string] $shard.name) } }
            }
            continue
        }
        if ($normalized -cnotin @('.github/workflows/sonarcloud.yml', '.github/test-shards.yml')) { continue }
        if ($null -eq $manifest) { $manifest = (Read-CiGitYaml HEAD '.github/test-shards.yml').shard }
        $before = Read-CiGitYaml $BaseSha $normalized
        $after = Read-CiGitYaml HEAD $normalized
        if ($normalized -ceq '.github/workflows/sonarcloud.yml') {
            $impact = Get-CiExecutionImpact -Before $before -After $after -Manifest $manifest
            if ($impact.Full) { Write-Host "Full execution remains required: $($impact.Reason)"; continue }
            foreach ($name in $impact.Shards) { [void] $required.Add($name) }
        }
        else {
            $unknownSetting = @(@($before.Keys) + @($after.Keys) | Sort-Object -Unique | Where-Object {
                $_ -cne 'shard' -and (Get-CiCanonicalJson $before[$_]) -cne (Get-CiCanonicalJson $after[$_])
            })
            if ($unknownSetting.Count -gt 0) { continue }
            $old = @{}; $current = @{}
            foreach ($shard in $before.shard) { if ($old.ContainsKey($shard.name)) { throw 'Duplicate old shard name.' }; $old[$shard.name] = $shard }
            foreach ($shard in $manifest) { if ($current.ContainsKey($shard.name)) { throw 'Duplicate current shard name.' }; $current[$shard.name] = $shard }
            if (@($old.Keys | Where-Object { -not $current.ContainsKey($_) }).Count -gt 0) { continue }
            foreach ($name in $current.Keys) {
                if ((Get-CiCanonicalJson $old[$name]) -cne (Get-CiCanonicalJson $current[$name])) { [void] $required.Add($name) }
            }
        }
        $handled.Add($path)
    }
    return [pscustomobject]@{ Paths = $handled.ToArray(); Shards = @($required | Sort-Object) }
}
