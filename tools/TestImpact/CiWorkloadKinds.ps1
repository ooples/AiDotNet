# Dot-source from both workflow emission and its regression tests.
enum CiWorkloadKind {
    Tests
    ParameterSweep
    ModelShape
}

function Get-CiWorkloadKind {
    param([Parameter(Mandatory)] [object] $Shard)
    $property = $Shard.PSObject.Properties['workload']
    if ($null -eq $property) { return [CiWorkloadKind]::Tests }
    # Reject numeric enum aliases and misspellings instead of silently dropping a workload.
    if ($property.Value -isnot [string] -or
        $property.Value -cnotin [Enum]::GetNames([CiWorkloadKind])) {
        throw "Unknown workload kind for '$($Shard.name)'."
    }
    return [CiWorkloadKind] $property.Value
}

function Split-CiWorkloads {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Shards)
    $tests = [Collections.Generic.List[object]]::new()
    $sweeps = [Collections.Generic.List[object]]::new()
    $shapes = [Collections.Generic.List[object]]::new()
    foreach ($shard in $Shards) {
        switch (Get-CiWorkloadKind $shard) {
            ([CiWorkloadKind]::Tests) { $tests.Add($shard) }
            ([CiWorkloadKind]::ParameterSweep) { $sweeps.Add($shard) }
            ([CiWorkloadKind]::ModelShape) { $shapes.Add($shard) }
        }
    }
    return [pscustomobject]@{ Tests = $tests.ToArray(); ParameterSweep = $sweeps.ToArray(); ModelShape = $shapes.ToArray() }
}

function Complete-CiWorkloadSelection {
    param(
        [Parameter(Mandatory)] [object[]] $All,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Selected,
        [Parameter(Mandatory)] [bool] $RequiresValidation,
        [Parameter(Mandatory)] [bool] $Escalated
    )
    $allPartitions = Split-CiWorkloads $All
    if ($allPartitions.Tests.Count -eq 0) { throw 'The validation catalog has no ordinary ledger-producing workloads.' }
    $partitions = Split-CiWorkloads $Selected
    if (-not $RequiresValidation -and $Selected.Count -gt 0) { throw 'Non-runtime selection contains workloads.' }
    # Current maps retain mandatory ordinary shards. If a future map removes all of
    # them, an auxiliary-only run cannot mint the ordinary ledger required by reuse.
    # Fail closed rather than emit an empty ledger or silently redefine reporter findings.
    if ($RequiresValidation -and ($Escalated -or $partitions.Tests.Count -eq 0)) {
        return [pscustomobject]@{ Shards = $All; Escalated = $true }
    }
    return [pscustomobject]@{ Shards = $Selected; Escalated = $false }
}

function Complete-CiMapWorkloads {
    param([Parameter(Mandatory)] [object] $Map, [Parameter(Mandatory)] [object[]] $Manifest)
    $existing = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in @($Map.knownShards) + @($Map.alwaysRun)) {
        if (-not $existing.Add([string] $name)) { throw 'Duplicate workload in the source map.' }
    }
    $current = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $added = [Collections.Generic.List[string]]::new()
    foreach ($workload in $Manifest) {
        if (-not $current.Add([string] $workload.name)) { throw 'Duplicate workload in the manifest.' }
        $kind = Get-CiWorkloadKind $workload
        if ($existing.Contains([string] $workload.name)) { continue }
        if ($kind -eq [CiWorkloadKind]::Tests) {
            throw "The map is missing ordinary workload '$($workload.name)'; auxiliary rollout cannot repair that."
        }
        $added.Add([string] $workload.name)
    }
    foreach ($name in $existing) {
        if (-not $current.Contains($name)) { throw "The source map contains an obsolete workload '$name'." }
    }
    $completed = $Map.PSObject.Copy()
    $completed.alwaysRun = @($Map.alwaysRun) + $added.ToArray()
    # This is a conservative in-memory extension, not newly measured coverage and not
    # a certificate for a larger historical run. The original artifact is untouched.
    if ($added.Count -gt 0) {
        $completed | Add-Member -NotePropertyName requiredWorkloadExtension -NotePropertyValue $added.ToArray() -Force
    }
    return [pscustomobject]@{ Map = $completed; Added = $added.ToArray() }
}
