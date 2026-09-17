# Dot-source from both workflow emission and its regression tests.
enum CiWorkloadKind {
    Tests
    ParameterSweep
    ModelShape
}

function Get-CiWorkloadKind {
    param([Parameter(Mandatory)] [object] $Shard)
    if ($Shard -is [Collections.IDictionary]) {
        if (-not $Shard.Contains('workload')) { return [CiWorkloadKind]::Tests }
        $value = $Shard['workload']
    }
    else {
        $property = $Shard.PSObject.Properties['workload']
        if ($null -eq $property) { return [CiWorkloadKind]::Tests }
        $value = $property.Value
    }
    # Reject numeric enum aliases and misspellings instead of silently dropping a workload.
    if ($value -isnot [string] -or
        $value -cnotin [Enum]::GetNames([CiWorkloadKind])) {
        throw "Unknown workload kind for '$($Shard.name)'."
    }
    return [CiWorkloadKind] $value
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
        [Parameter(Mandatory)] [bool] $Escalated,
        [string[]] $ImportedNames = @()
    )
    $allPartitions = Split-CiWorkloads $All
    if ($allPartitions.Tests.Count -eq 0) { throw 'The validation catalog has no ordinary ledger-producing workloads.' }
    $partitions = Split-CiWorkloads $Selected
    $imported = @($All | Where-Object { $_.name -cin $ImportedNames })
    if ($imported.Count -ne $ImportedNames.Count -or
        @($ImportedNames | Where-Object { $_ -cin @($Selected | ForEach-Object name) }).Count -gt 0) {
        throw 'Imported workloads must be unique catalog entries disjoint from reruns.'
    }
    if (-not $RequiresValidation -and ($Selected.Count + $imported.Count) -gt 0) { throw 'Non-runtime selection contains workloads.' }
    $ledgerShards = @($partitions.Tests) + @((Split-CiWorkloads $imported).Tests)
    # Imported ordinary results satisfy the ledger requirement without being rerun.
    # They must also be expected downstream, so missing imports cannot produce green evidence.
    if ($RequiresValidation -and ($Escalated -or $ledgerShards.Count -eq 0)) {
        return [pscustomobject]@{ Shards = $All; LedgerShards = $allPartitions.Tests; Escalated = $true }
    }
    return [pscustomobject]@{ Shards = $Selected; LedgerShards = $ledgerShards; Escalated = $false }
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
        # Validates the kind even for mapped workloads, so a misspelling cannot slip through.
        $null = Get-CiWorkloadKind $workload
        if ($existing.Contains([string] $workload.name)) { continue }
        # A workload the map has never measured runs on every selection until a map includes it.
        # Throwing here instead sent EVERY pull request to the full matrix from the moment any
        # ordinary shard was added until a new map was certified - and certification replays
        # through this same function, so the map could not catch up either (116 -> 209 shards,
        # 2026-09-15..17, zero reduced PRs).
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
