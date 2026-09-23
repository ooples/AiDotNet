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
    # A workload the manifest no longer has is retired. An always-run entry carries no coverage and
    # is simply dropped. An indexed one keeps its place, because the file index addresses shards by
    # position; Select-Shards escalates any change that reaches it, since nothing records where its
    # tests went.
    $retired = @($existing | Where-Object { -not $current.Contains($_) } | Sort-Object)
    $completed = $Map.PSObject.Copy()
    $completed.alwaysRun = @(@($Map.alwaysRun) | Where-Object { $current.Contains([string] $_) }) + $added.ToArray()
    # A retired INDEXED shard cannot leave knownShards - the file index addresses shards by
    # position - but no job will ever report an outcome for it again. Record the retirement so
    # certification can require outcomes for the live shards only. Without this the first
    # retirement wedges every later map: certification demands an outcome that cannot exist, and
    # the previous map only advances when something certifies. (2026-09-17 retirement -> no
    # certified map from 09-20 on -> every PR and every master push ran the full 164-shard matrix.)
    $retiredIndexed = @($retired | Where-Object { $_ -cin @($Map.knownShards) })
    $completed | Add-Member -NotePropertyName retiredShards -NotePropertyValue $retiredIndexed -Force
    # This is a conservative in-memory extension, not newly measured coverage and not
    # a certificate for a larger historical run. The original artifact is untouched.
    if ($added.Count -gt 0) {
        $completed | Add-Member -NotePropertyName requiredWorkloadExtension -NotePropertyValue $added.ToArray() -Force
    }
    return [pscustomobject]@{
        Map = $completed
        Added = $added.ToArray()
        Retired = $retired
        RetiredIndexed = $retiredIndexed
    }
}

function Get-DeferredNightlyShards {
    <#
    .SYNOPSIS
        Names of the selected nightlyOnly shards a pull request or push can leave to the nightly run.
    .DESCRIPTION
        Sweeps and conformance windows (nightlyOnly in test-shards.yml) exercise every model, so the
        coverage map attributes almost any model change to them: 46 of them sat on every pull request
        and every post-merge push. They run nightly instead, in the coverage-everywhere run on master.

        A nightlyOnly shard is KEPT when the change can only be proved by running it:
         - its manifest or execution policy changed. Without this the change that introduced
           nightlyOnly deferred itself: the selector required all 46 for their redefinition and a
           route-blind filter removed exactly those, leaving 2 shards to validate it.
         - it runs a changed test file that no non-nightly selected shard also runs. Sweep filters
           are broad - #2226's new Finance test file was routed to every Conformance window - so that
           route alone does not mean the change touched the sweep. Every owner is kept, because the
           eight ParameterCountContractTests shards each run a different eighth of one sweep.
        Everything else (always-run, coverage attribution, map bookkeeping) defers.

        Routes are the selector's "<shard> <= <why>" lines. Null routes - any selection path other
        than the mapped selector - defer nothing.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Shards,
        [AllowNull()] [string[]] $Routes
    )
    if ($null -eq $Routes) { return @() }
    $isNightly = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($shard in $Shards) {
        if ($shard.PSObject.Properties['nightlyOnly'] -and $shard.nightlyOnly -eq $true) {
            [void] $isNightly.Add([string] $shard.name)
        }
    }
    if ($isNightly.Count -eq 0) { return @() }
    $keep = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    $testOwners = @{}
    foreach ($route in $Routes) {
        $m = [regex]::Match([string] $route, '^(?<shard>.+?) <= (?<why>.*)$')
        if (-not $m.Success) { continue }
        $name = $m.Groups['shard'].Value
        $why = $m.Groups['why'].Value
        if ($why -ceq 'its manifest or execution policy changed') { [void] $keep.Add($name) }
        $t = [regex]::Match($why, '^runs tests affected by (?<path>\S+)')
        if ($t.Success) {
            $path = $t.Groups['path'].Value
            if (-not $testOwners.ContainsKey($path)) {
                $testOwners[$path] = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
            }
            [void] $testOwners[$path].Add($name)
        }
    }
    foreach ($path in @($testOwners.Keys)) {
        $owners = $testOwners[$path]
        if (@($owners | Where-Object { -not $isNightly.Contains($_) }).Count -eq 0) {
            foreach ($owner in $owners) { [void] $keep.Add($owner) }
        }
    }
    return @($Shards | Where-Object {
        $isNightly.Contains([string] $_.name) -and -not $keep.Contains([string] $_.name)
    } | ForEach-Object { [string] $_.name })
}
