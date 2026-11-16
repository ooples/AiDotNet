# Rolling per-shard coverage-map refresh. Dot-source; Test-RollingShardMap.ps1 proves every rule here.
#
# THE PROBLEM. The certified shard map is rebuilt only by the weekly coverage-everywhere run, so by the end of
# the week every shard's coverage is up to eight days old. Selection then leans on its two widening rules (the
# stale-map sweep for files master changed since the map, and the directory rule for files the map has never
# seen) and a typical pull request selects most of the matrix.
#
# THE DESIGN. Each digest records the commit it was measured at. The nightly full run instruments a budgeted
# subset of the heavy shards (the light ones collect coverage every night anyway), and the map workflow
# overlays the newest valid digest per shard on the certified weekly map - the ANCHOR. The result is a rolling
# map: one ordinary schema-1 sub-map (a "layer") per distinct measured commit.
#
# THE HAZARD. A map mixing coordinate systems - ranges measured at one commit read against a diff from another -
# is the failure this feature has been bitten by twice (Select-CoverageShards.ps1, .DESCRIPTION). A rolling map
# holds several coordinate systems on purpose, so they are never merged: a layer's ranges are only ever compared
# with the change translated to that layer's own commit (Select-Shards.ps1, Invoke-RollingMapSelection).
#
# THE FLAG. Repository variable TEST_IMPACT_ROLLING_MAP: off (default; unset, empty or unrecognised), shadow, on.

# Named settings. Every one is overridable where it is used; these are the defaults and their reasons.
#
# RollingBudgetMinutes: predicted EXTRA runner-minutes a nightly may spend on instrumentation.
#   Measured 2026-11: coverage-everywhere run 42322440147 spent 4,343 shard-minutes against 1,716 for the
#   ordinary nightly 43146799491 at 165 shards. The 103 heavy shards the map indexes account for 2,468 of the
#   2,627 extra minutes: 24 minutes each on average, median 25, maximum 32 - close to a fixed cost per shard
#   rather than a multiple of its runtime (the per-shard ratio ranges from 1.5x to 11.7x). 450 minutes buys 18
#   shards a night, so all 103 are re-measured in 6 nights, for about 26% on top of the nightly.
# RollingShardExtraMinutes: the per-shard prediction used when no measured cost is supplied (the median above).
# RollingMaxDigestAgeDays: a nightly digest older than this is not used; the shard keeps its anchor entry.
#   The anchor is weekly and one lost weekly run is tolerated elsewhere (14-day artifacts), so 10 days lets a
#   rolling map outlive one late anchor without ever holding a digest from before the previous one.
$script:RollingBudgetMinutes = 450
$script:RollingShardExtraMinutes = 25
$script:RollingMaxDigestAgeDays = 10
# Matrix-entry keys the selection job adds per run; they are not part of a shard's definition.
$script:RollingVolatileShardKeys = @('runner')

function Get-RollingMapMode {
    <# off | shadow | on. Anything else, including unset, is off: an unreadable flag must change nothing. #>
    param([AllowNull()] [AllowEmptyString()] [string] $Value)
    $text = ([string] $Value).Trim().ToLowerInvariant()
    if ($text -ceq 'shadow') { return 'shadow' }
    if ($text -ceq 'on') { return 'on' }
    return 'off'
}

function ConvertTo-CanonicalDefinitionJson {
    <# One text per shard definition however it was loaded (yq, GitHub's toJSON, ConvertFrom-Json). #>
    param([AllowNull()] $Value)

    if ($null -eq $Value) { return 'null' }
    if ($Value -is [string]) { return ConvertTo-Json -InputObject $Value -Compress }
    if ($Value -is [bool]) { return $(if ($Value) { 'true' } else { 'false' }) }
    if ($Value -is [double] -or $Value -is [single] -or $Value -is [decimal]) {
        # GitHub expressions hold every number as a double; 60 and 60.0 are one definition.
        if ([double] $Value -eq [Math]::Floor([double] $Value)) { return ([long] $Value).ToString([Globalization.CultureInfo]::InvariantCulture) }
        return ([double] $Value).ToString('R', [Globalization.CultureInfo]::InvariantCulture)
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $pairs = foreach ($key in @($Value.Keys | ForEach-Object { [string] $_ } | Sort-Object -CaseSensitive)) {
            (ConvertTo-Json -InputObject $key -Compress) + ':' + (ConvertTo-CanonicalDefinitionJson $Value[$key])
        }
        return '{' + (@($pairs) -join ',') + '}'
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $pairs = foreach ($key in @($Value.PSObject.Properties.Name | Sort-Object -CaseSensitive)) {
            (ConvertTo-Json -InputObject $key -Compress) + ':' +
                (ConvertTo-CanonicalDefinitionJson $Value.PSObject.Properties[$key].Value)
        }
        return '{' + (@($pairs) -join ',') + '}'
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $items = foreach ($item in $Value) { ConvertTo-CanonicalDefinitionJson $item }
        return '[' + (@($items) -join ',') + ']'
    }
    return ConvertTo-Json -InputObject $Value -Compress
}

function Get-ShardDefinitionHash {
    <#
        SHA-256 of a shard's manifest entry, without the keys the selection job adds per run. A digest records
        the hash of the matrix entry its job actually ran; the map builder compares it with the manifest entry
        committed at the digest's commit.
    #>
    param([Parameter(Mandatory)] [object] $Shard)

    $definition = [ordered]@{}
    if ($Shard -is [System.Collections.IDictionary]) {
        foreach ($key in $Shard.Keys) { if ([string] $key -cnotin $script:RollingVolatileShardKeys) { $definition[[string] $key] = $Shard[$key] } }
    }
    else {
        foreach ($property in $Shard.PSObject.Properties) {
            if ($property.Name -cnotin $script:RollingVolatileShardKeys) { $definition[$property.Name] = $property.Value }
        }
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes((ConvertTo-CanonicalDefinitionJson $definition))
    return ([BitConverter]::ToString([Security.Cryptography.SHA256]::HashData($bytes)) -replace '-', '').ToLowerInvariant()
}

function Get-RollingFileHash {
    param([Parameter(Mandatory)] [string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-RollingObjectId {
    param([AllowNull()] $Value)
    return ([string] $Value) -cmatch '^[0-9a-f]{40}$'
}

# ------------------------------------------------------------------ nightly budget

function Resolve-RollingBudgetMinutes {
    <# The budget from repository variable TEST_IMPACT_ROLLING_BUDGET_MINUTES; unset or unreadable is the default. #>
    param([AllowNull()] [AllowEmptyString()] [string] $Value)
    $text = ([string] $Value).Trim()
    if (-not $text) { return [double] $script:RollingBudgetMinutes }
    $parsed = 0.0
    if (-not [double]::TryParse($text, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref] $parsed) -or
        [double]::IsNaN($parsed) -or $parsed -lt 0 -or $parsed -gt 100000) {
        Write-Host "::warning::TEST_IMPACT_ROLLING_BUDGET_MINUTES '$text' is not a number of minutes; using $script:RollingBudgetMinutes"
        return [double] $script:RollingBudgetMinutes
    }
    return $parsed
}

function Get-NoCoverageShardLists {
    <#
        The shards that collect no coverage on an ordinary run, read from the script that decides it
        (.github/scripts/Invoke-Shard.ps1: $heavyShards and $performanceShards) plus the manifest's
        heavy: true entries, so there is no second list to drift. Matched on the trimmed line being exactly
        the definition, as the coverage-everywhere parser does.
    #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string[]] $ScriptLines, [Parameter(Mandatory)] [object[]] $Manifest)

    $lists = @{}
    foreach ($marker in @('$heavyShards = @(', '$performanceShards = @(')) {
        $names = [System.Collections.Generic.List[string]]::new()
        $start = -1
        for ($i = 0; $i -lt $ScriptLines.Count; $i++) { if ($ScriptLines[$i].Trim() -ceq $marker) { $start = $i; break } }
        if ($start -lt 0) { throw "the shard script no longer defines $marker" }
        $closed = $false
        for ($i = $start + 1; $i -lt $ScriptLines.Count; $i++) {
            if ($ScriptLines[$i].Trim().StartsWith(')')) { $closed = $true; break }
            foreach ($match in [regex]::Matches($ScriptLines[$i], "'([^']+)'")) { [void] $names.Add($match.Groups[1].Value) }
        }
        if (-not $closed -or $names.Count -eq 0) { throw "the shard script's $marker list could not be read" }
        $lists[$marker] = @($names)
    }
    $heavy = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in $lists['$heavyShards = @(']) { [void] $heavy.Add($name) }
    foreach ($entry in $Manifest) {
        if ($entry.PSObject.Properties['heavy'] -and $entry.heavy -eq $true) { [void] $heavy.Add([string] $entry.name) }
    }
    $timing = @($lists['$performanceShards = @('])
    return [pscustomobject]@{
        # Everything Invoke-Shard.ps1 runs without coverage on a non-pull-request run.
        NoCoverage = @(@($heavy) + $timing | Sort-Object -Unique -CaseSensitive)
        Timing     = $timing
    }
}

function Select-RollingCoverageShards {
    <#
    .SYNOPSIS
        Pure. Which heavy shards tonight's full run instruments, within the budget.
    .DESCRIPTION
        Order: (1) shards the manifest has but the anchor map does not, and indexed shards whose definition
        changed since they were measured - the map knows nothing current about either; (2) shards selection
        picks for the diff since their own measured commit - their coverage is the most out of date; (3) the
        rest, oldest measured commit first. Within a class the oldest measurement goes first, then name, so a
        refreshed shard moves to the back and the whole matrix rotates.

        Never instrumented here, whatever the order says:
          * a light shard: every non-pull-request run already collects its coverage, at no extra cost;
          * a timing shard: instrumentation distorts the ratios it asserts, and this run is the correctness
            net under selection (Invoke-Shard.ps1, $performanceShards);
          * a heavy or timing shard the anchor lists as always-run: the weekly run already tried to instrument
            it and got no digest (Select-CoverageShards.ps1, RunWithoutCoverage). The weekly run's
            ReattemptShards stays the only way back.
        All three still get whatever the weekly coverage-everywhere run gives them.
    .OUTPUTS
        Instrument (ordered names), PredictedExtraMinutes, Candidates (every eligible shard with its class and
        whether the budget took it), Excluded (name and reason).
    #>
    param(
        [Parameter(Mandatory)] [object[]] $Manifest,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $NoCoverageShards,
        [AllowEmptyCollection()] [string[]] $TimingShards = @(),
        [Parameter(Mandatory)] $AnchorMap,
        # shard -> ordinal of its measured commit (larger is newer). Absent means the anchor's commit (0).
        [hashtable] $MeasuredOrder = @{},
        [AllowEmptyCollection()] [string[]] $RedefinedShards = @(),
        [AllowEmptyCollection()] [string[]] $StaleShards = @(),
        [double] $BudgetMinutes = $script:RollingBudgetMinutes,
        [double] $DefaultShardExtraMinutes = $script:RollingShardExtraMinutes,
        [hashtable] $ShardExtraMinutes = @{}
    )

    if ($DefaultShardExtraMinutes -le 0) { throw 'DefaultShardExtraMinutes must be positive' }
    $ordinal = [StringComparer]::Ordinal
    $noCoverage = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in $NoCoverageShards) { if ($name) { [void] $noCoverage.Add([string] $name) } }
    $timing = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in $TimingShards) { if ($name) { [void] $timing.Add([string] $name) } }
    $known = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in @($AnchorMap.knownShards)) { [void] $known.Add([string] $name) }
    $always = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in @($AnchorMap.alwaysRun)) { [void] $always.Add([string] $name) }
    $redefined = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in $RedefinedShards) { if ($name) { [void] $redefined.Add([string] $name) } }
    $stale = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in $StaleShards) { if ($name) { [void] $stale.Add([string] $name) } }

    $candidates = [System.Collections.Generic.List[object]]::new()
    $excluded = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($entry in $Manifest) {
        $name = [string] $entry.name
        if ([string]::IsNullOrWhiteSpace($name)) { throw 'the manifest contains an empty shard name' }
        if (-not $seen.Add($name)) { throw "the manifest contains duplicate shard '$name'" }
        if (-not $noCoverage.Contains($name)) {
            [void] $excluded.Add([pscustomobject]@{ name = $name; why = 'light shard: it collects coverage on every nightly already' })
            continue
        }
        if ($timing.Contains($name)) {
            [void] $excluded.Add([pscustomobject]@{ name = $name; why = 'timing shard: instrumentation distorts what it asserts; weekly run only' })
            continue
        }
        if ($always.Contains($name)) {
            [void] $excluded.Add([pscustomobject]@{ name = $name; why = 'always-run in the anchor: the weekly run could not produce its digest; weekly run only' })
            continue
        }
        $class = 3
        $why = 'oldest measurement first'
        if (-not $known.Contains($name)) { $class = 1; $why = 'not in the anchor map' }
        elseif ($redefined.Contains($name)) { $class = 1; $why = 'redefined since it was measured' }
        elseif ($stale.Contains($name)) { $class = 2; $why = 'selected for the changes since it was measured' }
        $order = if ($MeasuredOrder.ContainsKey($name)) { [long] $MeasuredOrder[$name] } else { 0L }
        $cost = if ($ShardExtraMinutes.ContainsKey($name)) { [double] $ShardExtraMinutes[$name] } else { $DefaultShardExtraMinutes }
        if ($cost -le 0) { throw "predicted extra minutes for '$name' must be positive" }
        [void] $candidates.Add([pscustomobject]@{ name = $name; class = $class; measuredOrder = $order; extraMinutes = $cost; why = $why; taken = $false })
    }

    # An explicit comparison: Sort-Object compares names by culture, and the order here decides what is measured.
    $candidates.Sort([Comparison[object]] {
        param($a, $b)
        if ($a.class -ne $b.class) { return ([int] $a.class).CompareTo([int] $b.class) }
        if ($a.measuredOrder -ne $b.measuredOrder) { return ([long] $a.measuredOrder).CompareTo([long] $b.measuredOrder) }
        return [string]::CompareOrdinal([string] $a.name, [string] $b.name)
    })
    $ordered = @($candidates)

    $spent = 0.0
    $instrument = [System.Collections.Generic.List[string]]::new()
    foreach ($candidate in $ordered) {
        # A shard that does not fit is passed over, not a stop: a cheaper one further down may still fit.
        if ($spent + $candidate.extraMinutes -gt $BudgetMinutes) { continue }
        $spent += $candidate.extraMinutes
        $candidate.taken = $true
        [void] $instrument.Add($candidate.name)
    }
    return [pscustomobject]@{
        Instrument            = @($instrument)
        PredictedExtraMinutes = $spent
        BudgetMinutes         = $BudgetMinutes
        Candidates            = @($ordered)
        Excluded              = @($excluded)
    }
}

# ------------------------------------------------------------------ map structure

function Get-RollingShardHomes {
    <# shard -> index of the layer that holds its coverage. #>
    param([Parameter(Mandatory)] $Map)
    $homes = [System.Collections.Generic.Dictionary[string, int]]::new([StringComparer]::Ordinal)
    $layers = @($Map.layers)
    for ($i = 0; $i -lt $layers.Count; $i++) {
        foreach ($name in @($layers[$i].map.knownShards)) { $homes[[string] $name] = $i }
    }
    return , $homes
}

function Assert-RollingShardMap {
    <#
        Structure only: a rolling map is a list of ordinary schema-1 maps, one per measured commit, over
        disjoint shard sets, plus the anchor's always-run list. Whether it may be USED is the certificate's
        business (Assert-CertifiedRollingShardMap).
    #>
    param([Parameter(Mandatory)] $Map)

    foreach ($required in 'schemaVersion', 'kind', 'anchor', 'alwaysRun', 'shards', 'layers') {
        if (-not $Map.PSObject.Properties[$required]) { throw "rolling map has no '$required' property" }
    }
    if ([string] $Map.schemaVersion -cne '1') { throw "unsupported rolling map schemaVersion '$($Map.schemaVersion)'" }
    if ([string] $Map.kind -cne 'rolling') { throw "rolling map kind is '$($Map.kind)'" }
    foreach ($required in 'sha', 'runId') {
        if (-not $Map.anchor.PSObject.Properties[$required]) { throw "rolling map anchor has no '$required'" }
    }
    if (-not (Test-RollingObjectId $Map.anchor.sha)) { throw 'rolling map anchor sha is not a full lowercase commit id' }
    $anchorRun = 0L
    if (-not [long]::TryParse([string] $Map.anchor.runId, [ref] $anchorRun) -or $anchorRun -lt 1) {
        throw 'rolling map anchor runId must be a positive integer'
    }
    if ($Map.alwaysRun -isnot [array]) { throw 'rolling map alwaysRun must be an array' }
    if ($Map.layers -isnot [array] -or @($Map.layers).Count -eq 0) { throw 'rolling map has no layers' }
    if ($Map.shards -isnot [pscustomobject]) { throw 'rolling map shards must be an object' }

    $ordinal = [StringComparer]::Ordinal
    $universe = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($value in @($Map.alwaysRun)) {
        $name = [string] $value
        if ([string]::IsNullOrWhiteSpace($name)) { throw 'rolling map alwaysRun contains an empty shard name' }
        if (-not $universe.Add($name)) { throw "rolling map alwaysRun contains duplicate '$name'" }
    }
    $layerShas = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($layer in @($Map.layers)) {
        foreach ($required in 'sha', 'map') {
            if (-not $layer.PSObject.Properties[$required]) { throw "a rolling map layer has no '$required'" }
        }
        $sha = [string] $layer.sha
        if (-not (Test-RollingObjectId $sha)) { throw "rolling map layer sha '$sha' is not a full lowercase commit id" }
        # One layer per commit: two layers at one commit would be one coordinate system split in two, and the
        # selector's per-commit grouping would hide whichever came second.
        if (-not $layerShas.Add($sha)) { throw "rolling map has two layers at commit $sha" }
        $inner = $layer.map
        foreach ($required in 'schemaVersion', 'sha', 'knownShards', 'alwaysRun', 'files') {
            if (-not $inner.PSObject.Properties[$required]) { throw "layer $sha map has no '$required'" }
        }
        if ([string] $inner.schemaVersion -cne '1') { throw "layer $sha map has unsupported schemaVersion" }
        # The line that keeps coordinate systems apart: a layer's ranges are in its own commit's numbering.
        if ([string] $inner.sha -cne $sha) { throw "layer $sha holds a map measured at $($inner.sha)" }
        if ($inner.knownShards -isnot [array] -or @($inner.knownShards).Count -eq 0) { throw "layer $sha indexes no shard" }
        if ($inner.alwaysRun -isnot [array] -or @($inner.alwaysRun).Count -ne 0) {
            throw "layer $sha carries its own always-run list; only the anchor's applies"
        }
        if ($inner.files -isnot [pscustomobject] -or @($inner.files.PSObject.Properties).Count -eq 0) {
            throw "layer $sha has an empty file index"
        }
        foreach ($value in @($inner.knownShards)) {
            $name = [string] $value
            if ([string]::IsNullOrWhiteSpace($name)) { throw "layer $sha contains an empty shard name" }
            if (-not $universe.Add($name)) { throw "shard '$name' appears in more than one place in the rolling map" }
            $provenance = $Map.shards.PSObject.Properties[$name]
            if ($null -eq $provenance) { throw "shard '$name' has no recorded measurement" }
            if ([string] $provenance.Value.sha -cne $sha) {
                throw "shard '$name' is recorded as measured at $($provenance.Value.sha) but indexed at $sha"
            }
            if ([string] $provenance.Value.source -cnotin @('anchor', 'nightly')) {
                throw "shard '$name' has unsupported source '$($provenance.Value.source)'"
            }
            if ([string] $provenance.Value.source -ceq 'anchor' -and $sha -cne [string] $Map.anchor.sha) {
                throw "shard '$name' claims the anchor's measurement at another commit"
            }
        }
        $count = @($inner.knownShards).Count
        foreach ($file in $inner.files.PSObject.Properties) {
            if ($file.Value -isnot [array] -or @($file.Value).Count -eq 0) { throw "layer $sha '$($file.Name)' has no occurrences" }
            foreach ($occurrence in @($file.Value)) {
                if (-not $occurrence.PSObject.Properties['s'] -or -not $occurrence.PSObject.Properties['r']) {
                    throw "layer $sha '$($file.Name)' has an occurrence without s or r"
                }
                $index = [long] $occurrence.s
                if ($index -lt 0 -or $index -ge $count) { throw "layer $sha '$($file.Name)' names shard index $index" }
                $ranges = @($occurrence.r)
                if ($ranges.Count -eq 0 -or $ranges.Count % 2 -ne 0) { throw "layer $sha '$($file.Name)' ranges must be start/end pairs" }
            }
        }
    }
    foreach ($provenance in $Map.shards.PSObject.Properties) {
        if (-not $universe.Contains($provenance.Name)) { throw "a measurement is recorded for '$($provenance.Name)', which no layer indexes" }
    }
}

# ------------------------------------------------------------------ map builder

function Test-RollingDigestRanges {
    param([Parameter(Mandatory)] $Digest)
    if ($Digest.files -isnot [pscustomobject]) { return 'its files value is not an object' }
    foreach ($property in $Digest.files.PSObject.Properties) {
        if ([string]::IsNullOrWhiteSpace([string] $property.Name)) { return 'it has an empty file path' }
        if ($property.Value -isnot [array]) { return "'$($property.Name)' ranges are not an array" }
        $ranges = @($property.Value)
        if ($ranges.Count -eq 0 -or $ranges.Count % 2 -ne 0) { return "'$($property.Name)' ranges are not start/end pairs" }
        for ($i = 0; $i -lt $ranges.Count; $i += 2) {
            if ($ranges[$i] -isnot [long] -and $ranges[$i] -isnot [int]) { return "'$($property.Name)' has a non-integer range" }
            if ($ranges[$i + 1] -isnot [long] -and $ranges[$i + 1] -isnot [int]) { return "'$($property.Name)' has a non-integer range" }
            if ([long] $ranges[$i] -lt 1 -or [long] $ranges[$i] -gt [long] $ranges[$i + 1]) { return "'$($property.Name)' has an invalid range" }
        }
    }
    return ''
}

function Test-RollingDigest {
    <#
        Rule (a) for one digest, and the per-digest half of rule (d). Returns '' when the digest may enter a
        rolling map, otherwise why not. Source is the run it was downloaded from.
    #>
    param(
        [Parameter(Mandatory)] $Digest,
        [Parameter(Mandatory)] $Source,
        [AllowNull()] $ManifestEntry
    )

    foreach ($required in 'schemaVersion', 'shard', 'files') {
        if (-not $Digest.PSObject.Properties[$required]) { return "it has no '$required' property" }
    }
    if ([string] $Digest.schemaVersion -cne '1') { return "unsupported schemaVersion '$($Digest.schemaVersion)'" }
    $name = [string] $Digest.shard
    if ([string]::IsNullOrWhiteSpace($name)) { return 'it names no shard' }
    # A carried digest is the previous map restated, not a measurement of this run.
    if ($Digest.PSObject.Properties['carriedFrom']) { return 'it was carried forward from a previous map, not measured by this run' }
    foreach ($required in 'measuredSha', 'runId', 'definitionHash', 'manifestBlob') {
        if (-not $Digest.PSObject.Properties[$required] -or [string]::IsNullOrWhiteSpace([string] $Digest.$required)) {
            return "it does not record its $required"
        }
    }
    # The commit a digest's line numbers are in is the one it says, and it must be the run's.
    if ([string] $Digest.measuredSha -cne [string] $Source.Sha) {
        return "it was measured at $($Digest.measuredSha), not at its run's commit $($Source.Sha)"
    }
    if ([string] $Digest.runId -cne [string] $Source.RunId) { return "it belongs to run $($Digest.runId), not run $($Source.RunId)" }
    $conclusion = if ($Source.Outcomes.ContainsKey($name)) { [string] $Source.Outcomes[$name] } else { '' }
    if ($conclusion -cne 'success') {
        return "its shard job did not complete successfully ($(if ($conclusion) { $conclusion } else { 'no job' }))"
    }
    if ($null -eq $ManifestEntry) { return "the manifest at $($Source.Sha) has no shard of that name" }
    if ([string] $Digest.manifestBlob -cne [string] $Source.ManifestBlob) {
        return 'its job ran with a different shard manifest than the one committed at its commit'
    }
    if ([string] $Digest.definitionHash -cne (Get-ShardDefinitionHash $ManifestEntry)) {
        return 'the shard definition it ran with differs from the manifest at its commit'
    }
    $rangeProblem = Test-RollingDigestRanges -Digest $Digest
    if ($rangeProblem) { return $rangeProblem }
    if (@($Digest.files.PSObject.Properties).Count -eq 0) { return 'it executed no file' }
    if ($ManifestEntry.PSObject.Properties['mustCover']) {
        # Same rule as New-ShardMap.ps1: coverage that never reached the shard's subject is not evidence.
        $reached = $false
        foreach ($property in $Digest.files.PSObject.Properties) {
            foreach ($pattern in @($ManifestEntry.mustCover)) { if ($property.Name -like [string] $pattern) { $reached = $true } }
        }
        if (-not $reached) { return 'it recorded no file matching its mustCover list' }
    }
    return ''
}

function New-RollingShardMapObject {
    <#
    .SYNOPSIS
        Pure. The certified weekly map (the anchor) overlaid with the newest valid nightly digest per shard.
    .DESCRIPTION
        Sources are the nightly runs to draw from, oldest first; each is
          { RunId; Sha; CreatedAt; DescendsFromAnchor; Manifest; ManifestBlob; Outcomes; Digests }
        with Manifest the shard entries committed at Sha, Outcomes a hashtable shard -> job conclusion, and
        Digests the parsed digest objects. The caller establishes ancestry and reads the manifests; nothing
        here touches git or the network.

        A digest that fails any rule is left out and the shard keeps what it had (an older valid digest, else
        its anchor entry) - never a reason to select less, because the anchor entry is the certified one.
        Shards the anchor lists as always-run stay always-run: the anchor's classification stands. A shard the
        anchor has never seen can enter through a valid digest; until then it is unmapped and runs every time.
    .OUTPUTS
        Map (the rolling map), Refreshed (shard -> source run), Rejected ({ shard; runId; reason }).
    #>
    param(
        [Parameter(Mandatory)] $AnchorMap,
        [Parameter(Mandatory)] [long] $AnchorRunId,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Sources,
        [DateTimeOffset] $Now = [DateTimeOffset]::UtcNow,
        [int] $MaxDigestAgeDays = $script:RollingMaxDigestAgeDays
    )

    foreach ($required in 'schemaVersion', 'sha', 'knownShards', 'alwaysRun', 'files') {
        if (-not $AnchorMap.PSObject.Properties[$required]) { throw "the anchor map has no '$required' property" }
    }
    $anchorSha = [string] $AnchorMap.sha
    if (-not (Test-RollingObjectId $anchorSha)) { throw 'the anchor map sha is not a full lowercase commit id' }
    if ($AnchorRunId -lt 1) { throw 'the anchor run id must be positive' }
    $ordinal = [StringComparer]::Ordinal
    $anchorKnown = @($AnchorMap.knownShards | ForEach-Object { [string] $_ })
    $anchorAlways = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in @($AnchorMap.alwaysRun)) { [void] $anchorAlways.Add([string] $name) }

    $rejected = [System.Collections.Generic.List[object]]::new()
    # shard -> @{ Source; Digest }: the newest valid digest wins, so later sources overwrite earlier ones.
    $winners = [System.Collections.Generic.Dictionary[string, object]]::new($ordinal)
    $previous = $null
    foreach ($source in $Sources) {
        $runId = [long] $source.RunId
        $sourceProblem = ''
        if (-not (Test-RollingObjectId $source.Sha)) { $sourceProblem = 'its commit id is not a full lowercase commit id' }
        elseif (-not [bool] $source.DescendsFromAnchor) { $sourceProblem = "its commit $($source.Sha) does not descend from the anchor's commit $anchorSha" }
        elseif (($Now - [DateTimeOffset] $source.CreatedAt) -gt [TimeSpan]::FromDays($MaxDigestAgeDays)) {
            $sourceProblem = "it is older than $MaxDigestAgeDays day(s)"
        }
        elseif ($null -ne $previous -and [DateTimeOffset] $source.CreatedAt -lt [DateTimeOffset] $previous.CreatedAt) {
            throw 'rolling sources must be ordered oldest first'
        }
        $entries = @{}
        foreach ($entry in @($source.Manifest)) { $entries[[string] $entry.name] = $entry }
        $seen = [System.Collections.Generic.HashSet[string]]::new($ordinal)
        foreach ($digest in @($source.Digests)) {
            $name = if ($digest.PSObject.Properties['shard']) { [string] $digest.shard } else { '' }
            $reason = $sourceProblem
            if (-not $reason) { $reason = Test-RollingDigest -Digest $digest -Source $source -ManifestEntry $entries[$name] }
            if (-not $reason -and -not $seen.Add($name)) { $reason = 'its run uploaded more than one digest for the shard' }
            if (-not $reason -and $anchorAlways.Contains($name)) {
                $reason = 'the anchor lists the shard as always-run, and that classification stands until the next anchor'
            }
            if ($reason) {
                [void] $rejected.Add([pscustomobject]@{ shard = $name; runId = $runId; reason = $reason })
                continue
            }
            $winners[$name] = @{ Source = $source; Digest = $digest }
        }
        if (-not $sourceProblem) { $previous = $source }
    }

    # Layers by commit, the anchor's first. A nightly at the anchor's own commit shares its coordinate system.
    $layerShards = [ordered]@{}
    $layerShards[$anchorSha] = [System.Collections.Generic.List[string]]::new()
    $layerRuns = @{ $anchorSha = [System.Collections.Generic.SortedSet[long]]::new() }
    $provenance = [ordered]@{}
    foreach ($name in $anchorKnown) {
        if ($winners.ContainsKey($name)) { continue }
        [void] $layerShards[$anchorSha].Add($name)
        $provenance[$name] = [ordered]@{ sha = $anchorSha; runId = $AnchorRunId; source = 'anchor' }
    }
    foreach ($source in $Sources) {
        $names = @($winners.Keys | Where-Object { [object]::ReferenceEquals($winners[$_].Source, $source) })
        if ($names.Count -eq 0) { continue }
        $sha = [string] $source.Sha
        if (-not $layerShards.Contains($sha)) {
            $layerShards[$sha] = [System.Collections.Generic.List[string]]::new()
            $layerRuns[$sha] = [System.Collections.Generic.SortedSet[long]]::new()
        }
        $sorted = [string[]] $names
        [Array]::Sort($sorted, $ordinal)
        foreach ($name in $sorted) {
            [void] $layerShards[$sha].Add($name)
            [void] $layerRuns[$sha].Add([long] $source.RunId)
            $provenance[$name] = [ordered]@{
                sha = $sha; runId = [long] $source.RunId; source = 'nightly'
                measuredUtc = ([DateTimeOffset] $source.CreatedAt).UtcDateTime.ToString('o', [Globalization.CultureInfo]::InvariantCulture)
            }
        }
    }

    $layers = [System.Collections.Generic.List[object]]::new()
    foreach ($sha in $layerShards.Keys) {
        $names = @($layerShards[$sha])
        if ($names.Count -eq 0) { continue }
        $position = @{}
        for ($i = 0; $i -lt $names.Count; $i++) { $position[$names[$i]] = $i }
        $index = [System.Collections.Generic.SortedDictionary[string, System.Collections.Generic.List[object]]]::new($ordinal)
        if ($sha -ceq $anchorSha) {
            # The anchor's own occurrences, re-addressed to this layer's shard positions. Ranges are untouched.
            foreach ($file in $AnchorMap.files.PSObject.Properties) {
                foreach ($occurrence in @($file.Value)) {
                    $name = [string] $anchorKnown[[int] $occurrence.s]
                    if ($winners.ContainsKey($name) -or -not $position.ContainsKey($name)) { continue }
                    if (-not $index.ContainsKey($file.Name)) { $index[$file.Name] = [System.Collections.Generic.List[object]]::new() }
                    [void] $index[$file.Name].Add([ordered]@{ s = $position[$name]; r = @($occurrence.r) })
                }
            }
        }
        foreach ($name in $names) {
            if (-not $winners.ContainsKey($name)) { continue }
            foreach ($file in $winners[$name].Digest.files.PSObject.Properties) {
                if (-not $index.ContainsKey($file.Name)) { $index[$file.Name] = [System.Collections.Generic.List[object]]::new() }
                [void] $index[$file.Name].Add([ordered]@{ s = $position[$name]; r = @($file.Value) })
            }
        }
        if ($index.Count -eq 0) { throw "layer $sha would have an empty file index" }
        $files = [ordered]@{}
        foreach ($path in $index.Keys) { $files[$path] = @($index[$path]) }
        [void] $layers.Add([ordered]@{
            sha    = $sha
            runIds = @($layerRuns[$sha])
            map    = [ordered]@{
                schemaVersion = 1; sha = $sha; knownShards = @($names); alwaysRun = @()
                fileCount = $files.Count; files = $files
            }
        })
    }
    if ($layers.Count -eq 0) { throw 'the rolling map would index no shard' }

    $refreshed = [ordered]@{}
    foreach ($name in @($winners.Keys | Sort-Object)) { $refreshed[$name] = [long] $winners[$name].Source.RunId }
    $map = [ordered]@{
        schemaVersion = 1
        kind          = 'rolling'
        generatedUtc  = $Now.UtcDateTime.ToString('o', [Globalization.CultureInfo]::InvariantCulture)
        anchor        = [ordered]@{ sha = $anchorSha; runId = $AnchorRunId }
        alwaysRun     = @($AnchorMap.alwaysRun | ForEach-Object { [string] $_ })
        shards        = $provenance
        layers        = @($layers)
    }
    return [pscustomobject]@{ Map = $map; Refreshed = $refreshed; Rejected = @($rejected) }
}

# ------------------------------------------------------------------ certification

function Get-RollingCertificationDecision {
    <#
    .SYNOPSIS
        Pure. Whether a rolling map may decide selection, and the certificate if so.
    .DESCRIPTION
        A rolling map is usable only when ALL of these hold; otherwise the certified weekly map decides alone.
          (a) Every nightly digest in it came from a shard job that completed successfully and ran the shard
              definition committed at the digest's commit. New-RollingShardMapObject leaves out any digest
              that does not; here the map is checked to contain nothing else (every nightly shard must be in
              Refreshed, the builder's own account of what it verified).
          (b) Its anchor is the certified weekly map, and it was built by the run that certified that anchor.
              The selector only ever fetches a rolling map from the run the certified-map resolver returned,
              so a revoked, unclassified or expired anchor takes its rolling map with it.
          (c) Replaying it against the newest nightly's complete outcomes (Measure-SelectionMiss.ps1) finds
              no failed shard it would have skipped, did not escalate, and skipped something.
          (d) No digest is older than MaxDigestAgeDays, and every one descends from the anchor's commit
              (enforced by the builder; the oldest is recorded so selection can re-check the age).
        The certificate binds the exact map bytes by hash.
    #>
    param(
        [Parameter(Mandatory)] $RollingMap,
        [Parameter(Mandatory)] [string] $RollingMapHash,
        [Parameter(Mandatory)] [AllowNull()] $Refreshed,
        [Parameter(Mandatory)] $AnchorMap,
        [Parameter(Mandatory)] $AnchorCertificate,
        [Parameter(Mandatory)] [long] $AnchorRunId,
        [Parameter(Mandatory)] $Audit,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Outcomes,
        [Parameter(Mandatory)] [long] $AuditSourceRunId,
        [Parameter(Mandatory)] [string] $AuditSourceSha,
        [Parameter(Mandatory)] [long] $CertificationRunId,
        [DateTimeOffset] $Now = [DateTimeOffset]::UtcNow,
        [int] $MaxDigestAgeDays = $script:RollingMaxDigestAgeDays
    )

    Assert-RollingShardMap -Map $RollingMap
    if ($RollingMapHash -cnotmatch '^[0-9a-f]{64}$') { throw 'the rolling map hash is not a SHA-256' }
    # (b)
    if ($CertificationRunId -ne $AnchorRunId) { throw 'a rolling map must be certified by the run that certified its anchor' }
    if ([long] $RollingMap.anchor.runId -ne $AnchorRunId) { throw "the rolling map was built on anchor run $($RollingMap.anchor.runId), not $AnchorRunId" }
    if ([string] $RollingMap.anchor.sha -cne [string] $AnchorMap.sha) { throw 'the rolling map and the anchor map identify different commits' }
    foreach ($required in 'certificationRunId', 'candidateMapSha', 'missCount') {
        if (-not $AnchorCertificate.PSObject.Properties[$required]) { throw "the anchor certificate has no $required" }
    }
    if ([string] $AnchorCertificate.certificationRunId -cne [string] $AnchorRunId) { throw 'the anchor certificate belongs to another run' }
    if ([string] $AnchorCertificate.candidateMapSha -cne [string] $AnchorMap.sha) { throw 'the anchor certificate is for another map' }
    if ([string] $AnchorCertificate.missCount -cne '0') { throw 'the anchor certificate records a selection miss' }
    if (-not (Test-RollingObjectId $AuditSourceSha)) { throw 'the audit source sha is not a full lowercase commit id' }
    if ($AuditSourceRunId -lt 1) { throw 'the audit source run id must be positive' }

    # (a), (d): the map holds only what the builder verified, and nothing too old.
    $ordinal = [StringComparer]::Ordinal
    $anchorKnown = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in @($AnchorMap.knownShards)) { [void] $anchorKnown.Add([string] $name) }
    $refreshedCount = 0
    $oldest = $null
    foreach ($provenance in $RollingMap.shards.PSObject.Properties) {
        $name = $provenance.Name
        if ([string] $provenance.Value.source -ceq 'anchor') {
            if (-not $anchorKnown.Contains($name)) { throw "shard '$name' claims an anchor measurement the anchor does not have" }
            continue
        }
        $refreshedCount++
        $verified = if ($null -ne $Refreshed) {
            if ($Refreshed -is [System.Collections.IDictionary]) { $Refreshed[$name] } else { $Refreshed.PSObject.Properties[$name]?.Value }
        }
        if ($null -eq $verified -or [string] $verified -cne [string] $provenance.Value.runId) {
            throw "shard '$name' carries a nightly digest the builder did not verify"
        }
        $measured = [DateTimeOffset]::MinValue
        if (-not [DateTimeOffset]::TryParse([string] $provenance.Value.measuredUtc, [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::AssumeUniversal, [ref] $measured)) {
            throw "shard '$name' has no readable measurement time"
        }
        if (($Now - $measured) -gt [TimeSpan]::FromDays($MaxDigestAgeDays)) { throw "the digest of '$name' is older than $MaxDigestAgeDays day(s)" }
        if ($null -eq $oldest -or $measured -lt $oldest) { $oldest = $measured }
    }
    foreach ($name in $anchorKnown) {
        if (-not $RollingMap.shards.PSObject.Properties[$name]) { throw "the rolling map lost the anchor's shard '$name'" }
    }
    $alwaysRolling = [string[]] @($RollingMap.alwaysRun | ForEach-Object { [string] $_ })
    $alwaysAnchor = [string[]] @($AnchorMap.alwaysRun | ForEach-Object { [string] $_ })
    [Array]::Sort($alwaysRolling, $ordinal); [Array]::Sort($alwaysAnchor, $ordinal)
    if (($alwaysRolling -join "`n") -cne ($alwaysAnchor -join "`n")) { throw "the rolling map changed the anchor's always-run list" }

    # (c): the audit's arithmetic, recomputed from the outcomes rather than believed.
    foreach ($name in 'Escalated', 'TotalShards', 'WouldRun', 'WouldSkip', 'WouldRunShards', 'WouldSkipShards', 'Failed', 'Missed', 'MissCount') {
        if (-not $Audit.PSObject.Properties[$name]) { throw "audit is missing $name" }
    }
    if ($Audit.Escalated -isnot [bool]) { throw 'audit Escalated must be a JSON boolean' }
    $outcomeNames = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    $failedNames = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($outcome in $Outcomes) {
        $name = [string] $outcome.shard
        if (-not $outcomeNames.Add($name)) { throw "duplicate outcome for shard '$name'" }
        $conclusion = [string] $outcome.outcome
        # A cancelled or skipped shard is not evidence in either direction: the nightly was not complete.
        if ($conclusion -cnotin @('success', 'failure')) { throw "shard '$name' has unauditable outcome '$conclusion'" }
        if ($conclusion -ceq 'failure') { [void] $failedNames.Add($name) }
    }
    if ($outcomeNames.Count -lt 2) { throw 'the audited nightly has fewer than two shards' }
    $run = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($value in @($Audit.WouldRunShards)) {
        if (-not $outcomeNames.Contains([string] $value)) { throw "audit WouldRunShards names unknown shard '$value'" }
        if (-not $run.Add([string] $value)) { throw "audit WouldRunShards contains duplicate '$value'" }
    }
    $skip = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($value in @($Audit.WouldSkipShards)) {
        if (-not $outcomeNames.Contains([string] $value)) { throw "audit WouldSkipShards names unknown shard '$value'" }
        if ($run.Contains([string] $value)) { throw "audit shard partitions overlap at '$value'" }
        if (-not $skip.Add([string] $value)) { throw "audit WouldSkipShards contains duplicate '$value'" }
    }
    if ($run.Count + $skip.Count -ne $outcomeNames.Count) { throw 'audit shard partitions do not cover the audited nightly' }
    if ([long] $Audit.TotalShards -ne $outcomeNames.Count) { throw 'audit TotalShards does not match the audited nightly' }
    if ([bool] $Audit.Escalated -and $skip.Count -ne 0) { throw 'an escalated audit must run the complete matrix' }
    $missed = [string[]] @($failedNames | Where-Object { $skip.Contains($_) })
    [Array]::Sort($missed, $ordinal)
    $reported = [string[]] @($Audit.Missed | ForEach-Object { [string] $_ })
    [Array]::Sort($reported, $ordinal)
    if (($missed -join "`n") -cne ($reported -join "`n")) { throw 'audit Missed does not equal the failed shards it would skip' }
    if ([long] $Audit.MissCount -ne $missed.Count) { throw 'audit MissCount does not match the recomputed selection misses' }
    if ([long] $Audit.Failed -ne $failedNames.Count) { throw 'audit Failed does not match the nightly outcomes' }

    $disposition = if ($missed.Count -gt 0) { 'RevokedMiss' }
        elseif ([bool] $Audit.Escalated -or $skip.Count -eq 0 -or $run.Count -eq 0) { 'SafeNoReduction' }
        else { 'Certified' }
    $summary = [ordered]@{
        schemaVersion      = 1
        kind               = 'rolling'
        disposition        = $disposition
        anchorRunId        = $AnchorRunId
        anchorSha          = [string] $AnchorMap.sha
        certificationRunId = $CertificationRunId
        auditSourceRunId   = $AuditSourceRunId
        auditSourceSha     = $AuditSourceSha
        mapHash            = $RollingMapHash
        layers             = @($RollingMap.layers).Count
        refreshedShards    = $refreshedCount
        oldestDigestUtc    = $(if ($null -ne $oldest) { $oldest.UtcDateTime.ToString('o', [Globalization.CultureInfo]::InvariantCulture) } else { $null })
        maxDigestAgeDays   = $MaxDigestAgeDays
        escalated          = [bool] $Audit.Escalated
        auditedShards      = $outcomeNames.Count
        wouldRun           = $run.Count
        wouldSkip          = $skip.Count
        failedShards       = $failedNames.Count
        missCount          = $missed.Count
        missed             = @($missed)
    }
    return [pscustomobject]@{
        Eligible    = $disposition -ceq 'Certified'
        Disposition = $disposition
        Decision    = [pscustomobject] $summary
        Certificate = $(if ($disposition -ceq 'Certified') { [pscustomobject] $summary } else { $null })
    }
}

function Assert-CertifiedRollingShardMap {
    <#
        What selection checks before a rolling map may decide anything: the certificate is this map's (by
        hash), from the run that certified its anchor, on the anchor selection is using right now, with a
        clean reducing audit and no digest past the age limit. Throws with the reason otherwise.
    #>
    param(
        [Parameter(Mandatory)] $RollingMap,
        [Parameter(Mandatory)] [string] $RollingMapHash,
        [Parameter(Mandatory)] $Certificate,
        [Parameter(Mandatory)] $AnchorMap,
        [Parameter(Mandatory)] [long] $AnchorRunId,
        [DateTimeOffset] $Now = [DateTimeOffset]::UtcNow,
        [int] $MaxDigestAgeDays = $script:RollingMaxDigestAgeDays
    )

    foreach ($name in 'schemaVersion', 'kind', 'disposition', 'anchorRunId', 'anchorSha', 'certificationRunId', 'mapHash',
        'escalated', 'auditedShards', 'wouldRun', 'wouldSkip', 'missCount', 'maxDigestAgeDays') {
        if (-not $Certificate.PSObject.Properties[$name]) { throw "rolling certificate is missing $name" }
    }
    if ([string] $Certificate.schemaVersion -cne '1' -or [string] $Certificate.kind -cne 'rolling') { throw 'unsupported rolling certificate' }
    if ([string] $Certificate.disposition -cne 'Certified') { throw "rolling certificate disposition is '$($Certificate.disposition)'" }
    if ([string] $Certificate.certificationRunId -cne [string] $AnchorRunId -or [string] $Certificate.anchorRunId -cne [string] $AnchorRunId) {
        throw "rolling certificate belongs to run $($Certificate.certificationRunId) on anchor $($Certificate.anchorRunId), not the anchor in use ($AnchorRunId)"
    }
    if ([string] $Certificate.anchorSha -cne [string] $AnchorMap.sha) { throw 'rolling certificate is for another anchor commit' }
    if ([string] $Certificate.mapHash -cne $RollingMapHash) { throw 'rolling certificate does not match the rolling map bytes' }
    if ([string] $Certificate.missCount -cne '0') { throw "rolling certificate records $($Certificate.missCount) selection miss(es)" }
    if ($Certificate.escalated -isnot [bool] -or [bool] $Certificate.escalated) { throw 'rolling certificate records an escalated audit' }
    if ([long] $Certificate.wouldSkip -lt 1 -or [long] $Certificate.wouldRun -lt 1) { throw 'rolling certificate does not prove a reduced selection' }
    if ([long] $Certificate.wouldRun + [long] $Certificate.wouldSkip -ne [long] $Certificate.auditedShards) {
        throw 'rolling certificate partitions do not cover its audited shards'
    }

    Assert-RollingShardMap -Map $RollingMap
    if ([long] $RollingMap.anchor.runId -ne $AnchorRunId -or [string] $RollingMap.anchor.sha -cne [string] $AnchorMap.sha) {
        throw 'the rolling map was not built on the anchor in use'
    }
    $ordinal = [StringComparer]::Ordinal
    $anchorKnown = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in @($AnchorMap.knownShards)) { [void] $anchorKnown.Add([string] $name) }
    foreach ($name in $anchorKnown) {
        if (-not $RollingMap.shards.PSObject.Properties[$name]) { throw "the rolling map lost the anchor's shard '$name'" }
    }
    $alwaysRolling = [string[]] @($RollingMap.alwaysRun | ForEach-Object { [string] $_ })
    $alwaysAnchor = [string[]] @($AnchorMap.alwaysRun | ForEach-Object { [string] $_ })
    [Array]::Sort($alwaysRolling, $ordinal); [Array]::Sort($alwaysAnchor, $ordinal)
    if (($alwaysRolling -join "`n") -cne ($alwaysAnchor -join "`n")) { throw "the rolling map changed the anchor's always-run list" }
    $limit = [Math]::Min([int] $MaxDigestAgeDays, [int] $Certificate.maxDigestAgeDays)
    foreach ($provenance in $RollingMap.shards.PSObject.Properties) {
        if ([string] $provenance.Value.source -ceq 'anchor') {
            if (-not $anchorKnown.Contains($provenance.Name)) { throw "shard '$($provenance.Name)' claims an anchor measurement the anchor does not have" }
            continue
        }
        $measured = [DateTimeOffset]::MinValue
        if (-not [DateTimeOffset]::TryParse([string] $provenance.Value.measuredUtc, [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::AssumeUniversal, [ref] $measured)) {
            throw "shard '$($provenance.Name)' has no readable measurement time"
        }
        if (($Now - $measured) -gt [TimeSpan]::FromDays($limit)) { throw "the digest of '$($provenance.Name)' is older than $limit day(s)" }
    }
}

# ------------------------------------------------------------------ shadow / on decision

function Get-RollingEffectiveShards {
    <# What a selection actually runs: every shard when it escalates, otherwise the shards it names. #>
    param([Parameter(Mandatory)] $Selection, [Parameter(Mandatory)] [string[]] $AllShards)
    if ([bool] $Selection.escalate) { return @($AllShards) }
    return @($Selection.shards | ForEach-Object { [string] $_ })
}

function Test-RollingSelectionUsable {
    <# '' when a selector result may decide; otherwise why it may not. #>
    param([AllowNull()] $Selection)
    if ($null -eq $Selection) { return 'the rolling selection produced no result' }
    foreach ($name in 'escalate', 'requiresValidation') {
        $property = $Selection.PSObject.Properties[$name]
        if ($null -eq $property -or $property.Value -isnot [bool]) { return "the rolling selection has no boolean '$name'" }
    }
    $shards = $Selection.PSObject.Properties['shards']
    if ($null -eq $shards -or $shards.Value -isnot [array] -or @($shards.Value | Where-Object { $_ -isnot [string] }).Count -gt 0) {
        return "the rolling selection's shards are not a string array"
    }
    $reason = [string] ($Selection.PSObject.Properties['reason']?.Value)
    # The selector's own fail-closed exits. With a rolling map they mean the rolling map could not be used,
    # and the certified weekly map still can.
    if ($reason -cin @('map-missing', 'map-unreadable', 'map-unresolvable', 'selection-failed')) {
        return "the selector could not use the rolling map ($reason)"
    }
    return ''
}

function Get-RollingSelectionDecision {
    <#
    .SYNOPSIS
        Pure. Which selection decides the matrix, and the comparison for the report.
    .DESCRIPTION
        off never reaches here. shadow: the anchor decides, always. on: the rolling map decides only when it
        is certified, its selection is usable, and it agrees with the anchor's about whether the change needs
        validation and shards at all; anything else leaves the anchor's selection exactly as it was. Nothing
        here can produce a matrix smaller than the anchor's except a certified rolling map in on mode.
    #>
    param(
        [Parameter(Mandatory)] [string] $Mode,
        [Parameter(Mandatory)] $AnchorSelection,
        [AllowNull()] $RollingSelection,
        # Why there is no usable rolling map (no artifact, invalid certificate, ...). Empty when there is one.
        [string] $RollingProblem = '',
        [Parameter(Mandatory)] [string[]] $AllShards
    )

    if ((Get-RollingMapMode $Mode) -ceq 'off') { throw 'the rolling map is off; nothing to decide' }
    $problem = $RollingProblem
    if (-not $problem) { $problem = Test-RollingSelectionUsable $RollingSelection }
    $ordinal = [StringComparer]::Ordinal
    $anchorShards = [string[]] @(Get-RollingEffectiveShards -Selection $AnchorSelection -AllShards $AllShards)
    $comparison = [ordered]@{
        anchor = [ordered]@{ escalated = [bool] $AnchorSelection.escalate; shardCount = $anchorShards.Count; shards = @($anchorShards) }
        rolling = $null; droppedByRolling = @(); addedByRolling = @(); inBoth = 0
    }
    if (-not $problem) {
        $rollingShards = [string[]] @(Get-RollingEffectiveShards -Selection $RollingSelection -AllShards $AllShards)
        $anchorSet = [System.Collections.Generic.HashSet[string]]::new($anchorShards, $ordinal)
        $rollingSet = [System.Collections.Generic.HashSet[string]]::new($rollingShards, $ordinal)
        $dropped = [string[]] @($anchorShards | Where-Object { -not $rollingSet.Contains($_) })
        $added = [string[]] @($rollingShards | Where-Object { -not $anchorSet.Contains($_) })
        [Array]::Sort($dropped, $ordinal); [Array]::Sort($added, $ordinal)
        $comparison.rolling = [ordered]@{
            escalated = [bool] $RollingSelection.escalate; shardCount = $rollingShards.Count; shards = @($rollingShards)
            layers = $(if ($RollingSelection.PSObject.Properties['rolling']) { $RollingSelection.rolling } else { $null })
        }
        $comparison.droppedByRolling = @($dropped)
        $comparison.addedByRolling = @($added)
        $comparison.inBoth = @($anchorShards | Where-Object { $rollingSet.Contains($_) }).Count
    }

    $decides = 'anchor'
    $reason = ''
    if ($problem) { $reason = "the certified weekly map decides: $problem" }
    elseif ((Get-RollingMapMode $Mode) -ceq 'shadow') { $reason = 'shadow mode: the certified weekly map decides and the rolling map is only compared' }
    else {
        $disagreement = ''
        foreach ($name in 'requiresValidation', 'requiresShards') {
            $anchorValue = $AnchorSelection.PSObject.Properties[$name]?.Value
            $rollingValue = $RollingSelection.PSObject.Properties[$name]?.Value
            if ("$anchorValue" -cne "$rollingValue") { $disagreement = $name }
        }
        if ($disagreement) { $reason = "the certified weekly map decides: the two selections disagree on $disagreement, which no coverage map should change" }
        else { $decides = 'rolling'; $reason = 'the certified rolling map decides' }
    }
    return [pscustomobject]@{
        mode = (Get-RollingMapMode $Mode); decides = $decides; reason = $reason; rollingProblem = $problem
        comparison = [pscustomobject] $comparison
    }
}

function Get-RollingShadowOutcome {
    <#
        Pure. After the matrix ran: the shards the rolling map would have dropped, and which of them failed.
        In shadow mode they all ran (the anchor decided), so a failure among them is exactly a regression the
        rolling map would have let through. In on mode they did not run and have no outcome.
    #>
    param(
        [Parameter(Mandatory)] $Decision,
        # shard -> job conclusion for this run.
        [Parameter(Mandatory)] [hashtable] $Outcomes
    )
    $dropped = @($Decision.comparison.droppedByRolling | ForEach-Object { [string] $_ })
    $failed = @($dropped | Where-Object { $Outcomes.ContainsKey($_) -and [string] $Outcomes[$_] -ceq 'failure' })
    $notRun = @($dropped | Where-Object { -not $Outcomes.ContainsKey($_) -or [string] $Outcomes[$_] -cin @('skipped', '') })
    return [pscustomobject]@{
        mode = [string] $Decision.mode; decides = [string] $Decision.decides
        dropped = @($dropped); droppedThenFailed = @($failed); droppedNotRun = @($notRun)
    }
}
