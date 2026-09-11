<#
.SYNOPSIS
    Chooses which CI shards a pull request needs, from the coverage-derived shard map.

.DESCRIPTION
    Intersects changed line ranges against the map's file -> shard index. Selection is deliberately
    fail-safe: every changed file and every changed hunk must be accounted for, or the caller is told
    to run the full matrix.

.PARAMETER MapFile
    shard-map.json from New-ShardMap.ps1.

.PARAMETER ExpectedShards
    The complete current shard manifest. A map for a different shard universe is never trusted.

.PARAMETER AuditUnchangedMap
    Allows the nightly selection-miss audit to evaluate an unchanged map tree by selecting only
    always-run shards. Ordinary PR selection must not pass this switch: an unexpectedly empty PR
    diff remains a fail-closed full-matrix decision.

.PARAMETER PullRequestHeadSha
    The pull request's head commit. The checkout must be GitHub's merge of that head onto the base
    branch; its first parent is then the exact base this run validates against, and only the
    paths the merge changes relative to it are this pull request's change. Mutually exclusive
    with BaseSha.

.PARAMETER BaseSha
    An explicit base for the current change. The nightly audit passes the audited commit itself,
    so every map-to-HEAD path is replayed as one change and only selection-control edits are
    treated as historical. Pull requests must use PullRequestHeadSha instead: the event's
    base.sha is stale whenever the pull request is behind its base branch.

.PARAMETER ShardManifestFile
    JSON array of { name, project, filter } for every shard (the converted test-shards.yml).
    Test sources are never in the coverage map, so without this every test-file change escalates;
    with it they are routed to the shards whose filters select their tests.

.PARAMETER SelfTest
    Runs the built-in adversarial checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string[]] $ExpectedShards,
    [Parameter(ParameterSetName = 'Select')] [switch] $AuditUnchangedMap,
    [Parameter(ParameterSetName = 'Select')] [string] $ShardManifestFile,
    [Parameter(ParameterSetName = 'Select')]
    [Parameter(ParameterSetName = 'Classify')] [string] $BaseSha,
    [Parameter(ParameterSetName = 'Select')]
    [Parameter(ParameterSetName = 'Classify')] [string] $PullRequestHeadSha,
    [Parameter(ParameterSetName = 'Select')]
    [Parameter(ParameterSetName = 'Classify')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'Classify')] [switch] $ClassifyOnly,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum ChangedPathImpact {
    NonRuntime
    MapCandidate
    SelectionControl
    FullValidation
}

$script:SharedInfrastructureFiles = @(
    'Directory.Build.props', 'Directory.Build.targets', 'Directory.Packages.props',
    'global.json', 'nuget.config', 'NuGet.config', '.editorconfig'
)
$script:FullValidationPaths = @(
    '.github/test-shards.yml',
    '.github/test-shard-changes.json'
)
$script:SelectionControlPaths = @(
    '.github/workflows/sonarcloud.yml',
    '.github/workflows/test-impact-map.yml',
    '.github/workflows/ci-shard-closure-policy.yml'
)
# Build-time code: the source generators the test project loads as an analyzer. It runs inside the
# compiler, so runtime coverage never records it, yet one edit can rewrite thousands of generated
# test classes across every ModelFamily shard. No coverage-derived routing can bound that.
$script:BuildTimeDirectories = @('src/AiDotNet.Generators/')
$script:FullValidationDirectories = @('.github/actions/', '.github/scripts/') + $script:BuildTimeDirectories
$script:SelectionControlDirectories = @('tools/TestImpact/')
$script:NonRuntimeWorkflowPaths = @(
    '.github/workflows/azure-functions-deploy.yml',
    '.github/workflows/cancel-on-pr-close.yml',
    '.github/workflows/ci-website.yml',
    '.github/workflows/codacy.yml',
    '.github/workflows/commitlint-fix.yml',
    '.github/workflows/commitlint.yml',
    '.github/workflows/copilot-review-gate.yml',
    '.github/workflows/deploy-serving.yml',
    '.github/workflows/deploy-website.yml',
    '.github/workflows/docs-wiki.yml',
    '.github/workflows/docs.yml',
    '.github/workflows/dotnet-format-autofix.yml',
    '.github/workflows/heavy-timeout-nightly.yml',
    '.github/workflows/model-performance-census.yml',
    '.github/workflows/pr-title-lint.yml',
    '.github/workflows/release-please.yml',
    '.github/workflows/samples.yml'
)

function Test-SelectionControl {
    param([string] $Path)
    $normalized = $Path.Replace('\', '/')
    foreach ($entry in $script:SelectionControlPaths) {
        if ($normalized.Equals($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    foreach ($entry in $script:SelectionControlDirectories) {
        if ($normalized.StartsWith($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Test-SharedInfrastructure {
    param([string] $Path)
    $normalized = $Path.Replace('\', '/')
    $name = [System.IO.Path]::GetFileName($normalized)
    foreach ($entry in $script:SharedInfrastructureFiles) {
        # MSBuild and NuGet apply these names per-directory, so a nested file is just as capable of
        # changing compilation as the root one. Exact basename matching keeps lookalike suffixes out.
        if ($name -ieq $entry) { return $true }
    }
    foreach ($entry in $script:FullValidationPaths) {
        if ($normalized.Equals($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    foreach ($entry in $script:FullValidationDirectories) {
        if ($normalized.StartsWith($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Get-ChangedPathImpact {
    param([Parameter(Mandatory)] [string] $Path)

    $normalized = $Path.Replace('\', '/')
    if (Test-SelectionControl -Path $normalized) {
        return [ChangedPathImpact]::SelectionControl
    }
    if (Test-SharedInfrastructure -Path $normalized) {
        return [ChangedPathImpact]::FullValidation
    }

    # Markdown cannot alter a build or runtime. Known independent workflows have their own triggers
    # and jobs; changing one cannot alter this validation workflow. This is an allowlist so a newly
    # added or renamed workflow remains full-validation until its independence is reviewed.
    if ([System.IO.Path]::GetExtension($normalized).Equals('.md', [StringComparison]::OrdinalIgnoreCase)) {
        return [ChangedPathImpact]::NonRuntime
    }
    if ($normalized.StartsWith('.github/workflows/', [StringComparison]::OrdinalIgnoreCase)) {
        foreach ($entry in $script:NonRuntimeWorkflowPaths) {
            if ($normalized.Equals($entry, [StringComparison]::OrdinalIgnoreCase)) {
                return [ChangedPathImpact]::NonRuntime
            }
        }
        return [ChangedPathImpact]::FullValidation
    }

    # Unknown GitHub configuration can affect analysis, generated reports, or a required check. It
    # is intentionally not eligible for coverage-map reduction until classified explicitly.
    if ($normalized.StartsWith('.github/', [StringComparison]::OrdinalIgnoreCase)) {
        return [ChangedPathImpact]::FullValidation
    }

    return [ChangedPathImpact]::MapCandidate
}

function Test-RangeOverlap {
    param([int] $AStart, [int] $AEnd, [int] $BStart, [int] $BEnd)
    return -not ($AEnd -lt $BStart -or $BEnd -lt $AStart)
}

function ConvertTo-ValidatedInteger {
    param(
        [Parameter(Mandatory)] $Value,
        [Parameter(Mandatory)] [string] $What,
        [int] $Minimum = 0,
        [int] $Maximum = [int]::MaxValue
    )

    $isInteger = $Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
    if (-not $isInteger) { throw "$What must be an integer." }

    $number = [long] $Value
    if ($number -lt $Minimum -or $number -gt $Maximum) {
        throw "$What must be between $Minimum and $Maximum."
    }
    return [int] $number
}

function Assert-ShardMap {
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [string[]] $Expected
    )

    foreach ($required in 'schemaVersion', 'sha', 'knownShards', 'alwaysRun', 'files') {
        if (-not $Map.PSObject.Properties[$required]) { throw "no '$required' property" }
    }
    $schemaVersion = ConvertTo-ValidatedInteger -Value $Map.schemaVersion -What 'schemaVersion' -Minimum 1
    if ($schemaVersion -ne 1) { throw "unsupported schemaVersion $schemaVersion" }
    if ([string] $Map.sha -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
        throw 'sha must be a complete hexadecimal Git object id'
    }
    if ($Map.knownShards -isnot [array] -or $Map.alwaysRun -isnot [array]) {
        throw 'knownShards and alwaysRun must be arrays'
    }
    if ($Map.files -isnot [pscustomobject]) { throw 'files must be an object' }

    $expectedSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in @($Expected)) {
        if ([string]::IsNullOrWhiteSpace([string] $name)) { throw 'expected shard names must be nonempty' }
        if (-not $expectedSet.Add([string] $name)) { throw "duplicate expected shard '$name'" }
    }
    if ($expectedSet.Count -eq 0) { throw 'the expected shard manifest is empty' }

    $known = @($Map.knownShards)
    $always = @($Map.alwaysRun)
    if ($known.Count -eq 0) { throw 'knownShards is empty' }

    $mapSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($group in @(@{ Name = 'knownShards'; Values = $known }, @{ Name = 'alwaysRun'; Values = $always })) {
        foreach ($nameValue in $group.Values) {
            $name = [string] $nameValue
            if ([string]::IsNullOrWhiteSpace($name)) { throw "$($group.Name) contains an empty shard name" }
            if (-not $mapSet.Add($name)) { throw "duplicate or overlapping shard '$name'" }
        }
    }

    $missing = @($expectedSet | Where-Object { -not $mapSet.Contains($_) } | Sort-Object)
    $extra = @($mapSet | Where-Object { -not $expectedSet.Contains($_) } | Sort-Object)
    if ($missing.Count -gt 0 -or $extra.Count -gt 0) {
        throw "map shard universe differs from the manifest (missing: $($missing -join ', '); extra: $($extra -join ', '))"
    }

    $fileProperties = @($Map.files.PSObject.Properties)
    if ($fileProperties.Count -eq 0) { throw 'files index is empty' }
    if ($Map.PSObject.Properties['fileCount']) {
        $fileCount = ConvertTo-ValidatedInteger -Value $Map.fileCount -What 'fileCount'
        if ($fileCount -ne $fileProperties.Count) { throw 'fileCount does not match the files index' }
    }
    foreach ($fileProperty in $fileProperties) {
        if ([string]::IsNullOrWhiteSpace([string] $fileProperty.Name)) { throw 'files contains an empty path' }
        if ($fileProperty.Value -isnot [array]) { throw "'$($fileProperty.Name)' occurrences must be an array" }
        $occurrences = @($fileProperty.Value)
        if ($occurrences.Count -eq 0) { throw "'$($fileProperty.Name)' has no occurrences" }
        foreach ($occurrence in $occurrences) {
            if (-not $occurrence.PSObject.Properties['s'] -or -not $occurrence.PSObject.Properties['r']) {
                throw "'$($fileProperty.Name)' has an occurrence without s or r"
            }
            [void] (ConvertTo-ValidatedInteger -Value $occurrence.s -What "'$($fileProperty.Name)' shard index" `
                -Minimum 0 -Maximum ($known.Count - 1))
            if ($occurrence.r -isnot [array]) { throw "'$($fileProperty.Name)' ranges must be an array" }
            $ranges = @($occurrence.r)
            if ($ranges.Count -eq 0 -or $ranges.Count % 2 -ne 0) {
                throw "'$($fileProperty.Name)' ranges must be nonempty start/end pairs"
            }
            for ($i = 0; $i -lt $ranges.Count; $i += 2) {
                $start = ConvertTo-ValidatedInteger -Value $ranges[$i] -What "'$($fileProperty.Name)' range start" -Minimum 1
                $end = ConvertTo-ValidatedInteger -Value $ranges[$i + 1] -What "'$($fileProperty.Name)' range end" -Minimum 1
                if ($start -gt $end) { throw "'$($fileProperty.Name)' range start $start exceeds end $end" }
            }
        }
    }
}

function ConvertTo-ChangedRanges {
    <#
        Parses zero-context hunks in the map commit's coordinates. ChangedFiles is authoritative:
        rename-only, binary and mode-only changes do not necessarily have an @@ header, but they
        must still reach the fail-safe selector.
    #>
    param(
        [AllowEmptyCollection()] [string[]] $DiffLines,
        [AllowEmptyCollection()] [string[]] $ChangedFiles = @()
    )

    $changed = @{}
    $current = $null
    $deletedPath = $null
    # Hunk BODY lines still pending. Headers are only recognised while this is zero, because diff
    # body lines are raw file content behind a one-character prefix, and content can forge any
    # header: a REMOVED line whose text begins with '-- ' is rendered '--- ...', byte-identical to
    # an old-file header. Reproduced with real git: deleting the line '-- remove me' emitted
    # '--- remove me', the old parser took it as a header, nulled $current, and silently dropped
    # every later hunk of that file - under-selection with no escalation. A zero-context hunk
    # '@@ -a,n +b,m @@' is followed by exactly n+m body lines (plus uncounted '\ No newline'
    # markers), so counting them makes body content inert no matter what it says.
    $pendingBody = 0
    foreach ($line in $DiffLines) {
        if ($pendingBody -gt 0) {
            if (-not $line.StartsWith('\')) { $pendingBody-- }
            continue
        }
        if ($line.StartsWith('--- ')) {
            $from = $line.Substring(4).Trim()
            $deletedPath = if ($from -eq '/dev/null') { $null } else { $from -replace '^a/', '' }
            $current = $null
        }
        elseif ($line.StartsWith('+++ ')) {
            $to = $line.Substring(4).Trim()
            $current = if ($to -eq '/dev/null') { $deletedPath } else { $to -replace '^b/', '' }
        }
        elseif ($line.StartsWith('@@') -and $line -match '^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@') {
            $start = [int] $Matches[1]
            $count = if ($Matches[2]) { [int] $Matches[2] } else { 1 }
            $newCount = if ($Matches[4]) { [int] $Matches[4] } else { 1 }
            $pendingBody = $count + $newCount
            if ($current) {
                if (-not $changed.ContainsKey($current)) {
                    $changed[$current] = [System.Collections.Generic.List[int]]::new()
                }
                if ($count -gt 0) {
                    [void] $changed[$current].Add($start)
                    [void] $changed[$current].Add($start + $count - 1)
                }
                else {
                    [void] $changed[$current].Add([Math]::Max(1, $start))
                    [void] $changed[$current].Add([Math]::Max(1, $start + 1))
                }
            }
        }
    }

    foreach ($path in $ChangedFiles) {
        if (-not [string]::IsNullOrWhiteSpace($path) -and -not $changed.ContainsKey($path)) {
            $changed[$path] = [System.Collections.Generic.List[int]]::new()
        }
    }
    return $changed
}

function Get-ChangedRanges {
    param([string] $MapSha)

    $changedFiles = @(& git -c core.quotepath=false diff --no-renames --name-only $MapSha HEAD --)
    if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from '$MapSha' failed" }
    $diff = @(& git -c core.quotepath=false diff --no-renames --no-ext-diff -U0 $MapSha HEAD --)
    if ($LASTEXITCODE -ne 0) { throw "git diff from '$MapSha' failed" }
    return ConvertTo-ChangedRanges -DiffLines $diff -ChangedFiles $changedFiles
}

function Get-DirectoryOwners {
    <#
        The shards that execute any mapped file in the nearest directory of Path that has one, never
        climbing above a two-segment root such as 'src/Finance'. A brand-new source file has no
        coverage of its own, but the code beside it does; climbing to 'src/' itself would stop
        meaning anything, so an orphan with no mapped neighbour below that depth still escalates.
    #>
    param([Parameter(Mandatory)] $Map, [Parameter(Mandatory)] [string] $Path)

    $segments = @(([string] $Path).Replace('\', '/').Split('/'))
    for ($depth = $segments.Count - 1; $depth -ge 2; $depth--) {
        $directory = ($segments[0..($depth - 1)] -join '/') + '/'
        $owners = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
        foreach ($property in $Map.files.PSObject.Properties) {
            if (-not $property.Name.StartsWith($directory, [StringComparison]::OrdinalIgnoreCase)) { continue }
            foreach ($occurrence in @($property.Value)) {
                [void] $owners.Add([string] $Map.knownShards[[int] $occurrence.s])
            }
        }
        if ($owners.Count -gt 0) {
            return [pscustomobject]@{ Directory = $directory.TrimEnd('/'); Shards = @($owners) }
        }
    }
    return $null
}

function Format-LineRanges {
    param([AllowEmptyCollection()] [object[]] $Ranges)
    return (@($Ranges | ForEach-Object { "$($_[0])-$($_[1])" }) -join ', ')
}

function Select-ImpactedShards {
    <#
        CurrentPaths is the change being validated. With -ScopeToCurrentPaths, only those paths are
        selected for: Changed still supplies their line ranges in the MAP's coordinates, but a file
        that differs from the map only because master moved on since the map was built belongs to
        commits that were validated when they landed, not to this pull request.

        Without the switch (the nightly audit, which replays everything since the map as one
        change) CurrentPaths only decides whether a selection-control edit is current.

        TestRoutes carries the precomputed routing for test sources the coverage map cannot index;
        see Get-TestFileRoutes. Every selected shard is recorded in Routes with the reason it was
        selected, so a log reader can see why a shard runs without re-deriving it.
    #>
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [hashtable] $Changed,
        [AllowEmptyCollection()] [string[]] $CurrentPaths = @(),
        [switch] $ScopeToCurrentPaths,
        [hashtable] $TestRoutes = @{},
        [switch] $AuditUnchangedMap
    )

    $selected = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $mappedPaths = [System.Collections.Generic.List[string]]::new()
    $reasons = [System.Collections.Generic.List[string]]::new()
    $routes = [System.Collections.Generic.List[string]]::new()
    $escalate = $false
    $currentPathSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $effectiveCurrentPaths = if ($PSBoundParameters.ContainsKey('CurrentPaths')) {
        @($CurrentPaths)
    }
    else {
        @($Changed.Keys)
    }
    foreach ($currentPath in $effectiveCurrentPaths) {
        if (-not [string]::IsNullOrWhiteSpace([string] $currentPath)) {
            [void] $currentPathSet.Add(([string] $currentPath).Replace('\', '/'))
        }
    }

    if ($ScopeToCurrentPaths) {
        # Scoping discards every path outside the pull request, so an empty pull-request path set
        # would otherwise discard everything and read as a deliberate non-runtime result.
        if ($currentPathSet.Count -eq 0) {
            return [pscustomobject]@{
                Escalate           = $true
                RequiresValidation = $true
                Reasons            = @("the pull request's own changed path set was empty")
                Shards             = @()
                Routes             = @()
            }
        }

        # A pull-request path whose content equals the map's own copy has no map-to-HEAD hunks, yet
        # it still differs from the merge base (master changed it after the map and this pull
        # request reverted that). Its effect cannot be expressed in the map's line numbers.
        $changedPathSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($key in $Changed.Keys) { [void] $changedPathSet.Add(([string] $key).Replace('\', '/')) }
        foreach ($currentPath in ($currentPathSet | Sort-Object)) {
            if ((Get-ChangedPathImpact -Path $currentPath) -eq [ChangedPathImpact]::NonRuntime) { continue }
            if (-not $changedPathSet.Contains($currentPath)) {
                $escalate = $true
                [void] $reasons.Add("changed by this pull request but identical to the map's copy, so its effect has no map line numbers: $currentPath")
            }
        }
    }

    foreach ($path in ($Changed.Keys | Sort-Object)) {
        if ($ScopeToCurrentPaths -and -not $currentPathSet.Contains(([string] $path).Replace('\', '/'))) {
            continue
        }
        $impact = Get-ChangedPathImpact -Path $path
        switch ($impact) {
            ([ChangedPathImpact]::NonRuntime) { continue }
            ([ChangedPathImpact]::SelectionControl) {
                # A control-path edit in THIS pull request must exercise the complete matrix. The
                # same path in the older map-to-HEAD delta was already validated when it landed and
                # cannot change source line ranges; treating it as permanently current otherwise
                # wedges every future runtime PR in full-matrix mode until another map is built.
                if ($currentPathSet.Contains(([string] $path).Replace('\', '/'))) {
                    $escalate = $true
                    [void] $reasons.Add("current validation-selection control change: $path")
                }
                continue
            }
            ([ChangedPathImpact]::FullValidation) {
                $escalate = $true
                [void] $reasons.Add("validation infrastructure or unknown GitHub configuration: $path")
                continue
            }
            ([ChangedPathImpact]::MapCandidate) {
                [void] $mappedPaths.Add([string] $path)
            }
        }
    }

    if ($Changed.Count -eq 0) {
        if ($AuditUnchangedMap) {
            foreach ($shard in @($Map.alwaysRun)) { [void] $selected.Add([string] $shard) }
            # Certificate policy requires both a nonempty selected set and a nonempty skipped set.
            # With no always-run shards, an unchanged audit is vacuous and must wait for a real
            # mapped change rather than certifying that selection was exercised when it was not.
            if ($selected.Count -gt 0) {
                return [pscustomobject]@{
                    Escalate           = $false
                    RequiresValidation = $true
                    Reasons            = @('map and audited tree are unchanged')
                    Shards             = @($selected | Sort-Object)
                    Routes             = @($selected | Sort-Object | ForEach-Object { "$_ <= is always run" })
                }
            }
        }

        return [pscustomobject]@{
            Escalate            = $true
            RequiresValidation = $true
            Reasons             = @('changed path set was empty')
            Shards              = @()
            Routes              = @()
        }
    }

    # A non-runtime-only change is a deliberate empty selection, distinct from a selector failure.
    # Keep that state typed here and expose only a JSON boolean at the workflow boundary.
    if ($mappedPaths.Count -eq 0) {
        return [pscustomobject]@{
            Escalate          = $escalate
            RequiresValidation = $escalate
            Reasons           = $reasons
            Shards            = @()
            Routes            = @()
        }
    }

    foreach ($shard in @($Map.alwaysRun)) {
        [void] $selected.Add([string] $shard)
        [void] $routes.Add("$shard <= is always run")
    }

    foreach ($path in $mappedPaths) {

        $entry = $Map.files.PSObject.Properties[$path]
        if (-not $entry) {
            if ($TestRoutes.ContainsKey($path)) {
                # Test sources are never instrumented, so the map can never index them. Their owning
                # shards come from the manifest's own filters instead; see Get-TestFileRoutes.
                $route = $TestRoutes[$path]
                if ($route.Routable) {
                    foreach ($shard in @($route.Shards)) {
                        [void] $selected.Add([string] $shard)
                        [void] $routes.Add("$shard <= runs tests affected by $path ($($route.Why))")
                    }
                }
                else {
                    $escalate = $true
                    [void] $reasons.Add("$($route.Why): $path")
                }
                continue
            }

            # An unmapped source file - normally one this pull request adds - has no coverage of its
            # own. Only C# is routed to its directory's owners: an unmapped project, props or data
            # file can change how everything builds or loads, which no neighbour's coverage bounds.
            $directoryOwners = $null
            if ($path.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase) -and
                -not $path.StartsWith('tests/', [StringComparison]::OrdinalIgnoreCase)) {
                $directoryOwners = Get-DirectoryOwners -Map $Map -Path $path
            }
            if ($null -eq $directoryOwners) {
                $escalate = $true
                [void] $reasons.Add("not executed by any mapped shard: $path")
                continue
            }
            foreach ($shard in @($directoryOwners.Shards)) {
                [void] $selected.Add([string] $shard)
                [void] $routes.Add("$shard <= executes mapped files in $($directoryOwners.Directory), beside the unmapped source $path")
            }
            continue
        }

        $hunks = @($Changed[$path])
        if ($hunks.Count -eq 0 -or $hunks.Count % 2 -ne 0) {
            $escalate = $true
            [void] $reasons.Add("changed file has no trustworthy line hunks: $path")
            continue
        }

        $covered = [bool[]]::new([int] ($hunks.Count / 2))
        $fileOwners = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
        $hitsByShard = [ordered]@{}
        foreach ($occurrence in @($entry.Value)) {
            $ranges = @($occurrence.r)
            $shardName = [string] $Map.knownShards[[int] $occurrence.s]
            [void] $fileOwners.Add($shardName)
            for ($h = 0; $h + 1 -lt $hunks.Count; $h += 2) {
                $hit = $false
                for ($i = 0; $i + 1 -lt $ranges.Count; $i += 2) {
                    if (Test-RangeOverlap -AStart ([int] $hunks[$h]) -AEnd ([int] $hunks[$h + 1]) `
                                          -BStart ([int] $ranges[$i]) -BEnd ([int] $ranges[$i + 1])) {
                        $hit = $true
                        break
                    }
                }
                if ($hit) {
                    $covered[[int] ($h / 2)] = $true
                    [void] $selected.Add($shardName)
                    if (-not $hitsByShard.Contains($shardName)) {
                        $hitsByShard[$shardName] = [System.Collections.Generic.List[object]]::new()
                    }
                    [void] $hitsByShard[$shardName].Add(@($hunks[$h], $hunks[$h + 1]))
                }
            }
        }
        foreach ($shardName in $hitsByShard.Keys) {
            [void] $routes.Add("$shardName <= executes changed lines $(Format-LineRanges $hitsByShard[$shardName]) of $path")
        }

        # A changed range no shard executes is routed to every shard that executes ANY line of the
        # same file. Most such ranges are not dead code: they are declarations coverage never
        # records (fields, attributes, braces between methods) or the insertion point of new code,
        # whose effect reaches tests only through the file's executed members. This is the
        # heuristic the nightly miss audit exists to check.
        $uncovered = [System.Collections.Generic.List[object]]::new()
        for ($h = 0; $h + 1 -lt $hunks.Count; $h += 2) {
            if (-not $covered[[int] ($h / 2)]) { [void] $uncovered.Add(@($hunks[$h], $hunks[$h + 1])) }
        }
        if ($uncovered.Count -gt 0) {
            foreach ($shardName in $fileOwners) {
                [void] $selected.Add($shardName)
                [void] $routes.Add("$shardName <= executes $path, whose changed lines $(Format-LineRanges $uncovered) no shard executes")
            }
        }
    }

    if ($selected.Count -eq 0) {
        $escalate = $true
        [void] $reasons.Add('selection was empty')
    }

    return [pscustomobject]@{
        Escalate           = $escalate
        RequiresValidation = $true
        Reasons            = $reasons
        Shards             = @($selected | Sort-Object)
        Routes             = @($routes)
    }
}

# ------------------------------------------------------------- test routing

function ConvertTo-TestFilter {
    <#
        Parses a VSTest filter into an expression tree. VSTest's grammar: conditions
        Property(=|!=|~|!~)Value joined by '&' and '|', '&' binding tighter, with parentheses.
        Whitespace around tokens is insignificant (folded YAML inserts it at line breaks).
        Anything outside that grammar throws, and the caller then routes nothing - it escalates.
    #>
    param([Parameter(Mandatory)] [string] $Text)

    if ($Text.Contains('\')) { throw 'escaped filter characters are not supported' }
    $tokens = [System.Collections.Generic.List[string]]::new()
    foreach ($match in [regex]::Matches($Text, '[()&|]|[^()&|]+')) {
        $value = $match.Value.Trim()
        if ($value.Length -gt 0) { [void] $tokens.Add($value) }
    }

    function Read-Or {
        $items = [System.Collections.Generic.List[object]]::new()
        [void] $items.Add((Read-And))
        while ($script:filterPosition -lt $tokens.Count -and $tokens[$script:filterPosition] -eq '|') {
            $script:filterPosition++
            [void] $items.Add((Read-And))
        }
        if ($items.Count -eq 1) { return $items[0] }
        return [pscustomobject]@{ Kind = 'Or'; Items = @($items) }
    }
    function Read-And {
        $items = [System.Collections.Generic.List[object]]::new()
        [void] $items.Add((Read-Factor))
        while ($script:filterPosition -lt $tokens.Count -and $tokens[$script:filterPosition] -eq '&') {
            $script:filterPosition++
            [void] $items.Add((Read-Factor))
        }
        if ($items.Count -eq 1) { return $items[0] }
        return [pscustomobject]@{ Kind = 'And'; Items = @($items) }
    }
    function Read-Factor {
        if ($script:filterPosition -ge $tokens.Count) { throw 'filter ended where a condition was expected' }
        $token = $tokens[$script:filterPosition]
        if ($token -eq '(') {
            $script:filterPosition++
            $inner = Read-Or
            if ($script:filterPosition -ge $tokens.Count -or $tokens[$script:filterPosition] -ne ')') {
                throw 'unbalanced parenthesis in filter'
            }
            $script:filterPosition++
            return $inner
        }
        if ($token -in @(')', '&', '|')) { throw "unexpected '$token' in filter" }
        if ($token -notmatch '^(?<p>[A-Za-z][\w.]*)\s*(?<op>!=|!~|=|~)\s*(?<v>\S(?:.*\S)?)$') {
            throw "unparseable filter condition '$token'"
        }
        $script:filterPosition++
        return [pscustomobject]@{ Kind = 'Condition'; Property = $Matches.p; Operator = $Matches.op; Value = $Matches.v }
    }

    $script:filterPosition = 0
    $tree = Read-Or
    if ($script:filterPosition -ne $tokens.Count) { throw "unexpected '$($tokens[$script:filterPosition])' in filter" }
    return $tree
}

# Three-valued results. Unknown means "this test might match", so a shard is skipped only when its
# filter is definitely false for every test the changed file can contain.
$script:FilterFalse = 0
$script:FilterTrue = 1
$script:FilterUnknown = 2

function Invert-FilterResult {
    param([int] $Value)
    if ($Value -eq $script:FilterUnknown) { return $script:FilterUnknown }
    return 1 - $Value
}

function Test-OpenSuffixCanComplete {
    <#
        Whether Prefix followed by some unknown suffix can contain (or, with -Exact, equal) Value.
        AnySuffix admits any text; otherwise the suffix is one method identifier, which contains no
        '.' or '+', so a namespace or class term that is not already in the prefix cannot appear.
    #>
    param([string] $Prefix, [string] $Value, [bool] $AnySuffix, [switch] $Exact)

    for ($k = 0; $k -le $Value.Length; $k++) {
        $head = $Value.Substring(0, $k)
        $tail = $Value.Substring($k)
        $headFits = if ($Exact) { $Prefix.Equals($head, [StringComparison]::OrdinalIgnoreCase) }
                    else { $Prefix.EndsWith($head, [StringComparison]::OrdinalIgnoreCase) }
        if (-not $headFits) { continue }
        if ($AnySuffix -or $tail -match '^\w*$') { return $true }
    }
    return $false
}

function Test-FilterStringCondition {
    <#
        Actual is either a test's complete fully qualified name, or - when PrefixOnly - the known
        start of names whose remainder cannot be read from this file (inherited tests, or a test
        whose declaration could not be read). Each comparison is made both ordinally and ignoring
        case; if the two disagree the result is Unknown, so neither VSTest casing behaviour can
        make a matching test look excluded.
    #>
    param([string] $Actual, [bool] $PrefixOnly, [bool] $AnySuffix, [string] $Operator, [string] $Value)

    $negated = $Operator.StartsWith('!')
    if ($Operator.EndsWith('~')) {
        $ordinal = $Actual.Contains($Value, [StringComparison]::Ordinal)
        $ignoreCase = $Actual.Contains($Value, [StringComparison]::OrdinalIgnoreCase)
        $result = if ($ordinal) { $script:FilterTrue }
                  elseif ($ignoreCase) { $script:FilterUnknown }
                  elseif ($PrefixOnly -and (Test-OpenSuffixCanComplete -Prefix $Actual -Value $Value -AnySuffix $AnySuffix)) {
                      $script:FilterUnknown
                  }
                  else { $script:FilterFalse }
    }
    else {
        if ($PrefixOnly) {
            $result = if (Test-OpenSuffixCanComplete -Prefix $Actual -Value $Value -AnySuffix $AnySuffix -Exact) {
                          $script:FilterUnknown
                      }
                      else { $script:FilterFalse }
        }
        else {
            $ordinal = $Actual.Equals($Value, [StringComparison]::Ordinal)
            $ignoreCase = $Actual.Equals($Value, [StringComparison]::OrdinalIgnoreCase)
            $result = if ($ordinal) { $script:FilterTrue }
                      elseif ($ignoreCase) { $script:FilterUnknown }
                      else { $script:FilterFalse }
        }
    }
    if ($negated) { return Invert-FilterResult $result }
    return $result
}

function Test-TestFilter {
    param([Parameter(Mandatory)] $Node, [Parameter(Mandatory)] $Candidate)

    switch ($Node.Kind) {
        'And' {
            $sawUnknown = $false
            foreach ($item in $Node.Items) {
                $value = Test-TestFilter -Node $item -Candidate $Candidate
                if ($value -eq $script:FilterFalse) { return $script:FilterFalse }
                if ($value -eq $script:FilterUnknown) { $sawUnknown = $true }
            }
            if ($sawUnknown) { return $script:FilterUnknown }
            return $script:FilterTrue
        }
        'Or' {
            $sawUnknown = $false
            foreach ($item in $Node.Items) {
                $value = Test-TestFilter -Node $item -Candidate $Candidate
                if ($value -eq $script:FilterTrue) { return $script:FilterTrue }
                if ($value -eq $script:FilterUnknown) { $sawUnknown = $true }
            }
            if ($sawUnknown) { return $script:FilterUnknown }
            return $script:FilterFalse
        }
        'Condition' {
            if ($Node.Property -ieq 'FullyQualifiedName') {
                return Test-FilterStringCondition -Actual $Candidate.Fqn -PrefixOnly $Candidate.PrefixOnly -AnySuffix $Candidate.AnySuffix `
                    -Operator $Node.Operator -Value $Node.Value
            }
            if ($Node.Property -ieq 'Category') {
                # Categories are known only as the union over the whole file, so a category that
                # appears somewhere in the file may or may not be on this particular test.
                if ($null -eq $Candidate.Categories) { return $script:FilterUnknown }
                $present = $false
                foreach ($category in $Candidate.Categories) {
                    $hit = if ($Node.Operator.EndsWith('~')) {
                        $category.Contains($Node.Value, [StringComparison]::OrdinalIgnoreCase)
                    } else {
                        $category.Equals($Node.Value, [StringComparison]::OrdinalIgnoreCase)
                    }
                    if ($hit) { $present = $true; break }
                }
                if ($present) { return $script:FilterUnknown }
                if ($Node.Operator.StartsWith('!')) { return $script:FilterTrue }
                return $script:FilterFalse
            }
            # Any other property is not modelled, so it can never be what excludes a test.
            return $script:FilterUnknown
        }
        default { throw "unknown filter node '$($Node.Kind)'" }
    }
}

function ConvertTo-CodeOnlyCSharp {
    <#
        Blanks the CONTENT of comments, strings and character literals (newlines are kept, so
        offsets and line numbers survive), leaving only code. Brace matching and declaration
        matching then cannot be fooled by '{' in a string or 'class X' in a comment.
    #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Text)

    $pattern = '(?s)//[^\n]*|/\*.*?\*/|\$*(?<q>"{3,}).*?\k<q>|(?:\$@|@\$|@)"(?:[^"]|"")*"|\$?"(?:[^"\\\n]|\\.)*"|''(?:[^''\\\n]|\\.){1,10}'''
    $builder = [System.Text.StringBuilder]::new($Text.Length)
    $last = 0
    foreach ($match in [regex]::Matches($Text, $pattern)) {
        [void] $builder.Append($Text, $last, $match.Index - $last)
        [void] $builder.Append(($match.Value -replace '[^\r\n]', ' '))
        $last = $match.Index + $match.Length
    }
    [void] $builder.Append($Text, $last, $Text.Length - $last)
    return $builder.ToString()
}

function Get-CSharpTestShape {
    <#
        Reads, from one C# test source, what VSTest filters can see: every test's fully qualified
        name, the Category traits the file can carry, and the top-level types other files could
        reference. Returns ParseError instead of guessing when the braces do not balance.
    #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Text)

    $code = ConvertTo-CodeOnlyCSharp -Text $Text
    $shape = [pscustomobject]@{
        Types          = [System.Collections.Generic.List[object]]::new()
        Tests          = [System.Collections.Generic.List[object]]::new()
        Categories     = $null
        Hazard         = $null
        ParseError     = $null
    }

    # Brace pairs by position.
    $open = [System.Collections.Generic.Stack[int]]::new()
    $closeOf = @{}
    foreach ($brace in [regex]::Matches($code, '[{}]')) {
        if ($brace.Value -eq '{') { $open.Push($brace.Index); continue }
        if ($open.Count -eq 0) { $shape.ParseError = 'unbalanced braces'; return $shape }
        $closeOf[$open.Pop()] = $brace.Index
    }
    if ($open.Count -ne 0) { $shape.ParseError = 'unbalanced braces'; return $shape }

    function Get-BodyRange([int] $From) {
        # The first '{' after a declaration opens its body, unless a ';' ends the declaration first.
        for ($k = $From; $k -lt $code.Length; $k++) {
            if ($code[$k] -eq ';') { return $null }
            if ($code[$k] -eq '{') { return @($k, [int] $closeOf[$k]) }
        }
        return $null
    }

    $namespaces = [System.Collections.Generic.List[object]]::new()
    $fileNamespace = ''
    foreach ($match in [regex]::Matches($code, '(?m)^[ \t]*namespace\s+(?<n>[A-Za-z_][\w.]*)\s*(?<t>[;{])')) {
        if ($match.Groups['t'].Value -eq ';') { $fileNamespace = $match.Groups['n'].Value; continue }
        $start = $match.Groups['t'].Index
        [void] $namespaces.Add([pscustomobject]@{ Name = $match.Groups['n'].Value; Start = $start; End = [int] $closeOf[$start] })
    }

    $typePattern = '\b(?<kind>class|struct|interface|enum|record(?:\s+(?:class|struct))?)\s+(?<name>[A-Za-z_]\w*)'
    foreach ($match in [regex]::Matches($code, $typePattern)) {
        # 'where T : class where U : new()' puts a keyword where a type name would be.
        if ($match.Groups['name'].Value -cin @('where', 'new', 'class', 'struct', 'unmanaged', 'notnull')) { continue }
        $body = Get-BodyRange ($match.Index + $match.Length)
        $header = if ($null -ne $body) { $code.Substring($match.Index + $match.Length, $body[0] - $match.Index - $match.Length) } else { '' }
        $lineStart = $code.LastIndexOf("`n", [Math]::Max(0, $match.Index - 1)) + 1
        $modifiers = $code.Substring($lineStart, $match.Index - $lineStart)

        # The base list follows the name, after any type parameters and primary constructor, and
        # before any constraint clause. Interfaces are recognised by the I-prefix convention; any
        # other base is a class whose tests and traits this type inherits from elsewhere.
        $baseText = [regex]::Replace($header, '<[^<>]*>', '')
        while ($baseText -match '\([^()]*\)') { $baseText = [regex]::Replace($baseText, '\([^()]*\)', '') }
        $baseText = ($baseText -split '\bwhere\b')[0]
        $hasClassBase = $false
        $colon = $baseText.IndexOf(':')
        if ($colon -ge 0) {
            foreach ($base in $baseText.Substring($colon + 1).Split(',')) {
                $baseName = ($base.Trim() -split '\.')[-1]
                if ($baseName -and $baseName -cnotmatch '^I[A-Z]') { $hasClassBase = $true }
            }
        }

        [void] $shape.Types.Add([pscustomobject]@{
            Name         = $match.Groups['name'].Value
            Kind         = ($match.Groups['kind'].Value -split '\s+')[0]
            Start        = $match.Index
            BodyStart    = if ($null -ne $body) { $body[0] } else { -1 }
            BodyEnd      = if ($null -ne $body) { $body[1] } else { -1 }
            FileLocal    = $modifiers -match '\bfile\b'
            IsAbstract   = $modifiers -match '\babstract\b'
            HasClassBase = $hasClassBase
            Parent       = $null
            Namespace    = $fileNamespace
        })
    }

    # Nesting and namespaces by containment.
    foreach ($type in $shape.Types) {
        $innermost = $null
        foreach ($candidate in $shape.Types) {
            if ($candidate -eq $type -or $candidate.BodyStart -lt 0) { continue }
            if ($type.Start -gt $candidate.BodyStart -and $type.Start -lt $candidate.BodyEnd) {
                if ($null -eq $innermost -or $candidate.BodyStart -gt $innermost.BodyStart) { $innermost = $candidate }
            }
        }
        $type.Parent = $innermost
        foreach ($namespace in $namespaces) {
            if ($type.Start -gt $namespace.Start -and $type.Start -lt $namespace.End) { $type.Namespace = $namespace.Name }
        }
    }

    function Get-TypeChain($Type) {
        $names = [System.Collections.Generic.List[string]]::new()
        for ($t = $Type; $null -ne $t; $t = $t.Parent) { $names.Insert(0, $t.Name) }
        return ($names -join '+')
    }
    function Get-TypePrefix($Type) {
        $chain = Get-TypeChain $Type
        if ($Type.Namespace) { return "$($Type.Namespace).$chain" }
        return $chain
    }

    # Test methods: an attribute list naming a *Fact or *Theory attribute, then the method whose
    # parameter list is the next '(' - its name is the last identifier before it. An attribute
    # list only starts a declaration, so it must follow a line start, '{', '}', ';' or ']'.
    foreach ($attribute in [regex]::Matches($code, '\[(?<body>[^\[\]]*)\]')) {
        $isTest = $false
        foreach ($part in $attribute.Groups['body'].Value.Split(',')) {
            if ($part.Trim() -match '^(?:[\w.]+\.)?\w*(?:Fact|Theory)(?:Attribute)?\s*(?:\(|$)') { $isTest = $true }
        }
        if (-not $isTest) { continue }
        $before = $code.Substring(0, $attribute.Index).TrimEnd(' ', "`t")
        if ($before.Length -gt 0 -and $before[-1] -notin @("`n", "`r", '{', '}', ';', ']')) { continue }

        $after = $attribute.Index + $attribute.Length
        $owner = $null
        foreach ($type in $shape.Types) {
            if ($type.BodyStart -lt 0) { continue }
            if ($after -gt $type.BodyStart -and $after -lt $type.BodyEnd) {
                if ($null -eq $owner -or $type.BodyStart -gt $owner.BodyStart) { $owner = $type }
            }
        }

        # Skip any further attribute lists (bracket depth counted, so '[InlineData(new[] { 1 })]'
        # is one list), then read the name before the parameter list.
        $position = $after
        while ($true) {
            while ($position -lt $code.Length -and [char]::IsWhiteSpace($code[$position])) { $position++ }
            if ($position -ge $code.Length -or $code[$position] -ne '[') { break }
            $depth = 0
            do {
                if ($code[$position] -eq '[') { $depth++ } elseif ($code[$position] -eq ']') { $depth-- }
                $position++
            } while ($depth -gt 0 -and $position -lt $code.Length)
        }
        $method = $null
        $paren = $code.IndexOf('(', $position)
        if ($paren -ge 0) {
            $signature = [regex]::Replace($code.Substring($position, $paren - $position), '<[^<>]*>\s*$', '')
            if ($signature -notmatch '[;{}=\[\]]' -and $signature -match '(?<m>[A-Za-z_]\w*)\s*$') { $method = $Matches.m }
        }

        # A test whose name cannot be read still exists. It is kept as an open-ended candidate so a
        # filter naming its method can never be what makes its shard look unaffected.
        if ($null -eq $owner) {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = $(if ($fileNamespace) { "$fileNamespace." } else { '' }); PrefixOnly = $true; AnySuffix = $true })
        }
        elseif ($null -eq $method) {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = "$(Get-TypePrefix $owner)."; PrefixOnly = $true; AnySuffix = $false })
        }
        else {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = "$(Get-TypePrefix $owner).$method"; PrefixOnly = $false; AnySuffix = $false })
        }
    }

    # A class with a class base can inherit tests whose method names live in another file.
    foreach ($type in $shape.Types) {
        if ($type.Kind -in @('class', 'record') -and $type.HasClassBase) {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = "$(Get-TypePrefix $type)."; PrefixOnly = $true; AnySuffix = $false })
        }
    }

    # Categories are knowable only when every Category trait is a literal and no class inherits
    # traits from a base declared elsewhere.
    $categories = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $categoriesKnown = -not @($shape.Types | Where-Object { $_.HasClassBase }).Count
    foreach ($trait in [regex]::Matches($Text, '\bTrait(?:Attribute)?\s*\(\s*"(?i:category)"\s*,\s*(?<v>[^)]*)\)')) {
        $value = $trait.Groups['v'].Value.Trim()
        if ($value -match '^"(?<c>[^"]*)"$') { [void] $categories.Add($Matches.c) }
        else { $categoriesKnown = $false }
    }
    if ($categoriesKnown) { $shape.Categories = @($categories) }

    # Constructs that reach tests without naming this file's types, which reference search cannot
    # follow: extension methods, global usings, assembly-level attributes, module initializers, and
    # xUnit collection definitions (bound by string name).
    $hazards = [ordered]@{
        'declares extension methods'           = '\(\s*this\s+[A-Za-z_]'
        'declares global usings'               = '(?m)^\s*global\s+using\b'
        'declares assembly-level attributes'   = '\[\s*assembly\s*:'
        'declares a module initializer'        = '\bModuleInitializer\b'
        'declares an xUnit collection definition' = '\bCollectionDefinition\b'
    }
    foreach ($hazard in $hazards.Keys) {
        if ($code -match $hazards[$hazard]) { $shape.Hazard = $hazard; break }
    }
    # An abstract class is a base by construction, and the scaffold generator derives test classes
    # from bases whose names it can compose at build time. Those derived classes are not in the
    # tree, so reference search cannot find them.
    if (-not $shape.Hazard -and @($shape.Types | Where-Object { $_.IsAbstract }).Count -gt 0) {
        $shape.Hazard = 'declares an abstract class, which build-time generated test classes may derive from'
    }
    return $shape
}

function Get-ShardProjectDirectory {
    param([Parameter(Mandatory)] $Shard)
    $project = ([string] $Shard.project).Replace('\', '/')
    $slash = $project.LastIndexOf('/')
    if ($slash -lt 0) { return '' }
    return $project.Substring(0, $slash + 1)
}

function Get-TestProjectDirectory {
    param([Parameter(Mandatory)] [string] $Path, [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Manifest)

    $best = $null
    foreach ($shard in $Manifest) {
        $directory = Get-ShardProjectDirectory -Shard $shard
        if ($directory -and $Path.StartsWith($directory, [StringComparison]::OrdinalIgnoreCase) -and
            ($null -eq $best -or $directory.Length -gt $best.Length)) {
            $best = $directory
        }
    }
    return $best
}

function Get-TestFileRoutes {
    <#
        Routes changed test sources to the shards whose manifest filters select their tests.

        A test file affects the tests it declares and, through its types, every test file that uses
        them: a base class, fixture or helper changes the behaviour of its consumers. So routing
        follows the transitive closure of files that name this file's top-level types, and unions
        the shards that select any test in that closure. Constructs whose consumers cannot be found
        by name (see Get-CSharpTestShape hazards), a closure too wide to be selective, a file whose
        text cannot be parsed, and tests no shard filter selects all return Routable = $false,
        which the selector turns into an escalation.

        FindBuildTimeReferences reports which of a file's type names build-time code (the source
        generators) mentions: tests the generator emits exist only inside the compiler, so a type
        they depend on cannot be routed by reading the tree.

        ReadFile, FindReferrers and FindBuildTimeReferences are injected so the self-test can
        exercise routing without a repository; in production they read HEAD and use git grep.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Paths,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Manifest,
        [Parameter(Mandatory)] [scriptblock] $ReadFile,
        [Parameter(Mandatory)] [scriptblock] $FindReferrers,
        [Parameter(Mandatory)] [scriptblock] $FindBuildTimeReferences,
        [int] $MaximumClosure = 200
    )

    $filters = @{}
    foreach ($shard in $Manifest) {
        try { $filters[[string] $shard.name] = ConvertTo-TestFilter -Text ([string] $shard.filter) }
        catch { $filters[[string] $shard.name] = $null }
    }

    $routes = @{}
    foreach ($path in $Paths) {
        $normalized = $path.Replace('\', '/')
        $projectDirectory = Get-TestProjectDirectory -Path $normalized -Manifest $Manifest
        if ($null -eq $projectDirectory) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'test source belongs to no project any shard runs' }
            continue
        }
        $projectShards = @($Manifest | Where-Object { (Get-ShardProjectDirectory -Shard $_) -ieq $projectDirectory })

        $visited = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        $queue = [System.Collections.Generic.Queue[string]]::new()
        [void] $visited.Add($normalized)
        $queue.Enqueue($normalized)
        $candidates = [System.Collections.Generic.List[object]]::new()
        $failure = $null
        $ownTests = 0
        $ownReferrers = 0

        while ($queue.Count -gt 0 -and $null -eq $failure) {
            $file = $queue.Dequeue()
            $text = & $ReadFile $file
            if ($null -eq $text) { $failure = "cannot read test source $file"; break }
            $shape = Get-CSharpTestShape -Text $text
            if ($shape.ParseError) { $failure = "cannot parse test source $file ($($shape.ParseError))"; break }
            if ($shape.Hazard) { $failure = "test source $file $($shape.Hazard), whose consumers cannot be found by name"; break }

            foreach ($test in $shape.Tests) {
                [void] $candidates.Add([pscustomobject]@{ Fqn = $test.Fqn; PrefixOnly = $test.PrefixOnly; AnySuffix = $test.AnySuffix; Categories = $shape.Categories })
            }
            if ($file -ieq $normalized) { $ownTests = $shape.Tests.Count }

            $names = @($shape.Types | Where-Object { $null -eq $_.Parent -and -not $_.FileLocal } |
                ForEach-Object { $_.Name } | Sort-Object -Unique)
            if ($names.Count -eq 0) { continue }
            $generated = @(& $FindBuildTimeReferences $names)
            if ($generated.Count -gt 0) {
                $failure = "test source $file declares $($generated -join ', '), which a source generator references, so tests generated at build time may depend on it"
                break
            }
            foreach ($referrer in @(& $FindReferrers $names $projectDirectory)) {
                $referrerPath = ([string] $referrer).Replace('\', '/')
                if ($referrerPath -ieq $file) { continue }
                if ($file -ieq $normalized) { $ownReferrers++ }
                if ($visited.Add($referrerPath)) {
                    if ($visited.Count -gt $MaximumClosure) {
                        $failure = "test source is shared by more than $MaximumClosure test files"
                        break
                    }
                    $queue.Enqueue($referrerPath)
                }
            }
        }

        if ($null -ne $failure) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = $failure }
            continue
        }
        if ($ownTests -eq 0 -and $ownReferrers -eq 0) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'test support source declares no tests and no test file names its types' }
            continue
        }

        $shards = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
        $unparsed = $false
        foreach ($shard in $projectShards) {
            $filter = $filters[[string] $shard.name]
            if ($null -eq $filter) { $unparsed = $true; break }
            foreach ($candidate in $candidates) {
                if ((Test-TestFilter -Node $filter -Candidate $candidate) -ne $script:FilterFalse) {
                    [void] $shards.Add([string] $shard.name)
                    break
                }
            }
        }
        if ($unparsed) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'a shard filter for this test project cannot be parsed' }
        }
        elseif ($shards.Count -eq 0) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'no shard filter selects any test this source affects' }
        }
        else {
            $closure = $visited.Count - 1
            $why = if ($closure -gt 0) { "its filter selects tests in this file or in $closure test file(s) that use it" }
                   else { 'its filter selects tests in this file' }
            $routes[$path] = [pscustomobject]@{ Routable = $true; Shards = @($shards); Why = $why }
        }
    }
    return $routes
}

function Get-GitFileText {
    param([Parameter(Mandatory)] [string] $Path, [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Revisions)
    foreach ($revision in $Revisions) {
        if ([string]::IsNullOrWhiteSpace($revision)) { continue }
        $lines = @(& git -c core.quotepath=false show "${revision}:$Path" 2>$null)
        if ($LASTEXITCODE -eq 0) { return ($lines -join "`n") }
    }
    return $null
}

function Find-GitReferrers {
    param([Parameter(Mandatory)] [string[]] $Names, [Parameter(Mandatory)] [string] $Directory)
    $arguments = @('-c', 'core.quotepath=false', 'grep', '-l', '-w', '-F')
    foreach ($name in $Names) { $arguments += @('-e', $name) }
    $arguments += @('HEAD', '--', ":(glob)$Directory**/*.cs")
    $output = @(& git @arguments 2>$null)
    # git grep exits 1 when nothing matches, which is an answer, not a failure.
    if ($LASTEXITCODE -gt 1) { throw "git grep for test-type references failed with exit code $LASTEXITCODE" }
    return @($output | ForEach-Object { ([string] $_) -replace '^HEAD:', '' })
}

function Find-GitBuildTimeReferences {
    <#
        Which of Names build-time code mentions. One search per name keeps the answer exact; the
        lists are short (a file's top-level types) and git grep over the generator tree is fast.
    #>
    param([Parameter(Mandatory)] [string[]] $Names)
    # An empty pathspec would search the whole repository and implicate every type there is.
    $directories = @($script:BuildTimeDirectories | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($directories.Count -eq 0) { throw 'no build-time directories are configured' }
    $found = [System.Collections.Generic.List[string]]::new()
    foreach ($name in $Names) {
        $arguments = @('-c', 'core.quotepath=false', 'grep', '-q', '-w', '-F', '-e', $name, 'HEAD', '--')
        $arguments += $directories
        & git @arguments 2>$null
        if ($LASTEXITCODE -eq 0) { [void] $found.Add($name) }
        elseif ($LASTEXITCODE -gt 1) { throw "git grep for build-time references failed with exit code $LASTEXITCODE" }
    }
    return @($found)
}

function Resolve-PullRequestBase {
    <#
        The pull_request event's base.sha is the base branch as it was when the pull request was
        opened or last retargeted - not what GitHub merged it onto for this run. For a pull request
        behind master it attributes every commit master has gained since to the pull request.

        The checked-out merge ref is authoritative: its first parent IS the base this run tests
        against, and its second parent is the pull request head. Both are verified, so a checkout
        that is not the expected two-parent merge fails closed rather than guessing.
    #>
    param([Parameter(Mandatory)] [string] $PullRequestHeadSha)

    if ($PullRequestHeadSha -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
        throw "pull request head '$PullRequestHeadSha' is not a complete Git object id"
    }
    $parents = @((& git rev-list --parents -n 1 HEAD) -split '\s+' | Where-Object { $_ })
    if ($LASTEXITCODE -ne 0) { throw 'cannot read the checked-out commit' }
    if ($parents.Count -ne 3) {
        throw "the checkout is not a two-parent pull-request merge commit ($($parents.Count - 1) parent(s))"
    }
    if (-not $parents[2].Equals($PullRequestHeadSha, [StringComparison]::OrdinalIgnoreCase)) {
        throw "the merge commit's second parent $($parents[2]) is not the pull request head $PullRequestHeadSha"
    }
    return $parents[1]
}

# ---------------------------------------------------------------- self-test

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True {
        param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) }
    }
    function Assert-Throws {
        param([scriptblock] $Action, [string] $What)
        $threw = $false
        try { & $Action } catch { $threw = $true }
        Assert-True $threw $What
    }

    $map = @{
        schemaVersion = 1
        sha           = '0123456789abcdef0123456789abcdef01234567'
        knownShards   = @('Alpha', 'Beta')
        alwaysRun     = @('HeavyNoCoverage')
        files         = @{
            'src/Covered.cs' = @(
                @{ s = 0; r = @(10, 20) },
                @{ s = 1; r = @(100, 110) }
            )
            'src/SingleRange.cs' = @( @{ s = 0; r = @(5, 6) } )
        }
    } | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json
    $expected = @('Alpha', 'Beta', 'HeavyNoCoverage')
    try { Assert-ShardMap -Map $map -Expected $expected } catch { [void] $failures.Add("valid map rejected: $_") }

    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14) }
    Assert-True (-not $r.Escalate) 'a mapped, fully covered change must not escalate'
    Assert-True $r.RequiresValidation 'a mapped change must require validation'
    Assert-True ($r.Shards -contains 'Alpha') 'a change on Alpha lines must select Alpha'
    Assert-True (-not ($r.Shards -contains 'Beta')) 'a change outside Beta lines must not select Beta'
    Assert-True ($r.Shards -contains 'HeavyNoCoverage') 'always-run shards must always be selected'

    # An unexecuted range in a mapped file routes to every shard that executes the file. Before this
    # it escalated, and #2100's field declarations and between-method insertions - lines coverage
    # never records - sent a four-file pull request to all 116 shards.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14, 50, 60) }
    Assert-True (-not $r.Escalate) 'an unexecuted range in a mapped file escalated instead of routing to its owners'
    Assert-True ($r.Shards -contains 'Alpha' -and $r.Shards -contains 'Beta') `
        'an unexecuted range was not routed to every shard executing the file'
    Assert-True (@($r.Routes | Where-Object { $_ -like 'Beta <= executes src/Covered.cs, whose changed lines 50-60*' }).Count -eq 1) `
        'the owner route did not record which lines no shard executes'
    Assert-True (@($r.Routes | Where-Object { $_ -like 'Alpha <= executes changed lines 12-14 of src/Covered.cs' }).Count -eq 1) `
        'a covered hit did not record the lines that selected it'

    foreach ($change in @(
        @{ Path = 'src/Unmapped.cs'; Why = 'an unmapped file must escalate' },
        @{ Path = 'Directory.Packages.props'; Why = 'dependency infrastructure must escalate' },
        @{ Path = '.github/workflows/sonarcloud.yml'; Why = 'the validation workflow must escalate' },
        @{ Path = '.github/workflows/test-impact-map.yml'; Why = 'the map workflow must escalate' },
        @{ Path = '.github/workflows/ci-shard-closure-policy.yml'; Why = 'the closure policy must escalate' },
        @{ Path = '.github/test-shards.yml'; Why = 'the shard manifest must escalate' },
        @{ Path = '.github/test-shard-changes.json'; Why = 'the shard history must escalate' },
        @{ Path = '.github/scripts/analyze-test-results.ps1'; Why = 'CI analysis scripts must escalate' },
        @{ Path = '.github/actions/local/action.yml'; Why = 'local actions must escalate' },
        @{ Path = 'tools/TestImpact/Select-Shards.ps1'; Why = 'impact tooling must escalate' },
        @{ Path = 'src/AiDotNet.Generators/TestScaffoldGenerator.cs'; Why = 'build-time source generators must escalate' },
        @{ Path = '.github/dependabot.yml'; Why = 'unknown GitHub configuration must escalate' },
        @{ Path = '.github/workflows/release-please.yml.backup'; Why = 'workflow lookalikes must escalate' },
        @{ Path = '.github/workflows/new-unknown.yml'; Why = 'unknown workflows must escalate' }
    )) {
        $r = Select-ImpactedShards -Map $map -Changed @{ $change.Path = @(1, 2) }
        Assert-True $r.Escalate $change.Why
        Assert-True $r.RequiresValidation "$($change.Why) and require validation"
    }

    # The exact counterexample that exposed the original defect: two GitHub-hosted documentation
    # files plus an independent release workflow must not instantiate the model/test graph.
    $r = Select-ImpactedShards -Map $map -Changed @{
        '.github/AUTOMATED_RELEASE_SETUP.md' = @(1, 2)
        '.github/VERSIONING.md' = @(1, 2)
        '.github/workflows/release-please.yml' = @(1, 2)
    }
    Assert-True (-not $r.Escalate) 'the PR #2118 path set must not escalate'
    Assert-True (-not $r.RequiresValidation) 'the PR #2118 path set must suppress runtime validation'
    Assert-True (@($r.Shards).Count -eq 0) 'the PR #2118 path set must select zero shards'

    $r = Select-ImpactedShards -Map $map -Changed @{
        'src/Covered.cs' = @(12, 14)
        '.github/workflows/release-please.yml' = @(1, 2)
        'docs/usage.md' = @(1, 2)
    }
    Assert-True (-not $r.Escalate) 'non-runtime files mixed with a covered edit must stay reducible'
    Assert-True $r.RequiresValidation 'a mixed change containing source must require validation'
    Assert-True ($r.Shards -contains 'Alpha') 'a mixed change must retain the mapped shard'
    Assert-True ($r.Shards -contains 'HeavyNoCoverage') 'a mixed change must retain always-run shards'

    $r = Select-ImpactedShards -Map $map -Changed @{
        'src/Covered.cs' = @(12, 14)
        '.github/workflows/sonarcloud.yml' = @(1, 2)
    } -CurrentPaths @('src/Covered.cs')
    Assert-True (-not $r.Escalate) `
        'a historical selector-control edit permanently escalated a later mapped PR'
    Assert-True ($r.Shards -contains 'Alpha') `
        'ignoring historical control churn dropped the mapped runtime shard'

    $r = Select-ImpactedShards -Map $map -Changed @{
        'src/Covered.cs' = @(12, 14)
        '.github/workflows/sonarcloud.yml' = @(1, 2)
    } -CurrentPaths @('.github/workflows/sonarcloud.yml', 'src/Covered.cs')
    Assert-True $r.Escalate `
        'a selector-control edit in the current PR did not force complete validation'

    $r = Select-ImpactedShards -Map $map -Changed @{}
    Assert-True $r.Escalate 'an empty changed path set must fail closed'
    Assert-True $r.RequiresValidation 'an empty changed path set must require validation'

    $r = Select-ImpactedShards -Map $map -Changed @{} -AuditUnchangedMap
    Assert-True (-not $r.Escalate) 'an unchanged map audit must exercise reduced selection'
    Assert-True $r.RequiresValidation 'an unchanged map audit must still represent runtime validation'
    Assert-True (($r.Shards -join ',') -eq 'HeavyNoCoverage') `
        'an unchanged map audit must select exactly the always-run shards'

    $fullyMapped = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $fullyMapped.alwaysRun = @()
    $r = Select-ImpactedShards -Map $fullyMapped -Changed @{} -AuditUnchangedMap
    Assert-True $r.Escalate `
        'a vacuous unchanged audit with no always-run shards must not authorize certification'

    foreach ($edge in @(10, 20)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True ($r.Shards -contains 'Alpha') "executed boundary line $edge must select Alpha"
    }
    foreach ($edge in @(9, 21)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True (-not $r.Escalate -and $r.Shards -contains 'Alpha' -and $r.Shards -contains 'Beta') `
            "unexecuted boundary line $edge was not routed to the file's owners"
        Assert-True (@($r.Routes | Where-Object { $_ -like 'Alpha <= executes changed lines*' }).Count -eq 0) `
            "unexecuted boundary line $edge was reported as executed"
    }

    # ---- A pull request BEHIND its base branch (the #2100 shape). -------------------------------
    # The map-to-HEAD delta holds master's own later commits (a selector-control edit and an
    # unmapped source file) as well as the pull request's one covered edit.
    $behind = @{
        'src/Covered.cs' = @(12, 14)
        'tools/TestImpact/Select-Shards.ps1' = @(1, 2)
        'src/Finance/MasterOnly.cs' = @(1, 5)
        '.github/workflows/sonarcloud.yml' = @(1, 2)
    }
    # Reproduction: the stale event base.sha made every path current, and it escalated.
    $r = Select-ImpactedShards -Map $map -Changed $behind -CurrentPaths @($behind.Keys)
    Assert-True $r.Escalate 'fixture error: the stale-base reproduction did not escalate'
    # Fixed: the merge commit's first parent limits the change to the pull request's own path.
    $r = Select-ImpactedShards -Map $map -Changed $behind -CurrentPaths @('src/Covered.cs') -ScopeToCurrentPaths
    Assert-True (-not $r.Escalate) 'a pull request behind master escalated on master''s own later commits'
    Assert-True ((@($r.Shards) -join ',') -eq 'Alpha,HeavyNoCoverage') `
        'a pull request behind master did not select exactly its own covering shard plus always-run'
    Assert-True (-not (@($r.Routes) -match 'MasterOnly')) 'master''s later source file influenced the pull request'

    # Scoping must still escalate a control edit that IS in the pull request.
    $r = Select-ImpactedShards -Map $map -Changed $behind `
        -CurrentPaths @('src/Covered.cs', 'tools/TestImpact/Select-Shards.ps1') -ScopeToCurrentPaths
    Assert-True $r.Escalate 'a selector edit inside a behind pull request did not escalate'

    # An empty pull-request path set must never read as a non-runtime change.
    $r = Select-ImpactedShards -Map $map -Changed $behind -CurrentPaths @() -ScopeToCurrentPaths
    Assert-True ($r.Escalate -and $r.RequiresValidation) 'an empty pull-request path set suppressed validation'

    # A pull-request path with no map-to-HEAD hunks cannot be placed in map coordinates.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14) } `
        -CurrentPaths @('src/Covered.cs', 'src/SingleRange.cs') -ScopeToCurrentPaths
    Assert-True $r.Escalate 'a pull-request path absent from the map delta did not escalate'

    # ---- Unmapped sources. ------------------------------------------------------------------------
    $deepMap = @{
        schemaVersion = 1; sha = '0123456789abcdef0123456789abcdef01234567'
        knownShards = @('Alpha', 'Beta', 'Gamma'); alwaysRun = @()
        files = @{
            'src/Finance/Agents/Dqn.cs' = @( @{ s = 0; r = @(1, 9) } )
            'src/Finance/Agents/Sac.cs' = @( @{ s = 1; r = @(1, 9) } )
            'src/Finance/Data/Feed.cs' = @( @{ s = 2; r = @(1, 9) } )
        }
    } | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Agents/NewAgent.cs' = @(1, 40) }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -eq 'Alpha,Beta') `
        'a new source file was not routed to exactly the owners of its own directory'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Brand/New/Deep.cs' = @(1, 4) }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -eq 'Alpha,Beta,Gamma') `
        'a new source file with no mapped sibling did not climb to its nearest mapped ancestor'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Orphan/Thing.cs' = @(1, 4) }
    Assert-True $r.Escalate 'a new source file with no mapped neighbour below the root escalated nothing'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Agents/agents.json' = @(1, 4) }
    Assert-True $r.Escalate 'an unmapped non-C# file inside a mapped directory was routed instead of escalating'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Agents/Finance.csproj' = @(1, 4) }
    Assert-True $r.Escalate 'an unmapped project file was routed instead of escalating'

    # ---- Test sources, routed through precomputed manifest routes. --------------------------------
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tests/P/FooTests.cs' = @(1, 30) } -TestRoutes @{
        'tests/P/FooTests.cs' = [pscustomobject]@{ Routable = $true; Shards = @('Beta'); Why = 'filter selects tests in this source' }
    }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -eq 'Beta,HeavyNoCoverage') `
        'a routable test source did not select exactly its owning shard plus always-run'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tests/P/Helpers.cs' = @(1, 30) } -TestRoutes @{
        'tests/P/Helpers.cs' = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'declares extension methods' }
    }
    Assert-True ($r.Escalate -and ($r.Reasons -join ';') -like '*declares extension methods*') `
        'an unroutable test source did not escalate with its reason'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tests/P/FooTests.cs' = @(1, 30) }
    Assert-True $r.Escalate 'a test source with no route (no manifest) did not fail closed'

    # ---- VSTest filter grammar and three-valued evaluation. ---------------------------------------
    function New-Candidate([string] $Fqn, [bool] $PrefixOnly = $false, $Categories = @(), [bool] $AnySuffix = $false) {
        [pscustomobject]@{ Fqn = $Fqn; PrefixOnly = $PrefixOnly; AnySuffix = $AnySuffix; Categories = $Categories }
    }
    $f = ConvertTo-TestFilter "Category!=GPU&Category!=Stress& `n (FullyQualifiedName~UnitTests.Alpha|`n FullyQualifiedName~UnitTests.Beta)"
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Beta.C.M')) -eq $script:FilterTrue) `
        'a folded multi-line filter did not match its second alternative'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Gamma.C.M')) -eq $script:FilterFalse) `
        'a filter matched a test outside every alternative'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Alpha.C.M' $false @('GPU'))) -eq $script:FilterUnknown) `
        'a category present somewhere in the file was treated as definitely on this test'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Alpha.C.M' $false $null)) -eq $script:FilterUnknown) `
        'unknown categories were treated as known'
    $f = ConvertTo-TestFilter 'A=1|B=2&FullyQualifiedName~Nope'
    Assert-True ($f.Kind -eq 'Or' -and $f.Items[1].Kind -eq 'And') "'&' did not bind tighter than '|'"
    $f = ConvertTo-TestFilter 'FullyQualifiedName~Contracts.A&FullyQualifiedName!~Skip'
    Assert-True ((Test-TestFilter $f (New-Candidate 'N.Contracts.' $true)) -eq $script:FilterUnknown) `
        'an inherited test with an unknown method name was excluded by a method-level filter'
    Assert-True ((Test-TestFilter $f (New-Candidate 'N.Contracts.Skip' $true)) -eq $script:FilterFalse) `
        'a known prefix containing an excluded term was not excluded'
    # The unknown part of an inherited test is ONE method identifier, so it cannot supply a
    # namespace term the prefix lacks - but it can finish a term the prefix ends in.
    $f = ConvertTo-TestFilter 'FullyQualifiedName~P.Alpha'
    Assert-True ((Test-TestFilter $f (New-Candidate 'P.Beta.BTests.' $true)) -eq $script:FilterFalse) `
        'an inherited test was treated as able to acquire a namespace from its method name'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.P.' $true)) -eq $script:FilterUnknown) `
        'a method name completing a term the prefix ends in was ruled out'
    Assert-True ((Test-TestFilter $f (New-Candidate 'N.' $true @() $true)) -eq $script:FilterUnknown) `
        'a test whose class could not be read was ruled out by a namespace term'
    Assert-True ((Test-TestFilter (ConvertTo-TestFilter 'FullyQualifiedName~alpha') (New-Candidate 'X.Alpha.M')) -eq $script:FilterUnknown) `
        'a case-only match was decided instead of left unknown'
    Assert-True ((Test-TestFilter (ConvertTo-TestFilter 'Category=Playground') (New-Candidate 'X.Y.M')) -eq $script:FilterFalse) `
        'a positive category filter matched a file with no categories'
    foreach ($bad in @('FullyQualifiedName~A&', '(FullyQualifiedName~A', 'FullyQualifiedName', 'A=\(x\)')) {
        Assert-Throws { ConvertTo-TestFilter $bad } "malformed filter '$bad' was accepted"
    }

    # ---- C# test-shape parsing. -------------------------------------------------------------------
    $source = @'
using Xunit;
namespace AiDotNet.Tests.IntegrationTests.Finance;

// public class CommentedOut { [Fact] public void Ghost() {} }
[Trait("Category", "Slow")]
public class TradingTests : IClassFixture<Fixture>
{
    private const string Braces = "{ class Fake { ";
    private readonly char _brace = '}';

    [Fact]
    public void Learns() { var s = $"{1}"; }

    [Theory]
    [InlineData(new[] { 1, 2 })]
    public async Task Converges<T>(int[] x) { await Task.Yield(); }

    public class Nested
    {
        [Fact, Trait("Category", "Nested")]
        public void Inner() { }
    }
}

internal sealed class Helper { }
file class Private { }
'@
    $shape = Get-CSharpTestShape -Text $source
    Assert-True ($null -eq $shape.ParseError) "valid C# was reported unparseable: $($shape.ParseError)"
    $fqns = @($shape.Tests | Where-Object { -not $_.PrefixOnly } | ForEach-Object Fqn | Sort-Object)
    $expectedFqns = @('AiDotNet.Tests.IntegrationTests.Finance.TradingTests+Nested.Inner',
        'AiDotNet.Tests.IntegrationTests.Finance.TradingTests.Converges',
        'AiDotNet.Tests.IntegrationTests.Finance.TradingTests.Learns') | Sort-Object
    Assert-True (($fqns -join ',') -eq ($expectedFqns -join ',')) "test names were misread: $($fqns -join ',')"
    Assert-True (@($shape.Tests | Where-Object PrefixOnly).Count -eq 0) 'an interface base was treated as an inherited class'
    Assert-True ((@($shape.Categories | Sort-Object) -join ',') -eq 'Nested,Slow') 'literal categories were not collected'
    $topLevel = @($shape.Types | Where-Object { $null -eq $_.Parent -and -not $_.FileLocal } | ForEach-Object Name | Sort-Object)
    Assert-True (($topLevel -join ',') -eq 'Helper,TradingTests') `
        "referenceable top-level types were misread: $($topLevel -join ',')"

    $shape = Get-CSharpTestShape -Text "namespace N { public class D : ModelContractBase<int> { } }"
    Assert-True (@($shape.Tests | Where-Object { $_.PrefixOnly -and $_.Fqn -eq 'N.D.' }).Count -eq 1) `
        'a class with a class base did not expose its inherited tests as open-ended'
    Assert-True ($null -eq $shape.Categories) 'categories inherited from a base class were treated as known'

    $shape = Get-CSharpTestShape -Text "namespace N { public static class X { public static int Twice(this int v) => v * 2; } }"
    Assert-True ($shape.Hazard -eq 'declares extension methods') 'extension methods were not flagged'
    $shape = Get-CSharpTestShape -Text "namespace N { public class X { "
    Assert-True ([bool] $shape.ParseError) 'unbalanced braces were not reported'

    # ---- Routing with injected file reads and reference search. -----------------------------------
    $manifest = @(
        [pscustomobject]@{ name = 'Alpha'; project = 'tests/P/P.csproj'; filter = 'FullyQualifiedName~P.Alpha' },
        [pscustomobject]@{ name = 'Beta'; project = 'tests/P/P.csproj'; filter = 'FullyQualifiedName~P.Beta' },
        [pscustomobject]@{ name = 'Other'; project = 'tests/Q/Q.csproj'; filter = 'FullyQualifiedName~P' }
    )
    $files = @{
        'tests/P/Alpha/ATests.cs' = 'namespace P.Alpha; public class ATests { [Fact] public void A() { } }'
        'tests/P/Shared/Base.cs'  = 'namespace P.Shared; public class Base { [Fact] public void Common() { } }'
        'tests/P/Shared/Abstract.cs' = 'namespace P.Shared; public abstract class ShapeBase { [Fact] public void Common() { } }'
        'tests/P/Shared/GenBase.cs' = 'namespace P.Shared; public class LayerHarness { }'
        'tests/P/Beta/BTests.cs'  = 'namespace P.Beta; public class BTests : Base { }'
        'tests/P/Shared/Ext.cs'   = 'namespace P.Shared; public static class Ext { public static int X(this int v) => v; }'
        'tests/P/Shared/Unused.cs' = 'namespace P.Shared; public class Unused { }'
    }
    $referrers = @{ 'Base' = @('tests/P/Beta/BTests.cs'); 'ATests' = @(); 'BTests' = @(); 'Unused' = @() }
    $read = { param($path) $files[$path] }
    $find = { param($names, $directory) @($names | ForEach-Object { $referrers[$_] } | Where-Object { $_ }) }
    $generatorNames = @('LayerHarness')
    $findGenerated = { param($names) @($names | Where-Object { $generatorNames -contains $_ }) }
    $routed = Get-TestFileRoutes -Paths @('tests/P/Alpha/ATests.cs', 'tests/P/Shared/Base.cs', 'tests/P/Shared/Ext.cs',
        'tests/P/Shared/Unused.cs', 'tests/Z/Stray.cs', 'tests/P/Shared/Abstract.cs', 'tests/P/Shared/GenBase.cs') `
        -Manifest $manifest -ReadFile $read -FindReferrers $find -FindBuildTimeReferences $findGenerated
    Assert-True ($routed['tests/P/Alpha/ATests.cs'].Routable -and
        (@($routed['tests/P/Alpha/ATests.cs'].Shards) -join ',') -eq 'Alpha') `
        'a self-contained test file was not routed to exactly its own shard'
    Assert-True ($routed['tests/P/Shared/Base.cs'].Routable -and
        (@($routed['tests/P/Shared/Base.cs'].Shards) -join ',') -eq 'Beta') `
        'a base class was not routed to the shard running its derived tests (and not its own namespace)'
    Assert-True (-not $routed['tests/P/Shared/Ext.cs'].Routable) 'an extension-method helper was routed'
    Assert-True (-not $routed['tests/P/Shared/Unused.cs'].Routable) 'a support file with no tests and no consumers was routed'
    Assert-True (-not $routed['tests/Z/Stray.cs'].Routable) 'a test file outside every shard project was routed'
    Assert-True (-not $routed['tests/P/Shared/Abstract.cs'].Routable) `
        'an abstract test base was routed although build-time generated classes may derive from it'
    Assert-True (-not $routed['tests/P/Shared/GenBase.cs'].Routable -and
        $routed['tests/P/Shared/GenBase.cs'].Why -like '*source generator references*') `
        'a test type a source generator references was routed by reading only the tree'
    $wide = Get-TestFileRoutes -Paths @('tests/P/Shared/Base.cs') -Manifest $manifest -ReadFile $read `
        -FindReferrers $find -FindBuildTimeReferences $findGenerated -MaximumClosure 1
    Assert-True (-not $wide['tests/P/Shared/Base.cs'].Routable) 'a closure wider than the limit was routed'

    # An empty build-time pathspec would make git grep search the whole repository.
    $configuredBuildTime = $script:BuildTimeDirectories
    try {
        $script:BuildTimeDirectories = @()
        Assert-Throws { Find-GitBuildTimeReferences -Names @('Anything') } `
            'an empty build-time directory list searched the whole repository instead of failing'
    }
    finally { $script:BuildTimeDirectories = $configuredBuildTime }

    # Faithful to real git: every hunk header is followed by its body lines. An earlier revision
    # used header-only fixtures, which real git never emits - and which masked the forged-header
    # parse bug that body-line counting exists to prevent (see check 15).
    $diff = @('--- a/src/Covered.cs', '+++ b/src/Covered.cs', '@@ -1,0 +1,50 @@') +
            @(1..50 | ForEach-Object { "+inserted $_" }) +
            @('@@ -500 +550 @@', '-old text', '+new text')
    $parsed = ConvertTo-ChangedRanges -DiffLines $diff
    $ranges = @($parsed['src/Covered.cs'])
    Assert-True ($ranges -contains 500) 'the parser must report old-side line numbers'
    Assert-True (-not ($ranges -contains 550)) 'the parser must not report new-side line numbers'

    $diff = @('--- a/src/Covered.cs', '+++ b/src/Covered.cs', '@@ -100 +100 @@', '-x', '+y',
              '--- a/src/Deleted.cs', '+++ /dev/null', '@@ -1,200 +0,0 @@') +
            @(1..200 | ForEach-Object { "-gone $_" })
    $parsed = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($parsed.ContainsKey('src/Deleted.cs')) 'a deleted file must be reported under its own path'
    Assert-True (@($parsed['src/Covered.cs']).Count -eq 2) 'a deletion must not contaminate the prior file'

    $parsed = ConvertTo-ChangedRanges -DiffLines @() -ChangedFiles @('src/Renamed.cs', 'assets/blob.bin')
    Assert-True ($parsed.ContainsKey('src/Renamed.cs')) 'a rename-only file must be present without a hunk'
    Assert-True ($parsed.ContainsKey('assets/blob.bin')) 'a binary file must be present without a hunk'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @() }
    Assert-True $r.Escalate 'a mapped file without hunks must escalate'

    $badRange = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badRange.files.'src/Covered.cs'[0].r = @(10)
    Assert-Throws { Assert-ShardMap -Map $badRange -Expected $expected } 'an odd range list must be rejected'

    $badIndex = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badIndex.files.'src/Covered.cs'[0].s = 99
    Assert-Throws { Assert-ShardMap -Map $badIndex -Expected $expected } 'an out-of-range shard index must be rejected'

    $badSchema = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badSchema.schemaVersion = 2
    Assert-Throws { Assert-ShardMap -Map $badSchema -Expected $expected } 'an unknown map schema must be rejected'

    $overlap = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $overlap.alwaysRun = @('Alpha')
    Assert-Throws { Assert-ShardMap -Map $overlap -Expected @('Alpha', 'Beta') } `
        'known and always-run shard sets must be disjoint'

    $badLine = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badLine.files.'src/Covered.cs'[0].r = @(0, 10)
    Assert-Throws { Assert-ShardMap -Map $badLine -Expected $expected } 'non-positive map lines must be rejected'

    Assert-Throws { Assert-ShardMap -Map $map -Expected @('Alpha', 'Beta', 'HeavyNoCoverage', 'NewShard') } `
        'a stale shard universe must be rejected'

    $commaMap = @{
        schemaVersion = 1; sha = 'abcdefabcdefabcdefabcdefabcdefabcdefabcd'; knownShards = @('Comma, Shard'); alwaysRun = @()
        files = @{ 'src/Comma.cs' = @( @{ s = 0; r = @(1, 1) } ) }
    } | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    try { Assert-ShardMap -Map $commaMap -Expected @('Comma, Shard') } catch {
        [void] $failures.Add("a comma-containing shard name was rejected: $_")
    }
    $r = Select-ImpactedShards -Map $commaMap -Changed @{ 'src/Comma.cs' = @(1, 1) }
    Assert-True ($r.Shards -contains 'Comma, Shard') 'a comma-containing shard name must stay intact'

    # 14. Shared-infrastructure matching. File entries match by BASENAME at any depth - MSBuild
    #     and NuGet apply these files per-directory, and src/Directory.Build.props exists in this
    #     repo - while look-alike names must not match, and directories stay prefix-matched.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Directory.Build.props' = @(1, 2) }
    Assert-True $r.Escalate 'a NESTED Directory.Build.props must escalate'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/sub/nuget.config' = @(1, 2) }
    Assert-True $r.Escalate 'a nested nuget.config must escalate'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props.backup' = @(1, 2) }
    Assert-True (-not ($r.Reasons -join ';').Contains('shared infrastructure')) `
        'a look-alike suffix must not match shared infrastructure'
    $r = Select-ImpactedShards -Map $map -Changed @{ '.github/workflows/docs.yml' = @(1, 2) }
    Assert-True (-not $r.Escalate -and -not $r.RequiresValidation) `
        'an independent YAML workflow must be classified as non-runtime'

    # 15. Hunk BODY content must be inert. A removed line whose text begins with '-- ' renders as
    #     '--- ...', byte-identical to an old-file header; the pre-fix parser nulled $current on it
    #     and silently dropped every later hunk of the file - under-selection with no escalation.
    #     This diff is verbatim real-git output for: delete the line '-- remove me', edit line 81.
    $diff = @(
        'diff --git a/F.cs b/F.cs',
        'index 9329992..2c35db3 100644',
        '--- a/F.cs',
        '+++ b/F.cs',
        '@@ -50 +49,0 @@ line 49',
        '--- remove me',
        '@@ -81 +80 @@ line 80',
        '-line 81',
        '+line 80 EDITED'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($r.ContainsKey('F.cs')) 'the file must be reported'
    Assert-True (@($r['F.cs']) -contains 81) 'the hunk AFTER the forged header line must survive'
    Assert-True (-not $r.ContainsKey('remove me')) 'body content must never become a path'

    # 16. And a forged header inside an ADDED body line must not smuggle a file in.
    $diff = @(
        '--- a/G.cs',
        '+++ b/G.cs',
        '@@ -5,0 +6,2 @@',
        '+--- a/EVIL.cs',
        '++++ b/EVIL.cs',
        '@@ -30 +32 @@',
        '-x',
        '+y'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True (-not $r.ContainsKey('EVIL.cs')) 'added body content must never become a path'
    Assert-True (@($r['G.cs']) -contains 30) 'the following real hunk still lands on the right file'

    if ($failures.Count -gt 0) {
        Write-Host 'Select-Shards self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'Select-Shards self-test passed.'
    exit 0
}

# ---------------------------------------------------------------- selection

if ($ClassifyOnly) {
    $requiresValidation = $true
    $reason = 'classification-failed'
    $changedFiles = @()
    try {
        if ($PullRequestHeadSha -and $BaseSha) { throw 'pass PullRequestHeadSha or BaseSha, not both' }
        if ($PullRequestHeadSha) { $BaseSha = Resolve-PullRequestBase -PullRequestHeadSha $PullRequestHeadSha }
        if (-not $BaseSha) { throw 'classification needs PullRequestHeadSha or BaseSha' }
        & git cat-file -e "$BaseSha^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) { throw "base commit '$BaseSha' is not present in this checkout" }
        # A source-to-Markdown rename must report both the deleted source path and added Markdown
        # path; otherwise looking only at the destination could incorrectly authorize no CI.
        $changedFiles = @(& git -c core.quotepath=false diff --no-renames --name-only $BaseSha HEAD --)
        if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from '$BaseSha' failed" }
        $changedFiles = @($changedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($changedFiles.Count -eq 0) {
            $reason = 'changed-path-set-empty'
        }
        else {
            $requiresValidation = [bool] @(
                $changedFiles | Where-Object {
                    (Get-ChangedPathImpact -Path ([string] $_)) -ne [ChangedPathImpact]::NonRuntime
                }
            ).Count
            $reason = $(if ($requiresValidation) { 'runtime-or-unknown' } else { 'non-runtime-only' })
        }
    }
    catch {
        Write-Host "::warning::path classification failed, so runtime validation remains required: $($_.Exception.Message)"
    }

    $result = [pscustomobject]@{
        requiresValidation = $requiresValidation
        reason = $reason
        baseSha = [string] $BaseSha
        changedPaths = @($changedFiles)
    }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

function Exit-Escalated {
    param([Parameter(Mandatory)] [string] $Reason, [string] $Message)

    if ($Message) { Write-Host "::warning::$Message" }
    $result = [pscustomobject]@{
        escalate = $true; requiresValidation = $true; reason = $Reason; reasons = @(); shards = @()
    }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

if (-not (Test-Path -LiteralPath $MapFile)) {
    Exit-Escalated -Reason 'map-missing' -Message "no shard map at $MapFile - running the full matrix"
}

$map = $null
try {
    $map = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
    Assert-ShardMap -Map $map -Expected $ExpectedShards
}
catch {
    Exit-Escalated -Reason 'map-unreadable' `
        -Message "the shard map at $MapFile could not be trusted - running the full matrix ($($_.Exception.Message))"
}

$mapSha = [string] $map.sha
& git cat-file -e "$mapSha^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Exit-Escalated -Reason 'map-unresolvable' `
        -Message "the map's commit $mapSha is not present in this checkout - running the full matrix"
}

try {
    $changed = Get-ChangedRanges -MapSha $mapSha
    $currentPaths = @($changed.Keys)
    $scopeToPullRequest = $false
    if ($PullRequestHeadSha -and $BaseSha) { throw 'pass PullRequestHeadSha or BaseSha, not both' }
    if ($PullRequestHeadSha) {
        $BaseSha = Resolve-PullRequestBase -PullRequestHeadSha $PullRequestHeadSha
        $scopeToPullRequest = $true
        Write-Host "pull request base (merge commit's first parent): $BaseSha"
    }
    if ($BaseSha) {
        & git cat-file -e "$BaseSha^{commit}"
        if ($LASTEXITCODE -ne 0) { throw "base commit '$BaseSha' is not present in this checkout" }
        $currentPaths = @(& git -c core.quotepath=false diff --no-renames --name-only $BaseSha HEAD --)
        if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from current base '$BaseSha' failed" }
        $currentPaths = @($currentPaths | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }
    Write-Host "changed files since the map: $($changed.Count); changed by the current change: $($currentPaths.Count)"

    # Test sources the map cannot index, among the paths selection will actually consider.
    $testRoutes = @{}
    if ($ShardManifestFile) {
        $manifest = @(Get-Content -LiteralPath $ShardManifestFile -Raw | ConvertFrom-Json)
        $manifestNames = @($manifest | ForEach-Object { [string] $_.name } | Sort-Object)
        if (($manifestNames -join "`n") -cne (@($ExpectedShards | Sort-Object) -join "`n")) {
            throw 'the shard manifest does not describe exactly the expected shards'
        }
        $currentSet = [System.Collections.Generic.HashSet[string]]::new([string[]] $currentPaths, [StringComparer]::OrdinalIgnoreCase)
        $testPaths = @($changed.Keys | Where-Object {
            $candidate = [string] $_
            (-not $scopeToPullRequest -or $currentSet.Contains($candidate)) -and
            $candidate.StartsWith('tests/', [StringComparison]::OrdinalIgnoreCase) -and
            $candidate.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase) -and
            (Get-ChangedPathImpact -Path $candidate) -eq [ChangedPathImpact]::MapCandidate -and
            -not $map.files.PSObject.Properties[$candidate]
        })
        $revisions = @('HEAD', $BaseSha, $mapSha)
        $testRoutes = Get-TestFileRoutes -Paths $testPaths -Manifest $manifest `
            -ReadFile { param($path) Get-GitFileText -Path $path -Revisions $revisions } `
            -FindReferrers { param($names, $directory) Find-GitReferrers -Names $names -Directory $directory } `
            -FindBuildTimeReferences { param($names) Find-GitBuildTimeReferences -Names $names }
    }

    $selection = Select-ImpactedShards -Map $map -Changed $changed -CurrentPaths $currentPaths `
        -ScopeToCurrentPaths:$scopeToPullRequest -TestRoutes $testRoutes `
        -AuditUnchangedMap:$AuditUnchangedMap

    if ($selection.Escalate) {
        Write-Host '::warning::selection escalated to the full matrix'
        foreach ($reason in $selection.Reasons) { Write-Host "  reason: $reason" }
    }
    else {
        Write-Host "selected $($selection.Shards.Count) of $($ExpectedShards.Count) shard(s)"
        foreach ($shard in $selection.Shards) {
            Write-Host "  $shard"
            $why = @($selection.Routes | Where-Object { $_.StartsWith("$shard <= ", [StringComparison]::Ordinal) } |
                ForEach-Object { $_.Substring($shard.Length + 4) })
            foreach ($line in @($why | Select-Object -First 5)) { Write-Host "      because it $line" }
            if ($why.Count -gt 5) { Write-Host "      ... and $($why.Count - 5) more reason(s)" }
        }
    }

    $result = [pscustomobject]@{
        escalate          = $selection.Escalate
        requiresValidation = $selection.RequiresValidation
        reason            = $(if ($selection.Escalate) { 'impact-unknown' }
                              elseif (-not $selection.RequiresValidation) { 'non-runtime-only' }
                              else { 'selected' })
        reasons           = $selection.Reasons
        routes            = @($selection.Routes)
        shards            = $selection.Shards
    }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
}
catch {
    Exit-Escalated -Reason 'selection-failed' `
        -Message "shard selection failed, so the full matrix will run: $($_.Exception.Message)"
}
