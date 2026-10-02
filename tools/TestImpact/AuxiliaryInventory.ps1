# Model windows are ordinal partitions of a reflection-discovered catalog. A type or
# constructor signature change can move models between windows even if their executed
# source lines are unchanged. Coverage cannot establish that catalog stability.
function Get-AuxiliaryDeclarationTokens {
    param([Parameter(Mandatory)] [object] $Node)
    $text = [Text.StringBuilder]::new()
    foreach ($token in $Node.DescendantTokens()) {
        # Length-prefix text so punctuation in attribute strings cannot collide with
        # token boundaries. Trivia is absent: comments/formatting do not move models.
        [void] $text.Append($token.RawKind).Append(':').Append($token.Text.Length).Append(':').Append($token.Text)
    }
    return $text.ToString()
}

function Get-AuxiliaryInventorySignature {
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Source)
    # Conditional compilation must not hide the active configuration from a syntax-only
    # check. Conservatively compare the entire file when directives could be present.
    if ($Source -match '(?m)^\s*#(?:if|elif|else|endif|define|undef)\b') { return $Source }
    $tree = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($Source)
    if (@($tree.GetDiagnostics() | Where-Object Severity -eq Error).Count -gt 0) {
        throw 'Cannot establish the reflection inventory from invalid or unsupported syntax.'
    }
    $parts = [Collections.Generic.List[string]]::new()
    $emptyMembers = [Activator]::CreateInstance(
        [Microsoft.CodeAnalysis.SyntaxList[Microsoft.CodeAnalysis.CSharp.Syntax.MemberDeclarationSyntax]])
    foreach ($node in $tree.GetRoot().DescendantNodes()) {
        if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.UsingDirectiveSyntax]) {
            $parts.Add((Get-AuxiliaryDeclarationTokens $node))
        }
        elseif ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.AttributeListSyntax] -and
                $null -ne $node.Target -and $node.Target.Identifier.ValueText -cin @('assembly', 'module')) {
            $parts.Add((Get-AuxiliaryDeclarationTokens $node))
        }
        elseif ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseNamespaceDeclarationSyntax]) {
            $parts.Add((Get-AuxiliaryDeclarationTokens $node.Name))
        }
        elseif ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax]) {
            $parts.Add((Get-AuxiliaryDeclarationTokens ($node.WithMembers($emptyMembers))))
        }
        elseif ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseMethodDeclarationSyntax]) {
            $parts.Add((Get-AuxiliaryDeclarationTokens ($node.WithBody($null).WithExpressionBody($null))))
        }
        elseif ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.PropertyDeclarationSyntax] -or
                $node -is [Microsoft.CodeAnalysis.CSharp.Syntax.IndexerDeclarationSyntax]) {
            $parts.Add((Get-AuxiliaryDeclarationTokens ($node.WithAccessorList($null).WithExpressionBody($null))))
            if ($null -ne $node.AccessorList) {
                foreach ($accessor in $node.AccessorList.Accessors) {
                    $parts.Add((Get-AuxiliaryDeclarationTokens ($accessor.WithBody($null).WithExpressionBody($null))))
                }
            }
        }
        elseif ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.MemberDeclarationSyntax]) {
            # Enums, fields, delegates, events and unsupported member forms retain their
            # complete syntax. This may widen a run but cannot authorize an unsafe skip.
            $parts.Add((Get-AuxiliaryDeclarationTokens $node))
        }
    }
    return $parts -join "`n"
}

function Test-AuxiliaryInventoryChange {
    param([Parameter(Mandatory)] [string] $MapSha)
    try {
        $sha = $MapSha
        if ($sha -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid map source SHA.' }
        # Coverage is also tied to the execution configuration: a dependency or runner change must
        # not reuse old indexed membership. A window's own definition (offset, budget, filter) is
        # compared per shard by Select-Shards, and another shard's entry cannot move models between
        # windows, so manifest edits alone do not widen every window.
        & git diff --quiet $sha HEAD -- .github/workflows/sonarcloud.yml `
            coverlet.runsettings Directory.Packages.props Directory.Build.props Directory.Build.targets global.json `
            tools/TestImpact/Connect-WorkerCoverage.ps1 tools/TestImpact/Write-AuxiliaryEvidence.ps1
        if ($LASTEXITCODE -ne 0) { return $true }
        $paths = @(& git -c core.quotepath=false diff --no-renames --name-only $sha HEAD -- src)
        if ($LASTEXITCODE -ne 0) { throw 'Could not compare the map source to the current tree.' }
        foreach ($path in $paths) {
            if (-not $path.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase)) { continue }
            $before = @(& git show "${sha}:$path" 2>$null)
            if ($LASTEXITCODE -ne 0) { return $true }
            $after = @(& git show "HEAD:$path" 2>$null)
            if ($LASTEXITCODE -ne 0) { return $true }
            if ((Get-AuxiliaryInventorySignature ($before -join "`n")) -cne
                (Get-AuxiliaryInventorySignature ($after -join "`n"))) {
                Write-Host "Auxiliary inventory changed since mapping: $path"
                return $true
            }
        }
        return $false
    }
    catch {
        Write-Warning "Auxiliary inventory is unproven; retaining all auxiliary workloads: $($_.Exception.Message)"
        return $true
    }
    finally { $global:LASTEXITCODE = 0 }
}

# Window shards slice the model catalog by NAME since the stable assignment (#2276): ADNSHAPE_CONF_WINDOW
# reads the committed table in ModelContractConformanceTests.Windows.cs, AIDOTNET_PARAMETER_COUNT_SHARD a
# stable hash of the type name. A model can only change window when its own identity or constructors change,
# so a declaration change invalidates the windows of the names it touches, not every window.
$script:ConformanceWindowTable = 'tests/AiDotNet.Tests/NeuralNetworks/Graph/ModelContractConformanceTests.Windows.cs'
$script:CoverageSemanticsPaths = @('coverlet.runsettings', 'tools/TestImpact/Connect-WorkerCoverage.ps1', 'tools/TestImpact/Write-AuxiliaryEvidence.ps1')
$script:ExecutionConfigurationPaths = @('.github/workflows/sonarcloud.yml', 'Directory.Packages.props', 'Directory.Build.props',
    'Directory.Build.targets', 'global.json')

function Get-StableParameterCountShard {
    # Must equal ParameterCountContractTests.ShardOf: FNV-1a over the UTF-16 code units of the full name.
    param([Parameter(Mandatory)] [string] $FullName, [int] $Count = 8)
    [uint64] $hash = 2166136261
    foreach ($c in $FullName.ToCharArray()) {
        # [uint64] throughout: a bare 0xFFFFFFFF literal is the Int32 -1 in PowerShell and masks nothing.
        $hash = (($hash -bxor [uint64][uint32][char] $c) * [uint64] 16777619) % [uint64] 4294967296
    }
    return [int] ($hash % [uint64] $Count)
}

function Get-AuxiliaryModelIdentities {
    # Type full name (as Type.FullName spells it: Ns.Outer+Inner`1) -> its declaration header and constructor
    # signatures. $null when conditional compilation makes the active declarations unprovable.
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Source)
    if ($Source -match '(?m)^\s*#(?:if|elif|else|endif|define|undef)\b') { return $null }
    $tree = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($Source)
    if (@($tree.GetDiagnostics() | Where-Object Severity -eq Error).Count -gt 0) {
        throw 'Cannot establish the model identities from invalid or unsupported syntax.'
    }
    $emptyMembers = [Activator]::CreateInstance(
        [Microsoft.CodeAnalysis.SyntaxList[Microsoft.CodeAnalysis.CSharp.Syntax.MemberDeclarationSyntax]])
    $result = @{}
    foreach ($type in $tree.GetRoot().DescendantNodes() | Where-Object { $_ -is [Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax] }) {
        $names = [Collections.Generic.List[string]]::new()
        $namespaces = [Collections.Generic.List[string]]::new()
        for ($node = $type; $null -ne $node; $node = $node.Parent) {
            if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax]) {
                $arity = if ($null -ne $node.TypeParameterList) { '`' + $node.TypeParameterList.Parameters.Count } else { '' }
                $names.Insert(0, $node.Identifier.ValueText + $arity)
            }
            elseif ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseNamespaceDeclarationSyntax]) {
                $namespaces.Insert(0, $node.Name.ToString())
            }
        }
        $prefix = if ($namespaces.Count -gt 0) { ($namespaces -join '.') + '.' } else { '' }
        $text = [Text.StringBuilder]::new((Get-AuxiliaryDeclarationTokens ($type.WithMembers($emptyMembers))))
        foreach ($ctor in $type.Members | Where-Object { $_ -is [Microsoft.CodeAnalysis.CSharp.Syntax.ConstructorDeclarationSyntax] }) {
            [void] $text.Append("`n").Append((Get-AuxiliaryDeclarationTokens ($ctor.WithBody($null).WithExpressionBody($null).WithInitializer($null))))
        }
        $result[$prefix + ($names -join '+')] = $text.ToString()
    }
    return $result
}

function Get-ConformanceWindowAssignments {
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Source)
    $table = @{}
    foreach ($m in [regex]::Matches($Source, '\["([^"]+)"\]\s*=\s*(\d+)\s*,', 'None', [TimeSpan]::FromSeconds(2))) {
        $table[$m.Groups[1].Value] = [int] $m.Groups[2].Value
    }
    return $table
}

function Get-AuxiliaryWindowInvalidation {
    # Which window shards a change since the map can have moved models into or out of. Returns All=$true when
    # that cannot be established (the rule then keeps every window, as before).
    param(
        [Parameter(Mandatory)] [string] $MapSha,
        [string] $BaseSha,
        [Parameter(Mandatory)] [object[]] $Windows
    )
    $all = { param($why) [pscustomobject]@{ All = $true; Shards = @($Windows | ForEach-Object name); Reason = $why } }
    try {
        if ($MapSha -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid map source SHA.' }
        # Coverage semantics decide what the map's indexed membership means, so they compare against the map.
        & git diff --quiet $MapSha HEAD -- @script:CoverageSemanticsPaths
        if ($LASTEXITCODE -ne 0) { return & $all 'coverage semantics changed since the map' }
        # Execution configuration compares against the pull request's base (R2): a change that landed on the
        # base since the map was escalated in its own pull request, and is in this merge commit either way.
        $configBase = if ($BaseSha) { $BaseSha } else { $MapSha }
        & git diff --quiet $configBase HEAD -- @script:ExecutionConfigurationPaths
        if ($LASTEXITCODE -ne 0) { return & $all 'execution configuration changed' }

        $mapTableText = @(& git show "${MapSha}:$script:ConformanceWindowTable" 2>$null) -join "`n"
        if ($LASTEXITCODE -ne 0) { return & $all 'the map predates the stable window table' }
        $headTableText = @(& git show "HEAD:$script:ConformanceWindowTable" 2>$null) -join "`n"
        if ($LASTEXITCODE -ne 0) { return & $all 'the stable window table is missing' }
        $mapTable = Get-ConformanceWindowAssignments $mapTableText
        $headTable = Get-ConformanceWindowAssignments $headTableText

        $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($key in @($mapTable.Keys) + @($headTable.Keys)) {
            if (-not $mapTable.ContainsKey($key) -or -not $headTable.ContainsKey($key) -or $mapTable[$key] -ne $headTable[$key]) {
                [void] $names.Add($key)
            }
        }
        $paths = @(& git -c core.quotepath=false diff --no-renames --name-only $MapSha HEAD -- src)
        if ($LASTEXITCODE -ne 0) { throw 'Could not compare the map source to the current tree.' }
        foreach ($path in $paths | Where-Object { $_.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase) }) {
            $beforeText = @(& git show "${MapSha}:$path" 2>$null) -join "`n"
            $before = if ($LASTEXITCODE -eq 0) { Get-AuxiliaryModelIdentities $beforeText } else { @{} }
            $afterText = @(& git show "HEAD:$path" 2>$null) -join "`n"
            $after = if ($LASTEXITCODE -eq 0) { Get-AuxiliaryModelIdentities $afterText } else { @{} }
            if ($null -eq $before -or $null -eq $after) { return & $all "conditional compilation in $path" }
            foreach ($key in @($before.Keys) + @($after.Keys)) {
                if (-not $before.ContainsKey($key) -or -not $after.ContainsKey($key) -or $before[$key] -cne $after[$key]) {
                    [void] $names.Add($key)
                }
            }
        }

        $hit = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($shard in $Windows) {
            $environment = $shard.env
            $window = $environment.PSObject.Properties['ADNSHAPE_CONF_WINDOW']
            $bucket = $environment.PSObject.Properties['AIDOTNET_PARAMETER_COUNT_SHARD']
            if ($null -ne $window) {
                foreach ($name in $names) {
                    foreach ($table in $mapTable, $headTable) {
                        if ($table.ContainsKey($name) -and $table[$name] -eq [int] $window.Value) { [void] $hit.Add([string] $shard.name) }
                    }
                }
            }
            elseif ($null -ne $bucket) {
                foreach ($name in $names) {
                    if ((Get-StableParameterCountShard $name) -eq [int] $bucket.Value) { [void] $hit.Add([string] $shard.name) }
                }
            }
            else {
                # A positional window (offset, budget) or an unrecognised one: still keyed to the whole catalog.
                if ($names.Count -gt 0) { [void] $hit.Add([string] $shard.name) }
            }
        }
        return [pscustomobject]@{ All = $false; Shards = @($hit | Sort-Object); Reason = "$($names.Count) model identity change(s) since the map" }
    }
    catch {
        Write-Warning "Window membership is unproven; retaining every window: $($_.Exception.Message)"
        return & $all 'window membership could not be established'
    }
    finally { $global:LASTEXITCODE = 0 }
}