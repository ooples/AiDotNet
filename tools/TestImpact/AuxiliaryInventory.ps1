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
