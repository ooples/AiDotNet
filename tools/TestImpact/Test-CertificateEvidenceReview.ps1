<# Execute the production tree-binding assignments and exact/delta eligibility branches offline. #>
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'Resolve-CiValidationReuse.ps1'), [ref] $tokens, [ref] $errors)
if ($errors.Count -ne 0) { throw 'The certificate resolver does not parse.' }
foreach ($declaration in $ast.EndBlock.Statements | Where-Object {
    $_ -is [System.Management.Automation.Language.FunctionDefinitionAst] -or
    $_ -is [System.Management.Automation.Language.TypeDefinitionAst]
}) { . ([scriptblock]::Create($declaration.Extent.Text)) }
$bindings = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $node.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
    $node.Left.VariablePath.UserPath -in @('certificateTreeMatches', 'treeMatches')
}, $true) | Sort-Object { $_.Extent.StartOffset })
if (@($bindings | Where-Object { $_.Left.VariablePath.UserPath -eq 'treeMatches' }).Count -ne 1) {
    throw 'Expected exactly one production landed-tree binding.'
}
$delta = @($ast.EndBlock.Statements | Where-Object {
    $_ -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $_.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
    $_.Left.VariablePath.UserPath -eq 'deltaCandidates'
})
if ($delta.Count -ne 1) { throw 'Expected exactly one production delta eligibility branch.' }
$failures = [Collections.Generic.List[string]]::new()
$cases = 0
foreach ($certificateMatches in @($true, $false)) {
    foreach ($landedMatches in @($true, $false)) {
        foreach ($requiresRuntime in @($true, $false)) {
            $cases++
            $testedTree = '0123456789abcdef0123456789abcdef01234567'
            $otherTree = '1123456789abcdef0123456789abcdef01234567'
            $masterTree = if ($landedMatches) { $testedTree } else { $otherTree }
            $parsed = [pscustomobject]@{ TestedTree = $(if ($certificateMatches) { $testedTree } else { $otherTree }) }
            $certificateTreeMatches = $true
            foreach ($binding in $bindings) { . ([scriptblock]::Create($binding.Extent.Text)) }
            $evidence = @([pscustomobject]@{
                Scope = [CiValidationReuseScope]::Validation; Event = 'pull_request'
                RunId = 1; CreatedAt = '2026-01-01T00:00:00Z'
                TreeMatches = $treeMatches; CertificateTreeMatches = $certificateTreeMatches
                RequiresValidation = $requiresRuntime
                HasAnalysis = $true; HasCoverage = $true; HasLedger = $true
            })
            $best = Select-BestCiEvidence $evidence
            . ([scriptblock]::Create($delta[0].Extent.Text))
            $expectedExact = $certificateMatches -and $landedMatches
            $expectedDelta = $certificateMatches -and -not $landedMatches
            if (($null -ne $best) -ne $expectedExact -or ($deltaCandidates.Count -gt 0) -ne $expectedDelta) {
                [void] $failures.Add("certificate=$certificateMatches landed=$landedMatches runtime=$requiresRuntime exact=$($null -ne $best) delta=$($deltaCandidates.Count)")
            }
        }
    }
}
if ($failures.Count -gt 0) { throw "Certificate eligibility failures: $($failures -join '; ')" }
Write-Host "Certificate evidence review: $cases exact/delta production eligibility cases passed."
