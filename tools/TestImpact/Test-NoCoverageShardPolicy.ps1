<# Exercise the production no-coverage parser/guard with a growing and duplicated manifest. #>
[CmdletBinding()]
param([string] $WorkflowPath = "$PSScriptRoot/../../.github/workflows/sonarcloud.yml")
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$workflow = Get-Content -LiteralPath $WorkflowPath -Raw
$match = [regex]::Match($workflow, '(?ms)^            \$nn = .*?(?=^            \$prevMapOk =)')
if (-not $match.Success) { throw 'Expected exactly one production no-coverage parser block.' }
$policy = [scriptblock]::Create($match.Value)
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$manifest = Get-Content -LiteralPath (Join-Path $repositoryRoot '.github/test-shards.yml') -Raw
$declared = @([regex]::Matches($manifest, '(?m)^  - name: ([^\r\n]+)') | ForEach-Object {
    [pscustomobject]@{ name = $_.Groups[1].Value; heavy = $false }
})
$extra = @(1..220 | ForEach-Object { [pscustomobject]@{ name = "Review synthetic heavy $_"; heavy = $true } })
Push-Location $repositoryRoot
try {
    $all = @($declared) + @($extra)
    . $policy
    if ($nn.Count -le 200 -or @($extra | Where-Object { -not $nn.Contains($_.name) }).Count -ne 0) {
        throw 'A valid growing no-coverage list was cleared by a fixed-size ceiling.'
    }
    $expectedCount = $nn.Count
    $all = @($declared) + @($extra) + @($extra)
    . $policy
    if ($nn.Count -ne $expectedCount) { throw 'Duplicate manifest names inflated the no-coverage policy.' }
    $all = @([pscustomobject]@{ name = 'Only declared shard'; heavy = $false })
    . $policy
    if ($nn.Count -ne 0) { throw 'An oversized invalid parse did not fail closed.' }
}
finally { Pop-Location }
Write-Host 'No-coverage policy: valid list above 200 retained, duplicates collapsed, oversized parse rejected.'
