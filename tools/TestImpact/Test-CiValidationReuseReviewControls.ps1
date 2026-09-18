<# Prove the emission fixture rejects fallback-only output when production decline notices vanish. #>
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'ReviewFixtureCleanup.ps1')
$resolver = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'Resolve-CiValidationReuse.ps1') -Raw
$pattern = '(?m)^\s*Write-Host "::notice::delta reuse declined:[^\r\n]*\r?$'
if ([regex]::Matches($resolver, $pattern).Count -ne 2) { throw 'Expected both production decline notices.' }
$mutant = [regex]::Replace($resolver, $pattern, '')
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('aidotnet-emission-review-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixture | Out-Null
try {
    $path = Join-Path $fixture 'resolver-without-decline-notices.ps1'
    Set-Content -LiteralPath $path -Value $mutant -Encoding utf8
    $output = @(& pwsh -NoProfile -File (Join-Path $PSScriptRoot 'Test-CiValidationReuseReview.ps1') -ResolverPath $path 2>&1)
    if ($LASTEXITCODE -eq 0) { throw 'Fallback-only output falsely passed the artifact emission review.' }
    $failures = @($output | Where-Object { [string] $_ -match 'production decline path was not observed:' })
    if ($failures.Count -ne 8) { throw "Expected eight missing-decline failures, got $($failures.Count): $($output -join [Environment]::NewLine)" }
}
finally { Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-emission-review-' }
Write-Host 'Artifact review guard: all eight fallback-only decline cases rejected.'
