[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'ReviewFixtureCleanup.ps1')

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ('aidotnet-cleanup-review-' + [guid]::NewGuid().ToString('N'))
$unsafePaths = @(
    $tempRoot,
    (Join-Path ($tempRoot.TrimEnd([char[]] @('/', '\')) + '-sibling') 'aidotnet-cleanup-review-forged'),
    (Join-Path $tempRoot 'wrong-prefix'),
    (Join-Path $tempRoot '../aidotnet-cleanup-review-escaped')
)
foreach ($unsafePath in $unsafePaths) {
    $rejected = $false
    try { Remove-ReviewFixtureDirectory -LiteralPath $unsafePath -ExpectedLeafPrefix 'aidotnet-cleanup-review-' -ThrowOnUnsafePath }
    catch {
        if ($_.Exception.Message -notlike 'refusing to remove unexpected fixture path*') { throw }
        $rejected = $true
    }
    if (-not $rejected) { throw "Unsafe fixture cleanup was accepted: $unsafePath" }
}
New-Item -ItemType Directory -Path $fixture | Out-Null
try {
    'owned fixture' | Set-Content -LiteralPath (Join-Path $fixture 'marker.txt')
    Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-cleanup-review-' -ThrowOnUnsafePath
    if (Test-Path -LiteralPath $fixture) { throw 'The owned fixture was not removed.' }
}
finally { Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-cleanup-review-' }
Write-Host 'Review fixture cleanup: four unsafe paths rejected; owned fixture removed.'
