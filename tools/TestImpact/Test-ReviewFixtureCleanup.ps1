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
try {
    New-Item -ItemType Directory -Path $fixture | Out-Null
    'owned fixture' | Set-Content -LiteralPath (Join-Path $fixture 'marker.txt')
    Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-cleanup-review-' -ThrowOnUnsafePath
    if (Test-Path -LiteralPath $fixture) { throw 'The owned fixture was not removed.' }
}
finally { Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-cleanup-review-' }

$linkFixture = Join-Path $tempRoot ('aidotnet-cleanup-review-link-' + [guid]::NewGuid().ToString('N'))
$sentinel = Join-Path $tempRoot ('aidotnet-cleanup-review-sentinel-' + [guid]::NewGuid().ToString('N'))
$link = Join-Path $linkFixture 'linked-ancestor'
$targetLeaf = 'aidotnet-cleanup-review-child'
$sentinelChild = Join-Path $sentinel $targetLeaf
$marker = Join-Path $sentinelChild 'marker.txt'
$markerText = 'a linked ancestor must not allow deleting this sentinel'
try {
    New-Item -ItemType Directory -Path $linkFixture, $sentinelChild | Out-Null
    $markerText | Set-Content -LiteralPath $marker
    # Junctions need no symlink privilege on Windows; Unix uses a directory symlink.
    $linkType = if ([IO.Path]::DirectorySeparatorChar -eq '\') { 'Junction' } else { 'SymbolicLink' }
    New-Item -ItemType $linkType -Path $link -Target $sentinel -ErrorAction Stop | Out-Null
    if (((Get-Item -LiteralPath $link -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
        throw 'The fixture did not create a real reparse-point ancestor.'
    }
    $rejected = $false
    try {
        Remove-ReviewFixtureDirectory -LiteralPath (Join-Path $link $targetLeaf) `
            -ExpectedLeafPrefix 'aidotnet-cleanup-review-' -ThrowOnUnsafePath
    }
    catch {
        if ($_.Exception.Message -notlike 'refusing to remove unexpected fixture path*') { throw }
        $rejected = $true
    }
    if (-not $rejected) { throw 'Cleanup accepted a reparse-point ancestor.' }
    if (-not (Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker -Raw).TrimEnd() -cne $markerText) {
        throw 'Cleanup changed or removed the linked sentinel.'
    }
}
finally {
    # Unlink only the exact link (never recurse through it), then clean the two
    # validated, independently owned temporary roots through the production guard.
    if (Test-Path -LiteralPath $link) {
        $linkItem = Get-Item -LiteralPath $link -Force
        if (($linkItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            throw 'Refusing to unlink an unexpected non-reparse fixture.'
        }
        Remove-Item -LiteralPath $link -Force -ErrorAction Stop
    }
    Remove-ReviewFixtureDirectory -LiteralPath $linkFixture -ExpectedLeafPrefix 'aidotnet-cleanup-review-' -ThrowOnUnsafePath
    Remove-ReviewFixtureDirectory -LiteralPath $sentinel -ExpectedLeafPrefix 'aidotnet-cleanup-review-' -ThrowOnUnsafePath
}
Write-Host 'Review fixture cleanup: four unsafe paths and a real linked ancestor rejected; sentinel preserved; owned fixtures removed.'
