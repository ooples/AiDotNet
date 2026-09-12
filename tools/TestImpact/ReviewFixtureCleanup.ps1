<# Shared cleanup boundary for the local test-impact review fixtures. #>
Set-StrictMode -Version Latest

function Remove-ReviewFixtureDirectory {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $LiteralPath,
        [Parameter(Mandatory)] [ValidatePattern('^aidotnet-[a-z0-9-]+-$')] [string] $ExpectedLeafPrefix,
        [switch] $ThrowOnUnsafePath
    )

    $resolved = [IO.Path]::GetFullPath($LiteralPath)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $separators = [char[]] @([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $prefix = $tempRoot.TrimEnd($separators) + [IO.Path]::DirectorySeparatorChar
    $comparison = if ([IO.Path]::DirectorySeparatorChar -eq '\') { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    $safe = $resolved.StartsWith($prefix, $comparison) -and
        [IO.Path]::GetFileName($resolved).StartsWith($ExpectedLeafPrefix, [StringComparison]::Ordinal)
    # Do not traverse a linked ancestor on the way to an otherwise correctly named fixture.
    $ancestor = $resolved
    while ($safe -and $ancestor.StartsWith($prefix, $comparison)) {
        if (Test-Path -LiteralPath $ancestor) {
            $item = Get-Item -LiteralPath $ancestor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { $safe = $false }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    if (-not $safe) {
        $message = "refusing to remove unexpected fixture path '$resolved'"
        if ($ThrowOnUnsafePath) { throw $message }
        Write-Warning $message
        return
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction SilentlyContinue
}
