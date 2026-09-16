<#
.SYNOPSIS
    Makes an isolated worker load the exact binary instrumented by the parent collector.
.DESCRIPTION
    Coverlet deduplicates modules by filename. IncludeDirectory does not instrument a
    second copy of AiDotNet.dll. A hard link shares the instrumented bytes and the hit
    counter destination while preserving the worker's separate process and heap limit.
    Only identical DLLs inside explicit build-output directories may be connected.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $ParentDirectory,
    [Parameter(Mandatory)] [string] $WorkerDirectory,
    [string] $AssemblyName = 'AiDotNet.dll'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([IO.Path]::GetFileName($AssemblyName) -cne $AssemblyName -or
    -not $AssemblyName.EndsWith('.dll', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'AssemblyName must be a DLL basename, not a path.'
}
$parent = (Get-Item -LiteralPath $ParentDirectory -ErrorAction Stop).FullName
$worker = (Get-Item -LiteralPath $WorkerDirectory -ErrorAction Stop).FullName
foreach ($directory in @($parent, $worker)) {
    if ($directory.Replace('\', '/') -notmatch '/bin/') {
        throw 'Worker coverage may only connect build outputs under bin directories.'
    }
}
$source = Join-Path $parent $AssemblyName
$destination = Join-Path $worker $AssemblyName
if ($source.Equals($destination, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Parent and worker must have distinct output paths.'
}
foreach ($path in @($source, $destination)) {
    $item = Get-Item -LiteralPath $path -ErrorAction Stop
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected a regular build-output DLL: $path"
    }
}
if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -cne
    (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) {
    throw 'Parent and worker binaries differ; refusing to substitute a different build.'
}

# Create before replacing: unsupported filesystems or cross-volume links leave the
# existing worker binary intact. These paths are individual, validated build outputs.
$staged = "$destination.coverage-$([Guid]::NewGuid().ToString('N'))"
try {
    New-Item -ItemType HardLink -Path $staged -Target $source -ErrorAction Stop | Out-Null
    Move-Item -LiteralPath $staged -Destination $destination -Force -ErrorAction Stop
}
finally {
    if (Test-Path -LiteralPath $staged -PathType Leaf) {
        Remove-Item -LiteralPath $staged -Force
    }
}
Write-Host "Worker coverage linked: $destination -> $source"
exit 0
