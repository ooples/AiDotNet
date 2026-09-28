# Copyright (c) AiDotNet. All rights reserved.

<#
.SYNOPSIS
    Puts every project a host references back at its own output path, from outputs the Build job's
    artifact already carries, so the host can be built against that artifact without recompiling
    the solution.

.DESCRIPTION
    The census host references the test project. Built with BuildProjectReferences=false, MSBuild
    compiles only the host, but still reads every project in the reference closure from its
    TargetPath (for example src/AiDotNet.Storage.Redis/bin/Release/net10.0/...). The Build
    artifact carries the test output - which holds a copy-local copy of every one of those
    assemblies - but not the individual project outputs, and shipping them would add hundreds of
    megabytes to what every test shard downloads.

    MSBuild itself is asked for the paths (ResolveProjectReferences with the same properties the
    host build uses), so the list is the one the compiler will read, not a reconstruction. Each
    missing path is filled from the first given output holding that file; one none of them holds
    fails. The samples use the same mechanism for the Serving sample, whose project output (800 MB)
    is not in the artifact but whose copy-local copy in the Serving test output is.

.PARAMETER Project
    The host project that will be built with BuildProjectReferences=false.

.PARAMETER SourceOutput
    Output directories from the Build artifact, searched in order.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Project,
    [Parameter(Mandatory)] [string[]] $SourceOutput,
    [string] $Configuration = 'Release',
    [string] $Framework = 'net10.0'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$output = @(& dotnet msbuild $Project -nologo -t:ResolveProjectReferences "-p:Configuration=$Configuration" `
    "-p:TargetFramework=$Framework" -p:BuildProjectReferences=false -p:ProduceReferenceAssembly=false `
    -getItem:_ResolvedProjectReferencePaths)
if ($LASTEXITCODE -ne 0) { $output | Write-Host; throw "Resolving the project references of $Project failed." }
# Restore warnings precede the JSON document on stdout.
$text = $output -join "`n"
$start = $text.IndexOf("`n{")
if ($text.StartsWith('{')) { $start = 0 } elseif ($start -ge 0) { $start++ }
if ($start -lt 0) { throw "MSBuild returned no item document for $Project." }
$items = @(($text.Substring($start) | ConvertFrom-Json).Items._ResolvedProjectReferencePaths)
if ($items.Count -eq 0) { throw "$Project resolves no project references." }

function Find-OutputFile([string] $Name) {
    foreach ($directory in $SourceOutput) {
        $candidate = Join-Path $directory $Name
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

$restored = 0
foreach ($item in $items) {
    $path = [string] $item.Identity
    $directory = [IO.Path]::GetDirectoryName($path)
    $name = [IO.Path]::GetFileName($path)
    if (-not (Test-Path -LiteralPath $path)) {
        $source = Find-OutputFile $name
        if (-not $source) {
            throw "No given output has a copy of $name, which $Project reads from $path."
        }
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
        Copy-Item -LiteralPath $source -Destination $path
        $restored++
    }
    # A referenced executable project also hands its runtime and dependency manifests to every
    # referencing output; the source outputs received them the same way.
    $stem = [IO.Path]::GetFileNameWithoutExtension($name)
    foreach ($companion in @("$stem.runtimeconfig.json", "$stem.deps.json")) {
        $source = Find-OutputFile $companion
        $into = Join-Path $directory $companion
        if ($source -and -not (Test-Path -LiteralPath $into)) {
            Copy-Item -LiteralPath $source -Destination $into
        }
    }
}
Write-Host "Restored $restored of $($items.Count) project reference output(s) for $Project."
