<#
.SYNOPSIS
    Writes a shard-specific copy of coverlet.runsettings that also instruments the assemblies in one more
    directory.
.DESCRIPTION
    The model-inventory sweeps and conformance windows run each model in a child worker process
    (tests/AiDotNet.ParameterSweepWorker), which loads its OWN copy of AiDotNet.dll from its own output
    directory. Coverlet instruments only the test assembly's directory, so the model code those shards exist
    to test executes uninstrumented and never reaches their coverage digest. Coverlet's IncludeDirectory
    names extra directories whose assemblies it instruments; a child process running an instrumented module
    records its hits when it exits, and the collector reads them with the rest.

    Generated per shard rather than added to coverlet.runsettings because only these shards run the worker,
    and instrumenting a second copy of AiDotNet costs every other shard the same again in finalization time
    (the base file records that finalization already scales with the instrumented surface). The directory is
    written as an absolute path: coverlet resolves a relative one against a working directory that differs
    between the test host and the data collector.
.PARAMETER Base
    The runsettings to copy (coverlet.runsettings).
.PARAMETER IncludeDirectory
    Absolute path of the directory whose assemblies should also be instrumented. Must exist.
.PARAMETER OutFile
    Where to write the shard's runsettings.
.PARAMETER SelfTest
    Runs the checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Write')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Write')] [string] $Base,
    [Parameter(Mandatory, ParameterSetName = 'Write')] [string] $IncludeDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Write')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function New-RunSettingsXml {
    param([Parameter(Mandatory)] [string] $BaseText, [Parameter(Mandatory)] [string] $Directory)

    if (-not [System.IO.Path]::IsPathRooted($Directory)) {
        throw "IncludeDirectory must be absolute, got '$Directory'"
    }
    [xml] $document = $BaseText
    $configuration = $document.SelectSingleNode(
        "/RunSettings/DataCollectionRunSettings/DataCollectors/DataCollector[@friendlyName='XPlat Code Coverage']/Configuration")
    if ($null -eq $configuration) {
        throw 'the base runsettings has no XPlat Code Coverage configuration to extend'
    }
    if ($null -ne $configuration.SelectSingleNode('IncludeDirectory')) {
        throw 'the base runsettings already declares IncludeDirectory; merge it deliberately instead'
    }
    $element = $document.CreateElement('IncludeDirectory')
    $element.InnerText = $Directory
    [void] $configuration.AppendChild($element)
    return $document
}

if ($SelfTest) {
    $base = @'
<?xml version="1.0" encoding="utf-8" ?>
<RunSettings><DataCollectionRunSettings><DataCollectors>
  <DataCollector friendlyName="XPlat Code Coverage"><Configuration><Format>opencover</Format></Configuration></DataCollector>
</DataCollectors></DataCollectionRunSettings></RunSettings>
'@
    $absolute = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $written = New-RunSettingsXml -BaseText $base -Directory $absolute
    $node = $written.SelectSingleNode('//DataCollector/Configuration/IncludeDirectory')
    if ($null -eq $node -or $node.InnerText -ne $absolute) { throw 'self-test: IncludeDirectory was not written' }
    if ($written.SelectSingleNode('//Configuration/Format').InnerText -ne 'opencover') { throw 'self-test: the base settings were lost' }

    $rejected = 0
    foreach ($case in @(
            @{ Text = $base; Directory = 'relative/dir' },
            @{ Text = '<RunSettings />'; Directory = $absolute },
            @{ Text = $base.Replace('<Format>', "<IncludeDirectory>$absolute</IncludeDirectory><Format>"); Directory = $absolute })) {
        try { New-RunSettingsXml -BaseText $case.Text -Directory $case.Directory | Out-Null } catch { $rejected++ }
    }
    if ($rejected -ne 3) { throw "self-test: expected 3 rejections, got $rejected" }
    Write-Host 'New-CoverageRunSettings self-test passed.'
    exit 0
}

if (-not (Test-Path -LiteralPath $IncludeDirectory -PathType Container)) {
    throw "IncludeDirectory '$IncludeDirectory' does not exist; the worker it should instrument was not built into this artifact"
}
$document = New-RunSettingsXml -BaseText (Get-Content -LiteralPath $Base -Raw) -Directory (Resolve-Path -LiteralPath $IncludeDirectory).Path
$document.Save($OutFile)
Write-Host "coverage runsettings: $OutFile also instruments $IncludeDirectory"
exit 0
