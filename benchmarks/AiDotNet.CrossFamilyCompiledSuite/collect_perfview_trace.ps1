<#
.SYNOPSIS
    Turnkey PerfView / dotnet-trace profiler wrapper for AiDotNet.CrossFamilyCompiledSuite.

.DESCRIPTION
    Captures CPU sampling, .NET GC / AllocationTick / LOH, Exception, and Contention events
    during cross-family Wave 1 / Wave 2 execution so C# dispatch, GC allocation, and PCIe sync
    hotspots can be audited empirically rather than guessed.

    Enforces the hard GPU safety gate: -GpuIndex 0 is strictly rejected.
#>
[CmdletBinding()]
param(
    [switch]$CpuDryRun,
    [int]$GpuIndex = 1,
    [ValidateSet("auto", "direct-ptx", "eager", "uncached-eager")]
    [string]$Mode = "auto",
    [ValidateSet("fp32", "tf32", "bf16", "fp16")]
    [string]$Precision = "fp32",
    [ValidateSet("all", "1", "2", "wave1", "wave2")]
    [string]$Wave = "1",
    [string]$Filter = "",
    [int]$Seed = 42,
    [string]$LocalTensorsRoot = "",
    [string]$OutputDir = "",
    [switch]$SkipBuild,
    [switch]$UsePerfViewExe,
    [string]$PerfViewExePath = "PerfView.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SuiteDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CsprojPath = Join-Path $SuiteDir "AiDotNet.CrossFamilyCompiledSuite.csproj"

if ($CpuDryRun) {
    $Device = "cpu"
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:AIDOTNET_DISABLE_GPU = "1"
}
else {
    if ($GpuIndex -eq 0) {
        throw "Safety gate violation: GPU 0 is strictly off-limits. Specify -GpuIndex 1 or -CpuDryRun."
    }
    $Device = "cuda"
    $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    $env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
    Remove-Item Env:\AIDOTNET_DISABLE_GPU -ErrorAction SilentlyContinue
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
    $OutputDir = Join-Path $SuiteDir "traces\trace_${Device}_${Mode}_${Stamp}"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $SkipBuild) {
    $BuildArgs = @("build", $CsprojPath, "-c", "Release", "-p:GeneratePackageOnBuild=false", "-v", "minimal")
    if (-not [string]::IsNullOrWhiteSpace($LocalTensorsRoot)) {
        $BuildArgs += "-p:UseLocalTensors=true"
        $BuildArgs += "-p:LocalTensorsRoot=$LocalTensorsRoot"
    }

    Write-Host "[PerfViewWrapper] Building benchmark suite..."
    & dotnet @BuildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet build failed with exit code $LASTEXITCODE"
    }
}

$DllFile = Get-ChildItem -Path (Join-Path $SuiteDir "bin") -Recurse -Filter "AiDotNet.CrossFamilyCompiledSuite.dll" |
    Where-Object { $_.FullName -match '(?i)[\\/]Release[\\/]' } |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

if ($null -eq $DllFile) {
    throw "Could not find built AiDotNet.CrossFamilyCompiledSuite.dll."
}

$ResultJson = Join-Path $OutputDir "benchmark_metrics.json"
$RunnerArgs = @(
    $DllFile.FullName,
    "--device", $Device,
    "--mode", $Mode,
    "--precision", $Precision,
    "--wave", $Wave,
    "--seed", "$Seed",
    "--output", $ResultJson
)
if ($CpuDryRun) {
    $RunnerArgs += "--smoke"
}
if (-not [string]::IsNullOrWhiteSpace($Filter)) {
    $RunnerArgs += "--filter"
    $RunnerArgs += $Filter
}

if ($UsePerfViewExe) {
    $EtlPath = Join-Path $OutputDir "aidotnet_${Device}_${Mode}.etl"
    Write-Host "[PerfViewWrapper] Collecting ETW trace via $PerfViewExePath -> $EtlPath"
    & $PerfViewExePath /NoGui /AcceptEULA /DataFile:$EtlPath /Merge:true /Zip:true /GCCollectOnly:false run dotnet @RunnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PerfView collection exited with code $LASTEXITCODE"
    }
}
else {
    $DotnetTrace = Get-Command "dotnet-trace" -ErrorAction SilentlyContinue
    if ($null -ne $DotnetTrace) {
        $NetTracePath = Join-Path $OutputDir "aidotnet_${Device}_${Mode}.nettrace"
        Write-Host "[PerfViewWrapper] Collecting CPU + GC allocation + contention + exception EventPipe trace -> $NetTracePath"
        & dotnet-trace collect --providers "Microsoft-DotNETCore-SampleProfiler,Microsoft-Windows-DotNETRuntime:0x800008C01D:5" -o $NetTracePath -- dotnet @RunnerArgs
        if ($LASTEXITCODE -ne 0) {
            throw "dotnet-trace collection exited with code $LASTEXITCODE"
        }
    }
    else {
        Write-Warning "dotnet-trace tool is not installed in PATH; running benchmark suite directly and recording managed allocation metrics."
        & dotnet @RunnerArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark execution exited with code $LASTEXITCODE"
        }
    }
}

Write-Host "[PerfViewWrapper] Complete. Artifacts written under: $OutputDir"
