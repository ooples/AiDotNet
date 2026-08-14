# Copyright (c) AiDotNet. All rights reserved.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Runner,

    [Parameter(Mandatory = $true)]
    [string] $Inventory,

    [Parameter(Mandatory = $true)]
    [string] $OutputDirectory,

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 2147483647)]
    [int] $ShardIndex,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int] $ShardCount,

    [ValidateRange(1, 86400)]
    [int] $TimeoutSeconds = 180,

    [ValidateRange(1, 600)]
    [int] $CleanupGraceSeconds = 30
)

$ErrorActionPreference = 'Stop'

if ($ShardIndex -ge $ShardCount) {
    throw "ShardIndex ($ShardIndex) must be less than ShardCount ($ShardCount)."
}

$runnerPath = (Resolve-Path -LiteralPath $Runner).Path
$inventoryPath = (Resolve-Path -LiteralPath $Inventory).Path
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$runRoot = Join-Path $outputPath 'runs'
New-Item -ItemType Directory -Force -Path $outputPath, $runRoot | Out-Null

function Get-StablePerformanceHash([string] $Value) {
    [uint32] $hash = 2166136261
    foreach ($character in $Value.ToCharArray()) {
        $hash = [uint32]($hash -bxor [uint32][char]$character)
        $hash = [uint32](([uint64]$hash * 16777619L) -band 0xffffffffL)
    }
    return $hash
}

function Get-SafeFileName([string] $Value) {
    $safe = $Value
    foreach ($invalid in [IO.Path]::GetInvalidFileNameChars()) {
        $safe = $safe.Replace([string]$invalid, '_')
    }
    return $safe
}

function Write-AtomicJson([string] $Destination, [object] $Value) {
    $temporary = "$Destination.$([Guid]::NewGuid().ToString('N')).tmp"
    $Value | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Destination -Force
}

$tests = @(Get-Content -LiteralPath $inventoryPath |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -match '\.ModelPerformanceCensus$' })
if ($tests.Count -eq 0) {
    throw "No ModelPerformanceCensus fixtures were found in $inventoryPath."
}

$assigned = @($tests | Where-Object {
    $fixture = $_ -replace '\.ModelPerformanceCensus$', ''
    ((Get-StablePerformanceHash $fixture) % [uint32]$ShardCount) -eq [uint32]$ShardIndex
})
Write-Host "Shard $ShardIndex/$ShardCount owns $($assigned.Count) of $($tests.Count) model fixtures."

foreach ($testName in $assigned) {
    $fixture = $testName -replace '\.ModelPerformanceCensus$', ''
    $safeName = Get-SafeFileName $fixture
    $fixtureRunDirectory = Join-Path $runRoot $safeName
    $recordPath = Join-Path $outputPath "$safeName.json"
    $progressPath = Join-Path $outputPath "$safeName.progress.jsonl"
    $logPath = Join-Path $fixtureRunDirectory 'console.log'
    New-Item -ItemType Directory -Force -Path $fixtureRunDirectory | Out-Null
    Remove-Item -LiteralPath $recordPath, $progressPath -Force -ErrorAction SilentlyContinue

    $arguments = @($runnerPath, $fixture)
    $start = [DateTimeOffset]::UtcNow
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = [Diagnostics.ProcessStartInfo]::new()
    $process.StartInfo.FileName = 'dotnet'
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    foreach ($argument in $arguments) {
        [void]$process.StartInfo.ArgumentList.Add($argument)
    }
    $process.StartInfo.Environment['AIDOTNET_MODEL_PERF_DIR'] = $outputPath
    # External partitioning has already selected exactly one fixture. Make the in-test
    # guard accept it without coupling this child process to the matrix shard number.
    $process.StartInfo.Environment['AIDOTNET_MODEL_PERF_SHARD_INDEX'] = '0'
    $process.StartInfo.Environment['AIDOTNET_MODEL_PERF_SHARD_COUNT'] = '1'

    Write-Host "[$([DateTimeOffset]::UtcNow.ToString('u'))] START $fixture"
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    # Poll rather than blocking for the entire ceiling so both successful and timed-out
    # fixtures retain peak-memory and CPU-utilization evidence. A timeout without these
    # measurements only says that a model was slow; it cannot distinguish compute saturation
    # from runaway materialization / paging, which are very different engine regressions.
    [long]$peakWorkingSetBytes = 0
    [long]$peakPrivateMemoryBytes = 0
    [double]$runnerCpuMs = 0
    $completed = $false
    while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($process.WaitForExit(250)) {
            $completed = $true
            break
        }
        try {
            $process.Refresh()
            $peakWorkingSetBytes = [Math]::Max($peakWorkingSetBytes, $process.WorkingSet64)
            $peakPrivateMemoryBytes = [Math]::Max($peakPrivateMemoryBytes, $process.PrivateMemorySize64)
            $runnerCpuMs = [Math]::Max($runnerCpuMs, $process.TotalProcessorTime.TotalMilliseconds)
        } catch {
            # The process can exit between WaitForExit and Refresh. The next loop observes exit.
        }
    }

    # The workload writes its result atomically before fixture/network disposal. Treat that durable
    # record as the measurement boundary and allow a short, separately-bounded teardown grace. This
    # prevents a model that completed every measured phase inside the ceiling from being mislabeled
    # as a compute timeout merely because a compacting LOH collection takes a few more seconds. A
    # fixture that has not produced its record at the deadline receives no grace and is killed below.
    $workloadCompletedAtDeadline = Test-Path -LiteralPath $recordPath
    if (-not $completed -and $workloadCompletedAtDeadline) {
        $cleanupDeadlineMs = $stopwatch.Elapsed.TotalMilliseconds + ($CleanupGraceSeconds * 1000.0)
        while ($stopwatch.Elapsed.TotalMilliseconds -lt $cleanupDeadlineMs) {
            if ($process.WaitForExit(250)) {
                $completed = $true
                break
            }
            try {
                $process.Refresh()
                $peakWorkingSetBytes = [Math]::Max($peakWorkingSetBytes, $process.WorkingSet64)
                $peakPrivateMemoryBytes = [Math]::Max($peakPrivateMemoryBytes, $process.PrivateMemorySize64)
                $runnerCpuMs = [Math]::Max($runnerCpuMs, $process.TotalProcessorTime.TotalMilliseconds)
            } catch {
                # The process can exit between WaitForExit and Refresh. The next loop observes exit.
            }
        }
    }
    $status = 'ok'
    $exitCode = $null
    if (-not $completed) {
        $status = if ($workloadCompletedAtDeadline) { 'cleanup-timeout' } else { 'timeout' }
        try { $process.Kill($true) } catch { Write-Warning "Could not kill $fixture process tree: $_" }
        $process.WaitForExit()
    } else {
        $exitCode = $process.ExitCode
    }
    try {
        $process.Refresh()
        $peakWorkingSetBytes = [Math]::Max($peakWorkingSetBytes, $process.PeakWorkingSet64)
        $peakPrivateMemoryBytes = [Math]::Max($peakPrivateMemoryBytes, $process.PrivateMemorySize64)
        $runnerCpuMs = [Math]::Max($runnerCpuMs, $process.TotalProcessorTime.TotalMilliseconds)
    } catch {
        # The polling samples above remain durable even if final process metrics are unavailable.
    }
    $standardOutput = $stdout.GetAwaiter().GetResult()
    $standardError = $stderr.GetAwaiter().GetResult()
    $stopwatch.Stop()
    $runnerCpuToWallRatio = if ($stopwatch.Elapsed.TotalMilliseconds -gt 0) {
        $runnerCpuMs / $stopwatch.Elapsed.TotalMilliseconds
    } else { 0.0 }
    @($standardOutput, $standardError) | Set-Content -LiteralPath $logPath -Encoding utf8

    if ($status -eq 'ok' -and $exitCode -ne 0) {
        $status = if ($exitCode -eq 1) { 'failed' } else { 'crashed' }
    }

    if ($status -ne 'ok' -or -not (Test-Path -LiteralPath $recordPath)) {
        if ($status -eq 'ok') { $status = 'crashed' }
        $phase = 'process-start'
        if (Test-Path -LiteralPath $progressPath) {
            $lastProgress = Get-Content -LiteralPath $progressPath | Select-Object -Last 1
            if (-not [string]::IsNullOrWhiteSpace($lastProgress)) {
                try { $phase = ($lastProgress | ConvertFrom-Json).phase } catch { $phase = 'invalid-progress-record' }
            }
        }
        $combined = ($standardOutput + [Environment]::NewLine + $standardError).Trim()
        if ($combined.Length -gt 4000) { $combined = $combined.Substring($combined.Length - 4000) }
        $failure = [ordered]@{
            schemaVersion = 1
            status = $status
            fixture = $fixture
            model = ''
            precision = ''
            framework = [Runtime.InteropServices.RuntimeInformation]::FrameworkDescription
            frameworkMajor = [Environment]::Version.Major
            os = [Runtime.InteropServices.RuntimeInformation]::OSDescription
            osPlatform = if ($IsLinux) { 'Linux' } elseif ($IsMacOS) { 'macOS' } else { 'Windows' }
            processArchitecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
            processorCount = [Environment]::ProcessorCount
            runId = $env:GITHUB_RUN_ID
            commit = $env:GITHUB_SHA
            shardIndex = $ShardIndex
            shardCount = $ShardCount
            measuredUtc = [DateTimeOffset]::UtcNow
            startedUtc = $start
            elapsedMs = $stopwatch.Elapsed.TotalMilliseconds
            timeoutSeconds = $TimeoutSeconds
            cleanupGraceSeconds = $CleanupGraceSeconds
            workloadCompletedAtDeadline = $workloadCompletedAtDeadline
            peakWorkingSetBytes = $peakWorkingSetBytes
            peakPrivateMemoryBytes = $peakPrivateMemoryBytes
            runnerCpuMs = $runnerCpuMs
            runnerCpuToWallRatio = $runnerCpuToWallRatio
            exitCode = $exitCode
            phase = $phase
            error = $combined
        }
        Write-AtomicJson $recordPath $failure
    } else {
        # The in-process sample owns operation/phase timings. Add the supervising process's
        # resource envelope atomically so successful fixtures and timeout records have the same
        # comparable memory/CPU fields without coupling the test assembly to OS polling APIs.
        $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
        $record | Add-Member -NotePropertyName runnerElapsedMs -NotePropertyValue $stopwatch.Elapsed.TotalMilliseconds -Force
        $record | Add-Member -NotePropertyName timeoutSeconds -NotePropertyValue $TimeoutSeconds -Force
        $record | Add-Member -NotePropertyName cleanupGraceSeconds -NotePropertyValue $CleanupGraceSeconds -Force
        $record | Add-Member -NotePropertyName workloadCompletedAtDeadline -NotePropertyValue $workloadCompletedAtDeadline -Force
        $record | Add-Member -NotePropertyName peakWorkingSetBytes -NotePropertyValue $peakWorkingSetBytes -Force
        $record | Add-Member -NotePropertyName peakPrivateMemoryBytes -NotePropertyValue $peakPrivateMemoryBytes -Force
        $record | Add-Member -NotePropertyName runnerCpuMs -NotePropertyValue $runnerCpuMs -Force
        $record | Add-Member -NotePropertyName runnerCpuToWallRatio -NotePropertyValue $runnerCpuToWallRatio -Force
        Write-AtomicJson $recordPath $record
    }

    Write-Host "[$([DateTimeOffset]::UtcNow.ToString('u'))] $($status.ToUpperInvariant()) $fixture ($([Math]::Round($stopwatch.Elapsed.TotalSeconds, 1)) s)"
}

$records = @(Get-ChildItem -LiteralPath $outputPath -Filter '*.json' -File)
$statusCounts = @{}
foreach ($record in $records) {
    $status = (Get-Content -LiteralPath $record.FullName -Raw | ConvertFrom-Json).status
    if (-not $statusCounts.ContainsKey($status)) { $statusCounts[$status] = 0 }
    $statusCounts[$status]++
}
Write-Host "Shard $ShardIndex produced $($records.Count)/$($assigned.Count) durable record(s): $($statusCounts | ConvertTo-Json -Compress)"
if ($records.Count -ne $assigned.Count) {
    throw "Shard $ShardIndex produced $($records.Count) records for $($assigned.Count) assigned fixtures."
}
