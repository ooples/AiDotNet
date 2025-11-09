################################################################################
# AiDotNet Distributed Training Launcher (PowerShell)
#
# This script launches distributed training using MPI across multiple processes.
#
# For Beginners:
# MPI (Message Passing Interface) is a standard for running programs across
# multiple computers or processors. Think of it like a coordinator that starts
# your program on multiple machines at once and helps them communicate.
#
# Usage:
#   .\launch-distributed-training.ps1 -NumProcesses <num> -Program <path> [additional args...]
#
# Examples:
#   # Run on 4 GPUs locally
#   .\launch-distributed-training.ps1 -NumProcesses 4 -Program ".\MyTrainingApp.exe"
#
#   # Run on 8 GPUs with additional arguments
#   .\launch-distributed-training.ps1 -NumProcesses 8 -Program ".\MyTrainingApp.exe" --epochs 100 --lr 0.001
#
#   # Run with config file containing spaces in path
#   .\launch-distributed-training.ps1 -NumProcesses 8 -Program ".\MyTrainingApp.exe" --config "My Config.json"
#
#   # Run across 2 machines with 4 GPUs each
#   .\launch-distributed-training.ps1 -NumProcesses 8 -Program ".\MyTrainingApp.exe" -Hosts "machine1,machine2"
################################################################################

param(
    [Parameter(Mandatory=$true, HelpMessage="Number of processes to spawn (typically equals number of GPUs)")]
    [int]$NumProcesses,

    [Parameter(Mandatory=$true, HelpMessage="Path to your training program executable")]
    [string]$Program,

    [Parameter(Mandatory=$false, HelpMessage="Comma-separated list of host machines")]
    [string]$Hosts = "",

    [Parameter(
        Mandatory = $false,
        HelpMessage = "Additional arguments to pass to your program",
        ValueFromRemainingArguments = $true)]
    [string[]]$ProgramArgs = @()
)

# Display header
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "AiDotNet Distributed Training Launcher" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Display configuration
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Number of processes: $NumProcesses"
Write-Host "  Program: $Program"
if ($ProgramArgs.Count -gt 0) {
    Write-Host "  Program arguments: $($ProgramArgs -join ' ')"
}
if ($Hosts) {
    Write-Host "  Hosts: $Hosts"
}
Write-Host ""

# Check if mpiexec is available
$mpiexec = Get-Command mpiexec -ErrorAction SilentlyContinue

if (-not $mpiexec) {
    Write-Host "Error: mpiexec not found in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "For Beginners:" -ForegroundColor Yellow
    Write-Host "  You need to install Microsoft MPI to run distributed training on Windows."
    Write-Host "  Download from: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi"
    Write-Host ""
    Write-Host "  Installation steps:"
    Write-Host "    1. Download MS-MPI installer"
    Write-Host "    2. Install both the runtime (msmpisetup.exe) and SDK (msmpisdk.msi)"
    Write-Host "    3. Restart your terminal/PowerShell"
    exit 1
}

Write-Host "Using MPI command: $($mpiexec.Source)" -ForegroundColor Green
Write-Host ""

# Check if program exists
if (-not (Test-Path $Program)) {
    Write-Host "Error: Program '$Program' not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "For Beginners:" -ForegroundColor Yellow
    Write-Host "  Make sure you've built your training program and the path is correct."
    Write-Host "  Example: dotnet publish -c Release -o .\publish"
    Write-Host "  Then use: -Program '.\publish\MyTrainingApp.exe'"
    exit 1
}

# Security: Validate that Program is an executable file
$ProgramItem = Get-Item -Path $Program -ErrorAction Stop
$allowedExtensions = @('.exe', '.dll')
if ($ProgramItem.Extension -notin $allowedExtensions) {
    Write-Host "Error: Program must be an executable (.exe) or .NET assembly (.dll)" -ForegroundColor Red
    Write-Host "  Received: $($ProgramItem.Extension)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Security Note:" -ForegroundColor Yellow
    Write-Host "  Only executable files (.exe) and .NET assemblies (.dll) are allowed"
    Write-Host "  to prevent execution of potentially malicious scripts or documents."
    exit 1
}

# Security: Resolve to absolute path to prevent path traversal attacks
$Program = $ProgramItem.FullName
Write-Host "Resolved program path: $Program" -ForegroundColor Green
Write-Host ""

# Build mpiexec command
$mpiCommand = "mpiexec"
$mpiArgsList = @(
    "-n", $NumProcesses.ToString()
)

# Add hosts if specified
if ($Hosts) {
    $mpiArgsList += @("-hosts", $Hosts)
}

# Add the program
$mpiArgsList += $Program

# Add program arguments if specified
if ($ProgramArgs.Count -gt 0) {
    $mpiArgsList += $ProgramArgs
}

# Display command
Write-Host "Launching distributed training..." -ForegroundColor Yellow
Write-Host "Command: $mpiCommand $($mpiArgsList -join ' ')" -ForegroundColor Gray
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Launch distributed training
try {
    # Use Start-Process to capture output and wait for completion
    $process = Start-Process -FilePath $mpiCommand -ArgumentList $mpiArgsList -NoNewWindow -Wait -PassThru
    $exitCode = $process.ExitCode
}
catch {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "Error launching training: $_" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Cyan
    exit 1
}

# Display results
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
}
else {
    Write-Host "Training failed with exit code: $exitCode" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - Make sure all nodes can communicate (check firewalls)"
    Write-Host "  - Verify MS-MPI is installed on all machines"
    Write-Host "  - Check that the program path is correct on all machines"
    Write-Host "  - Ensure sufficient GPU memory is available"
    Write-Host "  - Try running with fewer processes to check for memory issues"
}
Write-Host "======================================" -ForegroundColor Cyan

exit $exitCode
