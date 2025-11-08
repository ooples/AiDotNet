#!/bin/bash

################################################################################
# AiDotNet Distributed Training Launcher (Bash)
#
# This script launches distributed training using MPI across multiple processes.
#
# For Beginners:
# MPI (Message Passing Interface) is a standard for running programs across
# multiple computers or processors. Think of it like a coordinator that starts
# your program on multiple machines at once and helps them communicate.
#
# Usage:
#   ./launch-distributed-training.sh <num_processes> <program> [args...]
#
# Examples:
#   # Run on 4 GPUs locally
#   ./launch-distributed-training.sh 4 ./MyTrainingApp
#
#   # Run on 8 GPUs with additional arguments
#   ./launch-distributed-training.sh 8 ./MyTrainingApp --epochs 100 --lr 0.001
#
#   # Run across 2 machines with 4 GPUs each
#   ./launch-distributed-training.sh 8 ./MyTrainingApp --hosts machine1,machine2
################################################################################

# Check if enough arguments provided
if [ "$#" -lt 2 ]; then
    echo "Error: Insufficient arguments"
    echo ""
    echo "Usage: $0 <num_processes> <program> [args...]"
    echo ""
    echo "Arguments:"
    echo "  num_processes  - Number of processes to spawn (typically equals number of GPUs)"
    echo "  program        - Path to your training program executable"
    echo "  args           - Any additional arguments to pass to your program"
    echo ""
    echo "Examples:"
    echo "  $0 4 ./MyTrainingApp"
    echo "  $0 8 ./MyTrainingApp --epochs 100"
    exit 1
fi

# Parse arguments
NUM_PROCESSES=$1
PROGRAM=$2
shift 2
PROGRAM_ARGS=("$@")

echo "======================================"
echo "AiDotNet Distributed Training Launcher"
echo "======================================"
echo ""
echo "Configuration:"
echo "  Number of processes: $NUM_PROCESSES"
echo "  Program: $PROGRAM"
if [ "${#PROGRAM_ARGS[@]}" -gt 0 ]; then
    echo "  Program arguments: ${PROGRAM_ARGS[*]}"
else
    echo "  Program arguments: (none)"
fi
echo ""

# Check if mpiexec/mpirun is available
if command -v mpiexec &> /dev/null; then
    MPI_CMD="mpiexec"
elif command -v mpirun &> /dev/null; then
    MPI_CMD="mpirun"
else
    echo "Error: Neither mpiexec nor mpirun found in PATH"
    echo ""
    echo "For Beginners:"
    echo "  You need to install MPI to run distributed training."
    echo "  On Ubuntu/Debian: sudo apt-get install mpich"
    echo "  On macOS: brew install mpich"
    echo "  On Windows: Install Microsoft MPI from https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi"
    exit 1
fi

echo "Using MPI command: $MPI_CMD"
echo ""

# Check if program exists
if [ ! -f "$PROGRAM" ]; then
    echo "Error: Program '$PROGRAM' not found"
    echo ""
    echo "For Beginners:"
    echo "  Make sure you've built your training program and the path is correct."
    echo "  Example: dotnet publish -c Release -o ./publish"
    echo "  Then use: $0 4 ./publish/MyTrainingApp"
    exit 1
fi

# Make program executable if it isn't
if [ ! -x "$PROGRAM" ]; then
    echo "Warning: Program is not executable. Making it executable..."
    chmod +x "$PROGRAM"
fi

# Launch distributed training
echo "Launching distributed training..."
echo "Command: $MPI_CMD -n $NUM_PROCESSES $PROGRAM ${PROGRAM_ARGS[*]}"
echo ""
echo "======================================"
echo ""

# Execute MPI command
# -n: Number of processes
# The program and its arguments follow
"$MPI_CMD" -n "$NUM_PROCESSES" "$PROGRAM" "${PROGRAM_ARGS[@]}"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Make sure all nodes can communicate (check firewalls)"
    echo "  - Verify MPI is installed on all machines"
    echo "  - Check that the program path is correct on all machines"
    echo "  - Ensure sufficient GPU memory is available"
fi
echo "======================================"

exit $EXIT_CODE
