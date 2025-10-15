#!/bin/bash

# Track build error progress
echo "=== Build Error Progress Tracker ==="
echo "Time: $(date)"

# Get current error count
TOTAL_ERRORS=$(dotnet build --no-restore 2>&1 | grep -c "error CS")
echo "Total Errors: $TOTAL_ERRORS"

# Count by error type
echo -e "\nError Types:"
dotnet build --no-restore 2>&1 | grep "error CS" | grep -oE "CS[0-9]+" | sort | uniq -c | sort -nr | head -10

# Count CS1061 errors by missing member
echo -e "\nCS1061 Missing Members (top 10):"
dotnet build --no-restore 2>&1 | grep "error CS1061" | grep -oE "does not contain a definition for '[^']+'" | sort | uniq -c | sort -nr | head -10

# Save to history file
echo "$(date),${TOTAL_ERRORS}" >> build_error_history.csv