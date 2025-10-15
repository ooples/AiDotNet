#!/bin/bash

# Fix async methods in all affected files
echo "Fixing async/await errors in ProductionMonitoring files..."

# Get all unique line numbers with CS1998 errors for each file
dotnet build src/AiDotNet.csproj --no-restore 2>&1 | grep "CS1998" | \
while IFS=':' read -r file line_info rest; do
    # Extract line number
    line_num=$(echo "$line_info" | sed 's/[^0-9]//g')
    echo "File: $file, Line: $line_num"
done | sort | uniq > async_errors_detail.txt

echo "Found $(wc -l < async_errors_detail.txt) unique async errors to fix"

# Count remaining errors
echo ""
echo "Summary of error types:"
dotnet build src/AiDotNet.csproj --no-restore 2>&1 | grep "error CS" | \
    sed -E 's/.*error (CS[0-9]+).*/\1/' | sort | uniq -c | sort -nr | head -10

echo ""
echo "Files with most errors:"
dotnet build src/AiDotNet.csproj --no-restore 2>&1 | grep "error CS" | \
    cut -d'(' -f1 | sort | uniq -c | sort -nr | head -10