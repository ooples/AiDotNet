#!/bin/bash
# Script to check UTF-8 encoding in all C# files

echo "Checking UTF-8 encoding across entire codebase..."

corruption_found=false
total_issues=0

# Check all .cs files
while IFS= read -r file; do
    if [ -f "$file" ]; then
        count=$(grep -o "�" "$file" 2>/dev/null | wc -l)
        count=$(echo "$count" | tr -d ' \n')
        if [ "$count" -gt 0 ]; then
            echo "❌ $file: $count issue(s)"
            corruption_found=true
            total_issues=$((total_issues + count))
        fi
    fi
done < <(find src tests -name "*.cs" 2>/dev/null)

echo ""
if [ "$corruption_found" = true ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  FAILED: Found $total_issues UTF-8 encoding corruption(s)       ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 1
else
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  PASSED: No UTF-8 encoding issues found                        ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 0
fi
