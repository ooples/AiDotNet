#!/bin/bash

# Script to fix GetModelMetaData method names to GetModelMetadata

echo "Fixing GetModelMetaData method names..."

# Find all files containing GetModelMetaData
find /home/ooples/AiDotNet/src -type f -name "*.cs" -exec grep -l "GetModelMetaData" {} + 2>/dev/null | while read -r file; do
    echo "Processing: $file"
    
    # Replace GetModelMetaData with GetModelMetadata
    sed -i 's/GetModelMetaData/GetModelMetadata/g' "$file"
    
    # Check if successful
    count=$(grep -c "GetModelMetaData" "$file" 2>/dev/null || echo "0")
    if [ "$count" -eq "0" ]; then
        echo "  ✓ Fixed all occurrences"
    else
        echo "  ⚠ Still has $count occurrences of GetModelMetaData"
    fi
done

echo "Done fixing GetModelMetaData method names."