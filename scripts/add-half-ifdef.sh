#!/bin/bash
# Add conditional compilation around Half methods in all NumericOperations files

set -e

cd /home/user/AiDotNet

for file in src/NumericOperations/{Double,Int32,Int64,Byte,SByte,Short,UInt16,UInt32,UInt64,UInt,Decimal,Complex}Operations.cs; do
    echo "Processing $file..."

    # Remove any stray #endif at the end of files
    sed -i '/^}$/,/#endif$/{ /#endif$/d; }' "$file"

    # Use awk to add conditional compilation around ToHalf and FromHalf methods
    awk '
    /\/\/\/ <summary>/ && !in_half && (getline tmp) && tmp ~ /Converts.*to Half/ {
        print "#if NET5_0_OR_GREATER"
        print
        print tmp
        in_half = 1
        next
    }
    in_half && /public .* FromHalf\(/ {
        print
        while (getline && !/^[[:space:]]*}[[:space:]]*$/) {
            print
        }
        print
        print "#endif"
        in_half = 0
        next
    }
    { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"

    echo "✓ Updated $file"
done

echo ""
echo "✅ All files updated successfully!"
