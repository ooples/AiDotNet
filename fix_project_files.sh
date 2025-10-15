#!/bin/bash

# Script to fix project file issues:
# 1. Remove net462 target framework
# 2. Update net7.0 to net8.0
# 3. Add missing project reference to BenchmarkTests

echo "Fixing project file issues..."

# Function to fix target frameworks in a project file
fix_target_frameworks() {
    local file=$1
    echo "Processing: $file"
    
    # Create a backup
    cp "$file" "${file}.bak"
    
    # Remove net462 and update net7.0 to net8.0
    # Handle both single line and multi-line TargetFramework(s) tags
    sed -i 's/<TargetFrameworks>net462;net6.0;net8.0<\/TargetFrameworks>/<TargetFrameworks>net6.0;net8.0<\/TargetFrameworks>/g' "$file"
    sed -i 's/<TargetFrameworks>net8.0;net7.0;net6.0;net462<\/TargetFrameworks>/<TargetFrameworks>net8.0;net6.0<\/TargetFrameworks>/g' "$file"
    sed -i 's/net7\.0/net8.0/g' "$file"
    
    # Remove .NET Framework specific conditions
    sed -i '/<ItemGroup Condition=".*net462.*">/,/<\/ItemGroup>/d' "$file"
    
    # Clean up backup if successful
    if [ $? -eq 0 ]; then
        rm "${file}.bak"
        echo "  ✓ Fixed target frameworks"
    else
        echo "  ✗ Error fixing file, backup kept at ${file}.bak"
    fi
}

# Fix all project files
fix_target_frameworks "/home/ooples/AiDotNet/src/AiDotNet.csproj"
fix_target_frameworks "/home/ooples/AiDotNet/testconsole/AiDotNetTestConsole.csproj"
fix_target_frameworks "/home/ooples/AiDotNet/tests/AiDotNetTests.csproj"
fix_target_frameworks "/home/ooples/AiDotNet/AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj"

# Add project reference to BenchmarkTests
echo "Adding project reference to BenchmarkTests..."
BENCHMARK_FILE="/home/ooples/AiDotNet/AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj"

# Check if project reference already exists
if ! grep -q "ProjectReference.*AiDotNet.csproj" "$BENCHMARK_FILE"; then
    # Find the closing </Project> tag and insert before it
    sed -i '/<\/Project>/i\
  <ItemGroup>\
    <ProjectReference Include="..\/src\/AiDotNet.csproj" \/>\
  <\/ItemGroup>\
' "$BENCHMARK_FILE"
    echo "  ✓ Added project reference to AiDotNet"
else
    echo "  ℹ Project reference already exists"
fi

echo "Done fixing project files."