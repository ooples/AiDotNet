#!/bin/bash

echo "Fixing duplicated content in files..."

# Fix FoundationModelAdapter.cs
echo "Fixing FoundationModelAdapter.cs..."
# Remove the duplicate line at the end
head -n 438 "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs" > "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs.tmp"
echo "        #endregion" >> "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs.tmp"
echo "    }" >> "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs.tmp"
echo "}" >> "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs.tmp"
mv "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs.tmp" "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs"

# Fix CloudOptimizer.cs
echo "Fixing CloudOptimizer.cs..."
# Keep only the first 160 lines (before duplication)
head -n 160 "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs" > "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs.tmp"
# Add the rest of the content properly
tail -n +162 "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs" | head -n 58 >> "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs.tmp"
mv "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs.tmp" "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs"

echo "Done fixing duplicated content."