#!/bin/bash

# Script to remove 'override' keyword from methods that don't exist in base classes

echo "Fixing override keyword issues..."

# Function to fix override issues in a file
fix_file() {
    local file=$1
    echo "Processing: $file"
    
    # Create a backup
    cp "$file" "${file}.bak"
    
    # Remove 'override' from specific methods that don't exist in base classes
    sed -i 's/public override void Save(/public virtual void Save(/g' "$file"
    sed -i 's/public override void Load(/public virtual void Load(/g' "$file"
    sed -i 's/public override void Dispose(/public virtual void Dispose(/g' "$file"
    sed -i 's/public override void SetModelMetadata(/public virtual void SetModelMetadata(/g' "$file"
    sed -i 's/public override async Task<[^>]*> PredictAsync(/public virtual async Task<\1> PredictAsync(/g' "$file"
    sed -i 's/public override async Task TrainAsync(/public virtual async Task TrainAsync(/g' "$file"
    sed -i 's/public override Task<[^>]*> PredictAsync(/public virtual Task<\1> PredictAsync(/g' "$file"
    sed -i 's/public override Task TrainAsync(/public virtual Task TrainAsync(/g' "$file"
    
    # Clean up backup if successful
    if [ $? -eq 0 ]; then
        rm "${file}.bak"
        echo "  ✓ Fixed override keywords"
    else
        echo "  ✗ Error fixing file, backup kept at ${file}.bak"
    fi
}

# Fix EnsembleModelBase.cs
fix_file "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"

# Fix FoundationModelAdapter.cs
fix_file "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelAdapter.cs"

# Fix FoundationModelBase.cs
fix_file "/home/ooples/AiDotNet/src/FoundationModels/FoundationModelBase.cs"

# Fix CloudOptimizer.cs (CachedModel class)
fix_file "/home/ooples/AiDotNet/src/Deployment/CloudOptimizer.cs"

# Fix ModelIndividual.cs
fix_file "/home/ooples/AiDotNet/src/Genetics/ModelIndividual.cs"

echo "Done fixing override keywords."