#!/bin/bash

# Script to fix remaining ModelMetaData references to ModelMetadata

echo "Fixing remaining ModelMetaData references..."

# Array of files to fix
files=(
    "/home/ooples/AiDotNet/src/Interfaces/IModel.cs"
    "/home/ooples/AiDotNet/src/Optimizers/OptimizerBase.cs"
    "/home/ooples/AiDotNet/src/ProductionMonitoring/MonitoredModelWrapper.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/Networks/PolicyNetwork.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/DQNAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/QRDQNModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/PPOModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/ActorCriticModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/REINFORCEModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/HRARLModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/DecisionTransformerModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/ReinforcementLearningModelBase.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/DDPGModel.cs"
    "/home/ooples/AiDotNet/src/Ensemble/EnsembleModelBase.cs"
    "/home/ooples/AiDotNet/src/Evaluation/DefaultModelEvaluator.cs"
    "/home/ooples/AiDotNet/src/CrossValidators/NestedCrossValidator.cs"
    "/home/ooples/AiDotNet/src/CrossValidators/CrossValidatorBase.cs"
    "/home/ooples/AiDotNet/src/AutoML/AutoMLModelBase.cs"
    "/home/ooples/AiDotNet/src/AutoML/SimpleAutoMLModel.cs"
)

# Also search for more files
echo "Searching for additional files with ModelMetaData..."
find /home/ooples/AiDotNet/src -type f -name "*.cs" -exec grep -l "ModelMetaData" {} + 2>/dev/null | while read -r file; do
    if [[ ! " ${files[@]} " =~ " ${file} " ]]; then
        files+=("$file")
    fi
done

# Fix each file
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        # Replace ModelMetaData with ModelMetadata (case-sensitive)
        sed -i 's/ModelMetaData/ModelMetadata/g' "$file"
        
        # Count remaining occurrences
        count=$(grep -c "ModelMetaData" "$file" 2>/dev/null || echo "0")
        if [ "$count" -eq "0" ]; then
            echo "  ✓ Fixed all occurrences"
        else
            echo "  ⚠ Still has $count occurrences of ModelMetaData"
        fi
    fi
done

echo "Done fixing ModelMetaData references."