#!/bin/bash

# Fix nullable parameter mismatches in OnlineLearning algorithms
echo "Fixing OnlineLearning algorithms..."

# Fix all OnlineLearning SaveModel methods
for file in /home/ooples/AiDotNet/src/OnlineLearning/Algorithms/*.cs; do
    if grep -q "public override void SaveModel(string filePath)" "$file"; then
        echo "Fixing SaveModel in $(basename "$file")"
        sed -i 's/public override void SaveModel(string filePath)/public override void SaveModel(string? filePath)/' "$file"
    fi
done

# Fix ReinforcementLearning SaveModel and LoadModel methods
echo -e "\nFixing ReinforcementLearning models..."

# List of files with SaveModel/LoadModel overrides
rl_files=(
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/ActorCriticModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/DDPGModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/HRARLModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/MultiAgentTransformerModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/PPOModel.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Models/REINFORCEModel.cs"
)

for file in "${rl_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Fixing $(basename "$file")"
        # Fix SaveModel
        sed -i 's/public override void SaveModel(string path)/public override void SaveModel(string? path)/' "$file"
        # Fix LoadModel
        sed -i 's/public override void LoadModel(string path)/public override void LoadModel(string? path)/' "$file"
    fi
done

# Fix QRDQNAgent which seems to use filePath instead of path
echo -e "\nFixing QRDQNAgent..."
if [ -f "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/QRDQNAgent.cs" ]; then
    sed -i 's/public override void SaveModel(string filePath)/public override void SaveModel(string? filePath)/' "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/QRDQNAgent.cs"
fi

echo -e "\nDone! All nullable parameter mismatches should be fixed."