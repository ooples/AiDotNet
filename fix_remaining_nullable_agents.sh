#!/bin/bash

# Fix nullable parameter mismatches in remaining ReinforcementLearning agents
echo "Fixing remaining ReinforcementLearning agents..."

# List of agent files that need fixing
agent_files=(
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/AdvantageActorCriticAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/DDPGAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/DecisionTransformerAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/HRARLAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/MBPOAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/PPOAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/SACAgent.cs"
    "/home/ooples/AiDotNet/src/ReinforcementLearning/Agents/TD3Agent.cs"
)

for file in "${agent_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Fixing $(basename "$file")"
        # Fix Save method
        sed -i 's/public override void Save(string filePath)/public override void Save(string? filePath)/' "$file"
        # Fix Load method
        sed -i 's/public override void Load(string filePath)/public override void Load(string? filePath)/' "$file"
    fi
done

echo -e "\nDone! All nullable parameter mismatches in agents should be fixed."