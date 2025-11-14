#!/bin/bash
# Fix PPO, Rainbow, DuelingDQN, MuZero agents

cd /c/Users/cheat/source/repos/worktrees/pr-481-1763014665

# Fix PPOAgent.cs
echo "Fixing PPOAgent.cs..."

# Fix BuildPolicyNetwork architecture
sed -i '/var architecture = new NeuralNetworkArchitecture<T>/,/};/{
  /var architecture = new NeuralNetworkArchitecture<T>/{
    N
    N
    N
    N
    s/.*/        int finalOutputSize = _ppoOptions.IsContinuous ? _ppoOptions.ActionSize * 2 : _ppoOptions.ActionSize;\n        var architecture = new NeuralNetworkArchitecture<T>(\n            inputType: InputType.OneDimensional,\n            taskType: NeuralNetworkTaskType.Regression,\n            complexity: NetworkComplexity.Medium,\n            inputSize: _ppoOptions.StateSize,\n            outputSize: finalOutputSize,\n            layers: layers);/
  }
}' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

# Fix BuildValueNetwork architecture
sed -i '131,135s/.*/        var architecture = new NeuralNetworkArchitecture<T>(\n            inputType: InputType.OneDimensional,\n            taskType: NeuralNetworkTaskType.Regression,\n            complexity: NetworkComplexity.Medium,\n            inputSize: _ppoOptions.StateSize,\n            outputSize: 1,\n            layers: layers);/' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

# Fix all Predict calls to use Tensor
sed -i 's/var policyOutput = _policyNetwork\.Predict(state);/var stateTensor = Tensor<T>.FromVector(state);\n        var policyOutputTensor = _policyNetwork.Predict(stateTensor);\n        var policyOutput = policyOutputTensor.ToVector();/g' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

sed -i 's/var valueOutput = _valueNetwork\.Predict(state);/var stateTensor = Tensor<T>.FromVector(state);\n        var valueOutputTensor = _valueNetwork.Predict(stateTensor);\n        var valueOutput = valueOutputTensor.ToVector();/g' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

# Fix Backpropagate calls
sed -i 's/_policyNetwork\.Backpropagate(gradOutput);/var gradTensor = Tensor<T>.FromVector(gradOutput);\n            _policyNetwork.Backpropagate(gradTensor);/g' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

sed -i 's/_valueNetwork\.Backpropagate(grad);/var gradTensor = Tensor<T>.FromVector(grad);\n            _valueNetwork.Backpropagate(gradTensor);/g' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

# Fix GetFlattenedGradients
sed -i 's/var grads = _policyNetwork\.GetFlattenedGradients();/var grads = _policyNetwork.GetParameters();/g' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs
sed -i 's/var grads = _valueNetwork\.GetFlattenedGradients();/var grads = _valueNetwork.GetParameters();/g' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

# Fix ComputeGradient to CalculateDerivative
sed -i 's/var grad = _ppoOptions\.ValueLossFunction\.ComputeGradient(valueOutput, target);/var gradMatrix = _ppoOptions.ValueLossFunction.CalculateDerivative(\n                new Matrix<T>(new[] { valueOutput }),\n                new Matrix<T>(new[] { target }));\n            var grad = gradMatrix.GetRow(0);/g' src/ReinforcementLearning/Agents/PPO/PPOAgent.cs

echo "PPOAgent.cs fixed"

# Build to check progress
dotnet build --framework net462 2>&1 | grep -E "(PPO|Rainbow|DuelingDQN|MuZero)Agent.cs" | wc -l
