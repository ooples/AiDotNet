with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix params keyword issue - revert my bad replacement
content = content.replace(
    '''                var gradTensor = Tensor<T>.FromVector(gradient);
                _predictionNetwork.Backpropagate(gradTensor);
                var params = _predictionNetwork.GetParameters();
                for (int i = 0; i < params.Length; i++)
                    params[i] = NumOps.Subtract(params[i], NumOps.Multiply(_options.LearningRate, params[i]));
                _predictionNetwork.UpdateParameters(params);''',
    '''                var gradTensor = Tensor<T>.FromVector(gradient);
                _predictionNetwork.Backpropagate(gradTensor);
                _predictionNetwork.UpdateParameters(_options.LearningRate);'''
)

with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'w', encoding='utf-8') as f:
    f.write(content)

print("MuZero syntax fixed")
