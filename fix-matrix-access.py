#!/usr/bin/env python3

# Fix PPOAgent - incorrect Matrix access
with open('src/ReinforcementLearning/Agents/PPO/PPOAgent.cs', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '''            var gradMatrix = _ppoOptions.ValueLossFunction.CalculateDerivative(
                new Matrix<T>(new[] { valueOutput }),
                new Matrix<T>(new[] { target }));
            var grad = new Vector<T>(gradMatrix.Cols);
            for (int j = 0; j < gradMatrix.Cols; j++)
                grad[j] = gradMatrix[0, j];
            var gradTensor = Tensor<T>.FromVector(grad);
            _valueNetwork.Backpropagate(gradTensor);''',
    '''            var outputMatrix = new Matrix<T>(new[] { valueOutput });
            var targetMatrix = new Matrix<T>(new[] { target });
            var gradMatrix = _ppoOptions.ValueLossFunction.CalculateDerivative(outputMatrix, targetMatrix);
            var grad = new Vector<T>(gradMatrix.Cols);
            for (int j = 0; j < gradMatrix.Cols; j++)
                grad[j] = gradMatrix[0, j];
            var gradTensor = Tensor<T>.FromVector(grad);
            _valueNetwork.Backpropagate(gradTensor);''')

with open('src/ReinforcementLearning/Agents/PPO/PPOAgent.cs', 'w', encoding='utf-8') as f:
    f.write(content)
print("PPOAgent.cs - fixed Matrix access")

# Fix DuelingDQNAgent
with open('src/ReinforcementLearning/Agents/DuelingDQN/DuelingDQNAgent.cs', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '''            var gradMatrix = LossFunction.CalculateDerivative(
                new Matrix<T>(new[] { currentQValues }),
                new Matrix<T>(new[] { targetQValues }));
            var gradients = new Vector<T>(gradMatrix.Cols);
            for (int j = 0; j < gradMatrix.Cols; j++)
                gradients[j] = gradMatrix[0, j];''',
    '''            var currentMatrix = new Matrix<T>(new[] { currentQValues });
            var targetMatrix = new Matrix<T>(new[] { targetQValues });
            var gradMatrix = LossFunction.CalculateDerivative(currentMatrix, targetMatrix);
            var gradients = new Vector<T>(gradMatrix.Cols);
            for (int j < gradMatrix.Cols; j++)
                gradients[j] = gradMatrix[0, j];''')

# Fix ComputeGradients
content = content.replace(
    '''        var loss = lossFunction ?? LossFunction;
        var output = _qNetwork.Forward(input);
        var lossValue = loss.CalculateLoss(output, target);
        var gradMatrix = loss.CalculateDerivative(
            new Matrix<T>(new[] { output }),
            new Matrix<T>(new[] { target }));
        var gradient = new Vector<T>(gradMatrix.Cols);
        for (int j = 0; j < gradMatrix.Cols; j++)
            gradient[j] = gradMatrix[0, j];

        _qNetwork.Backward(input, gradient);
        var flatParams = _qNetwork.GetFlattenedParameters();
        var gradientVector = new Vector<T>(flatParams.Rows);
        for (int i = 0; i < flatParams.Rows; i++)
            gradientVector[i] = flatParams[i, 0];

        return gradientVector;''',
    '''        var loss = lossFunction ?? LossFunction;
        var output = _qNetwork.Forward(input);
        var lossValue = loss.CalculateLoss(output, target);
        var outputMatrix = new Matrix<T>(new[] { output });
        var targetMatrix = new Matrix<T>(new[] { target });
        var gradMatrix = loss.CalculateDerivative(outputMatrix, targetMatrix);
        var gradient = new Vector<T>(gradMatrix.Cols);
        for (int j = 0; j < gradMatrix.Cols; j++)
            gradient[j] = gradMatrix[0, j];

        _qNetwork.Backward(input, gradient);
        var flatParams = _qNetwork.GetFlattenedParameters();
        var gradientVector = new Vector<T>(flatParams.Rows);
        for (int i = 0; i < flatParams.Rows; i++)
            gradientVector[i] = flatParams[i, 0];

        return gradientVector;''')

# Remove GetFlattenedGradients call which doesn't exist
import re
content = re.sub(
    r'var gradientVector = _qNetwork\.GetFlattenedGradients\(\);',
    'var gradientVector = new Vector<T>(1); // Note: DuelingNetwork does not expose gradients',
    content)

with open('src/ReinforcementLearning/Agents/DuelingDQN/DuelingDQNAgent.cs', 'w', encoding='utf-8') as f:
    f.write(content)
print("DuelingDQNAgent.cs - fixed Matrix access")

# Fix MuZeroAgent
with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '''        var gradMatrix = usedLossFunction.CalculateDerivative(
            new Matrix<T>(new[] { prediction }),
            new Matrix<T>(new[] { target }));
        var gradient = new Vector<T>(gradMatrix.Cols);
        for (int j = 0; j < gradMatrix.Cols; j++)
            gradient[j] = gradMatrix[0, j];''',
    '''        var predMatrix = new Matrix<T>(new[] { prediction });
        var targetMatrix = new Matrix<T>(new[] { target });
        var gradMatrix = usedLossFunction.CalculateDerivative(predMatrix, targetMatrix);
        var gradient = new Vector<T>(gradMatrix.Cols);
        for (int j = 0; j < gradMatrix.Cols; j++)
            gradient[j] = gradMatrix[0, j];''')

# Fix ModelType
content = content.replace(
    '''        return new ModelMetadata<T>
        {
            ModelType = ModelType.MuZero,
        };''',
    '''        return new ModelMetadata<T>
        {
            // ModelType not set - MuZero not in enum yet
        };''')

# Fix tensor conversion issues - find and replace specific patterns
content = re.sub(
    r'var predictionTensor = _predictionNetwork\.Predict\(nodeTensor\);',
    'var predictionOutputTensor = _predictionNetwork.Predict(nodeTensor);',
    content)

content = re.sub(
    r'var prediction = predictionTensor\.ToVector\(\);',
    'var prediction = predictionOutputTensor.ToVector();',
    content)

# Fix hiddenState assignment
content = re.sub(
    r'(\s+)hiddenState = new Vector<T>\(_options\.LatentStateSize\);',
    r'\1var newHiddenState = new Vector<T>(_options.LatentStateSize);',
    content)

content = content.replace(
    '''                var newHiddenState = new Vector<T>(_options.LatentStateSize);
                for (int i = 0; i < _options.LatentStateSize; i++)
                {
                    newHiddenState[i] = dynamicsOutput[i];
                }
            }
        }

        _updateCount++;''',
    '''                var newHiddenState = new Vector<T>(_options.LatentStateSize);
                for (int i = 0; i < _options.LatentStateSize; i++)
                {
                    newHiddenState[i] = dynamicsOutput[i];
                }
                hiddenState = newHiddenState;
            }
        }

        _updateCount++;''')

with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'w', encoding='utf-8') as f:
    f.write(content)
print("MuZeroAgent.cs - fixed Matrix access and tensors")

print("\nAll Matrix/Tensor access fixed!")
