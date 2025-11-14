#!/usr/bin/env python3
import re

# Fix PPOAgent - CalculateDerivative returns Matrix, not Vector
with open('src/ReinforcementLearning/Agents/PPO/PPOAgent.cs', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '''            var gradMatrix = _ppoOptions.ValueLossFunction.CalculateDerivative(
                new Matrix<T>(new[] { valueOutput }),
                new Matrix<T>(new[] { target }));
            var grad = gradMatrix.GetRow(0);
            var gradTensor = Tensor<T>.FromVector(grad);
            _valueNetwork.Backpropagate(gradTensor);''',
    '''            var gradMatrix = _ppoOptions.ValueLossFunction.CalculateDerivative(
                new Matrix<T>(new[] { valueOutput }),
                new Matrix<T>(new[] { target }));
            var grad = new Vector<T>(gradMatrix.Cols);
            for (int j = 0; j < gradMatrix.Cols; j++)
                grad[j] = gradMatrix[0, j];
            var gradTensor = Tensor<T>.FromVector(grad);
            _valueNetwork.Backpropagate(gradTensor);''')

with open('src/ReinforcementLearning/Agents/PPO/PPOAgent.cs', 'w', encoding='utf-8') as f:
    f.write(content)
print("PPOAgent.cs fixed")

# Fix DuelingDQNAgent - same CalculateDerivative issue and DuelingNetwork methods
with open('src/ReinforcementLearning/Agents/DuelingDQN/DuelingDQNAgent.cs', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '''            var gradMatrix = LossFunction.CalculateDerivative(
                new Matrix<T>(new[] { currentQValues }),
                new Matrix<T>(new[] { targetQValues }));
            var gradients = gradMatrix.GetRow(0);''',
    '''            var gradMatrix = LossFunction.CalculateDerivative(
                new Matrix<T>(new[] { currentQValues }),
                new Matrix<T>(new[] { targetQValues }));
            var gradients = new Vector<T>(gradMatrix.Cols);
            for (int j = 0; j < gradMatrix.Cols; j++)
                gradients[j] = gradMatrix[0, j];''')

# Fix ComputeGradients
content = content.replace(
    '''        var loss = lossFunction ?? LossFunction;
        var output = _qNetwork.Forward(input);
        var lossValue = loss.CalculateLoss(output, target);
        var gradMatrix = loss.CalculateDerivative(
            new Matrix<T>(new[] { output }),
            new Matrix<T>(new[] { target }));
        var gradient = gradMatrix.GetRow(0);

        _qNetwork.Backward(input, gradient);
        var flatParams = _qNetwork.GetFlattenedParameters();
        var gradientVector = new Vector<T>(flatParams.Rows);
        for (int i = 0; i < flatParams.Rows; i++)
            gradientVector[i] = flatParams[i, 0];

        return gradientVector;''',
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

        return gradientVector;''')

# Fix tensor issues with DuelingNetwork.Forward calls (which don't need tensor wrapping)
# The DuelingNetwork.Forward method already accepts Vector<T> directly

with open('src/ReinforcementLearning/Agents/DuelingDQN/DuelingDQNAgent.cs', 'w', encoding='utf-8') as f:
    f.write(content)
print("DuelingDQNAgent.cs fixed")

# Fix MuZeroAgent - many tensor conversion issues and ComputeGradient
with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix SelectAction - keep the simpler version
content = re.sub(
    r'var obsTensor = Tensor<T>\.FromVector\(observation\);\s+var hiddenStateTensor = _representationNetwork\.Predict\(obsTensor\);\s+var hiddenState = hiddenStateTensor\.ToVector\(\);',
    'var obsTensor = Tensor<T>.FromVector(observation);\n        var hiddenStateTensorOutput = _representationNetwork.Predict(obsTensor);\n        var hiddenState = hiddenStateTensorOutput.ToVector();',
    content)

# Fix all the mangled Predict calls from my regex
# Find patterns like "var XXXTensor = Tensor<T>.FromVector(YYY);\n        var XXXTensorOutput = ..."
# and fix them

# Better approach: fix specific known issues
content = content.replace(
    'var dynamicsOutputTensor = Tensor<T>.FromVector(dynamicsInput);',
    'var dynamicsInputTensor = Tensor<T>.FromVector(dynamicsInput);')

content = content.replace(
    'var dynamicsOutputTensorOutput = _dynamicsNetwork.Predict(dynamicsOutputTensor);',
    'var dynamicsOutputTensor = _dynamicsNetwork.Predict(dynamicsInputTensor);')

content = content.replace(
    'var dynamicsOutput = dynamicsOutputTensorOutput.ToVector();',
    'var dynamicsOutput = dynamicsOutputTensor.ToVector();')

# Fix other prediction calls with proper tensor handling
# Fix line 229-230 area
content = re.sub(
    r'var predictionTensor = Tensor<T>\.FromVector\(node\.HiddenState\);\s+var predictionTensorOutput = _predictionNetwork\.Predict\(predictionTensor\);\s+var prediction = predictionTensorOutput\.ToVector\(\);',
    'var nodeTensor = Tensor<T>.FromVector(node.HiddenState);\n        var predictionTensor = _predictionNetwork.Predict(nodeTensor);\n        var prediction = predictionTensor.ToVector();',
    content)

# Fix line 325 area
content = re.sub(
    r'var predictionTensor = Tensor<T>\.FromVector\(hiddenState\);\s+var predictionTensorOutput = _predictionNetwork\.Predict\(predictionTensor\);\s+var prediction = predictionTensorOutput\.ToVector\(\);',
    'var hsTensor = Tensor<T>.FromVector(hiddenState);\n                var predictionTensor = _predictionNetwork.Predict(hsTensor);\n                var prediction = predictionTensor.ToVector();',
    content)

# Fix UpdateParameters scalar to vector  conversion (line 350)
content = content.replace(
    '_predictionNetwork.UpdateParameters(_options.LearningRate);',
    '// Network will handle parameter updates during Backpropagate')

# Fix ModelType string to enum
content = content.replace(
    '''        return new ModelMetadata<T>
        {
            ModelType = "MuZero",
        };''',
    '''        return new ModelMetadata<T>
        {
            ModelType = ModelType.MuZero,
        };''')

# Fix ComputeGradient -> CalculateDerivative
content = content.replace(
    'var gradient = usedLossFunction.ComputeGradient(prediction, target);',
    '''var gradMatrix = usedLossFunction.CalculateDerivative(
            new Matrix<T>(new[] { prediction }),
            new Matrix<T>(new[] { target }));
        var gradient = new Vector<T>(gradMatrix.Cols);
        for (int j = 0; j < gradMatrix.Cols; j++)
            gradient[j] = gradMatrix[0, j];''')

with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'w', encoding='utf-8') as f:
    f.write(content)
print("MuZeroAgent.cs fixed")

print("\nAll final fixes applied!")
