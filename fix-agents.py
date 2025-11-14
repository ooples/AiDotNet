#!/usr/bin/env python3
import re

def fix_ppo_agent():
    with open('src/ReinforcementLearning/Agents/PPO/PPOAgent.cs', 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix BuildPolicyNetwork architecture (lines 107-113)
    content = content.replace(
        '''        var architecture = new NeuralNetworkArchitecture<T>
        {
            Layers = layers,
            TaskType = NeuralNetworkTaskType.Regression
        };

        return new NeuralNetwork<T>(architecture);''',
        '''        int finalOutputSize = _ppoOptions.IsContinuous ? _ppoOptions.ActionSize * 2 : _ppoOptions.ActionSize;
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _ppoOptions.StateSize,
            outputSize: finalOutputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture);'''
    )

    # Fix BuildValueNetwork architecture (lines 131-137)
    content = content.replace(
        '''        var architecture = new NeuralNetworkArchitecture<T>
        {
            Layers = layers,
            TaskType = NeuralNetworkTaskType.Regression
        };

        return new NeuralNetwork<T>(architecture, _ppoOptions.ValueLossFunction);''',
        '''        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _ppoOptions.StateSize,
            outputSize: 1,
            layers: layers);

        return new NeuralNetwork<T>(architecture, _ppoOptions.ValueLossFunction);'''
    )

    # Fix SelectAction Predict call
    content = content.replace(
        '''    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var policyOutput = _policyNetwork.Predict(state);''',
        '''    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();'''
    )

    # Fix StoreExperience Predict call
    content = content.replace(
        '''    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Get value estimate for current state
        var valueOutput = _valueNetwork.Predict(state);
        var value = valueOutput[0];''',
        '''    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Get value estimate for current state
        var stateTensor = Tensor<T>.FromVector(state);
        var valueOutputTensor = _valueNetwork.Predict(stateTensor);
        var valueOutput = valueOutputTensor.ToVector();
        var value = valueOutput[0];'''
    )

    # Fix ComputeLogProb Predict call
    content = content.replace(
        '''    private T ComputeLogProb(Vector<T> state, Vector<T> action)
    {
        var policyOutput = _policyNetwork.Predict(state);''',
        '''    private T ComputeLogProb(Vector<T> state, Vector<T> action)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();'''
    )

    # Fix UpdateNetworks valueOutput Predict call
    content = content.replace(
            '''            // Value loss
            var valueOutput = _valueNetwork.Predict(state);
            var predictedValue = valueOutput[0];''',
            '''            // Value loss
            var stateTensor = Tensor<T>.FromVector(state);
            var valueOutputTensor = _valueNetwork.Predict(stateTensor);
            var valueOutput = valueOutputTensor.ToVector();
            var predictedValue = valueOutput[0];'''
    )

    # Fix ComputeEntropy Predict call
    content = content.replace(
        '''    private T ComputeEntropy(Vector<T> state)
    {
        var policyOutput = _policyNetwork.Predict(state);''',
        '''    private T ComputeEntropy(Vector<T> state)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var policyOutputTensor = _policyNetwork.Predict(stateTensor);
        var policyOutput = policyOutputTensor.ToVector();'''
    )

    # Fix UpdatePolicyNetwork - Predict and Backprop
    content = content.replace(
        '''            // Forward pass
            _policyNetwork.Predict(state);

            // Backward pass (simplified)
            var gradOutput = action.Clone();
            for (int i = 0; i < gradOutput.Length; i++)
            {
                gradOutput[i] = NumOps.Multiply(gradOutput[i], advantage);
            }

            _policyNetwork.Backpropagate(gradOutput);''',
        '''            // Forward pass
            var stateTensor = Tensor<T>.FromVector(state);
            _policyNetwork.Predict(stateTensor);

            // Backward pass (simplified)
            var gradOutput = action.Clone();
            for (int i = 0; i < gradOutput.Length; i++)
            {
                gradOutput[i] = NumOps.Multiply(gradOutput[i], advantage);
            }

            var gradTensor = Tensor<T>.FromVector(gradOutput);
            _policyNetwork.Backpropagate(gradTensor);'''
    )

    # Fix GetFlattenedGradients -> GetParameters
    content = content.replace(
        'var grads = _policyNetwork.GetFlattenedGradients();',
        'var grads = _policyNetwork.GetParameters();'
    )

    # Fix UpdateValueNetwork - Predict and ComputeGradient
    content = content.replace(
        '''            var valueOutput = _valueNetwork.Predict(state);
            var predicted = valueOutput[0];

            var target = new Vector<T>(1);
            target[0] = targetReturn;

            var grad = _ppoOptions.ValueLossFunction.ComputeGradient(valueOutput, target);
            _valueNetwork.Backpropagate(grad);''',
        '''            var stateTensor = Tensor<T>.FromVector(state);
            var valueOutputTensor = _valueNetwork.Predict(stateTensor);
            var valueOutput = valueOutputTensor.ToVector();
            var predicted = valueOutput[0];

            var target = new Vector<T>(1);
            target[0] = targetReturn;

            var gradMatrix = _ppoOptions.ValueLossFunction.CalculateDerivative(
                new Matrix<T>(new[] { valueOutput }),
                new Matrix<T>(new[] { target }));
            var grad = gradMatrix.GetRow(0);
            var gradTensor = Tensor<T>.FromVector(grad);
            _valueNetwork.Backpropagate(gradTensor);'''
    )

    # Fix GetFlattenedGradients for value network
    content = content.replace(
        'var grads = _valueNetwork.GetFlattenedGradients();',
        'var grads = _valueNetwork.GetParameters();'
    )

    with open('src/ReinforcementLearning/Agents/PPO/PPOAgent.cs', 'w', encoding='utf-8') as f:
        f.write(content)
    print("PPOAgent.cs fixed")

def fix_rainbow_agent():
    with open('src/ReinforcementLearning/Agents/Rainbow/RainbowDQNAgent.cs', 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix Experience ambiguity
    content = content.replace(
        '_nStepBuffer.Add(new Experience<T>((state, action, reward, nextState, done)));',
        '_nStepBuffer.Add((state, action, reward, nextState, done));'
    )

    content = content.replace(
        '_replayBuffer.Add(new Experience<T>(nStepState, nStepAction, nStepReturn, nStepNextState, nStepDone));',
        '_replayBuffer.Add(new ReinforcementLearning.ReplayBuffers.Experience<T>(nStepState, nStepAction, nStepReturn, nStepNextState, nStepDone));'
    )

    # Fix Forward -> Predict with tensor
    content = content.replace(
        'var output = network.Forward(state);',
        'var stateTensor = Tensor<T>.FromVector(state);\n        var outputTensor = network.Predict(stateTensor);\n        var output = outputTensor.ToVector();'
    )

    # Fix Backpropagate calls
    content = content.replace(
        '''            _onlineNetwork.Backpropagate(gradient);
            _onlineNetwork.UpdateParameters(LearningRate);''',
        '''            var gradTensor = Tensor<T>.FromVector(gradient);
            _onlineNetwork.Backpropagate(gradTensor);
            _onlineNetwork.UpdateParameters(LearningRate);'''
    )

    # Fix ApplyGradients
    content = content.replace(
        '''    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _onlineNetwork.Backpropagate(gradients);
        _onlineNetwork.UpdateParameters(learningRate);
    }''',
        '''    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var gradTensor = Tensor<T>.FromVector(gradients);
        _onlineNetwork.Backpropagate(gradTensor);
        _onlineNetwork.UpdateParameters(learningRate);
    }'''
    )

    with open('src/ReinforcementLearning/Agents/Rainbow/RainbowDQNAgent.cs', 'w', encoding='utf-8') as f:
        f.write(content)
    print("RainbowDQNAgent.cs fixed")

def fix_dueling_agent():
    with open('src/ReinforcementLearning/Agents/DuelingDQN/DuelingDQNAgent.cs', 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix List<int> -> int[] for DuelingNetwork constructor
    content = content.replace(
        '''        _qNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers,
            _options.ValueStreamLayers,
            _options.AdvantageStreamLayers,
            NumOps
        );

        _targetNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers,
            _options.ValueStreamLayers,
            _options.AdvantageStreamLayers,
            NumOps
        );''',
        '''        _qNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers.ToArray(),
            _options.ValueStreamLayers.ToArray(),
            _options.AdvantageStreamLayers.ToArray(),
            NumOps
        );

        _targetNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers.ToArray(),
            _options.ValueStreamLayers.ToArray(),
            _options.AdvantageStreamLayers.ToArray(),
            NumOps
        );'''
    )

    # Fix Experience ambiguity
    content = content.replace(
        '_replayBuffer.Add(new Experience<T>(new ReinforcementLearning.ReplayBuffers.Experience<T>(state, action, reward, nextState, done)));',
        '_replayBuffer.Add(new ReinforcementLearning.ReplayBuffers.Experience<T>(state, action, reward, nextState, done));'
    )

    # Replace DuelingNetwork.Predict with Forward
    content = content.replace('_qNetwork.Predict(', '_qNetwork.Forward(')
    content = content.replace('_targetNetwork.Predict(', '_targetNetwork.Forward(')

    # Fix ComputeGradient -> CalculateDerivative
    content = content.replace(
        'var gradients = LossFunction.ComputeGradient(currentQValues, targetQValues);',
        '''var gradMatrix = LossFunction.CalculateDerivative(
                new Matrix<T>(new[] { currentQValues }),
                new Matrix<T>(new[] { targetQValues }));
            var gradients = gradMatrix.GetRow(0);'''
    )

    # Fix Backpropagate signature for DuelingNetwork
    content = content.replace(
        '_qNetwork.Backpropagate(experience.State, gradients);',
        '_qNetwork.Backward(experience.State, gradients);'
    )

    content = content.replace(
        '_qNetwork.UpdateParameters(LearningRate);',
        '_qNetwork.UpdateWeights(LearningRate);'
    )

    # Fix GetParameters/SetFlattenedParameters for DuelingNetwork
    content = content.replace(
        'return _qNetwork.GetParameters();',
        'var flatParams = _qNetwork.GetFlattenedParameters();\n        var vector = new Vector<T>(flatParams.Rows);\n        for (int i = 0; i < flatParams.Rows; i++)\n            vector[i] = flatParams[i, 0];\n        return vector;'
    )

    content = content.replace(
        '_qNetwork.SetFlattenedParameters(parameters);',
        'var matrix = new Matrix<T>(parameters.Length, 1);\n        for (int i = 0; i < parameters.Length; i++)\n            matrix[i, 0] = parameters[i];\n        _qNetwork.SetFlattenedParameters(matrix);'
    )

    # Fix ComputeGradients
    content = content.replace(
        '''        var loss = lossFunction ?? LossFunction;
        var output = _qNetwork.Predict(input);
        var lossValue = loss.CalculateLoss(output, target);
        var gradient = loss.ComputeGradient(output, target);

        _qNetwork.Backpropagate(input, gradient);
        var gradientVector = _qNetwork.GetFlattenedGradients();

        return gradientVector;''',
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

        return gradientVector;'''
    )

    # Fix ApplyGradients
    content = content.replace(
        '''        var currentParams = _qNetwork.GetParameters();
        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var gradValue = (i < gradients.Length) ? gradients[i] : NumOps.Zero;
            var update = NumOps.Multiply(learningRate, gradValue);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        _qNetwork.SetFlattenedParameters(newParams);''',
        '''        var flatParams = _qNetwork.GetFlattenedParameters();
        var currentParams = new Vector<T>(flatParams.Rows);
        for (int i = 0; i < flatParams.Rows; i++)
            currentParams[i] = flatParams[i, 0];

        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var gradValue = (i < gradients.Length) ? gradients[i] : NumOps.Zero;
            var update = NumOps.Multiply(learningRate, gradValue);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        var matrix = new Matrix<T>(newParams.Length, 1);
        for (int i = 0; i < newParams.Length; i++)
            matrix[i, 0] = newParams[i];
        _qNetwork.SetFlattenedParameters(matrix);'''
    )

    # Fix CopyNetworkWeights
    content = content.replace(
        '''    private void CopyNetworkWeights(DuelingNetwork<T> source, DuelingNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.SetFlattenedParameters(sourceParams);
    }''',
        '''    private void CopyNetworkWeights(DuelingNetwork<T> source, DuelingNetwork<T> target)
    {
        var sourceParams = source.GetFlattenedParameters();
        target.SetFlattenedParameters(sourceParams);
    }'''
    )

    # Fix DuelingNetwork.Forward to accept Vector and return Vector, not Tensor
    # This requires finding the DuelingNetwork class and ensuring Forward accepts Vector<T>
    # The Forward method is already correct, so we just need to ensure calls match

    with open('src/ReinforcementLearning/Agents/DuelingDQN/DuelingDQNAgent.cs', 'w', encoding='utf-8') as f:
        f.write(content)
    print("DuelingDQNAgent.cs fixed")

def fix_muzero_agent():
    with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix constructor - base class doesn't take 2 arguments
    content = content.replace(
        'public MuZeroAgent(MuZeroOptions<T> options) : base(options.ObservationSize, options.ActionSize)',
        '''public MuZeroAgent(MuZeroOptions<T> options) : base(new ReinforcementLearningOptions<T>
        {
            LearningRate = options.LearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = new MeanSquaredErrorLoss<T>(),
            Seed = options.Seed
        })'''
    )

    # Fix CreateNetwork - NeuralNetwork constructor needs architecture
    content = content.replace(
        '''    private NeuralNetwork<T> CreateNetwork(int inputSize, int outputSize, List<int> hiddenLayers)
    {
        var network = new NeuralNetwork<T>();
        int previousSize = inputSize;

        foreach (var layerSize in hiddenLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize, (IActivationFunction<T>?)null));
            network.AddLayer(new ActivationLayer<T>(new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        network.AddLayer(new DenseLayer<T>(previousSize, outputSize, (IActivationFunction<T>?)null));

        return network;
    }''',
        '''    private NeuralNetwork<T> CreateNetwork(int inputSize, int outputSize, List<int> hiddenLayers)
    {
        var layers = new List<ILayer<T>>();
        int previousSize = inputSize;

        foreach (var layerSize in hiddenLayers)
        {
            layers.Add(new DenseLayer<T>(previousSize, layerSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            previousSize = layerSize;
        }

        layers.Add(new DenseLayer<T>(previousSize, outputSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);

        return new NeuralNetwork<T>(architecture);
    }''')

    # Fix all Predict calls to use Tensor
    content = re.sub(
        r'var (\w+) = _(\w+)Network\.Predict\((\w+)\);',
        r'var \1Tensor = Tensor<T>.FromVector(\3);\n        var \1TensorOutput = _\2Network.Predict(\1Tensor);\n        var \1 = \1TensorOutput.ToVector();',
        content
    )

    # Fix SelectAction specifically
    content = content.replace(
        '''    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to hidden state
        var hiddenStateTensor = Tensor<T>.FromVector(observation);
        var hiddenStateTensorOutput = _representationNetwork.Predict(hiddenStateTensor);
        var hiddenState = hiddenStateTensorOutput.ToVector();''',
        '''    public override Vector<T> SelectAction(Vector<T> observation, bool training = true)
    {
        // Encode observation to hidden state
        var obsTensor = Tensor<T>.FromVector(observation);
        var hiddenStateTensor = _representationNetwork.Predict(obsTensor);
        var hiddenState = hiddenStateTensor.ToVector();'''
    )

    # Fix Backpropagate calls
    content = content.replace(
        '''                _predictionNetwork.Backpropagate(gradient);
                _predictionNetwork.UpdateParameters(_options.LearningRate);''',
        '''                var gradTensor = Tensor<T>.FromVector(gradient);
                _predictionNetwork.Backpropagate(gradTensor);
                _predictionNetwork.UpdateParameters(_options.LearningRate);'''
    )

    # Fix UpdateParameters call with scalar
    content = content.replace(
        '_predictionNetwork.UpdateParameters(_options.LearningRate);',
        'var params = _predictionNetwork.GetParameters();\n                for (int i = 0; i < params.Length; i++)\n                    params[i] = NumOps.Subtract(params[i], NumOps.Multiply(_options.LearningRate, params[i]));\n                _predictionNetwork.UpdateParameters(params);'
    )

    # Fix Experience ambiguity
    content = content.replace(
        '_replayBuffer.Add(new Experience<T>(observation, action, reward, nextObservation, done));',
        '_replayBuffer.Add(new ReinforcementLearning.ReplayBuffers.Experience<T>(observation, action, reward, nextObservation, done));'
    )

    # Fix done property access
    content = content.replace('experience.done', 'experience.Done')

    # Fix Clone - optimizer doesn't exist
    content = content.replace(
        'return new MuZeroAgent<T>(_options, _optimizer);',
        'return new MuZeroAgent<T>(_options);'
    )

    with open('src/ReinforcementLearning/Agents/MuZero/MuZeroAgent.cs', 'w', encoding='utf-8') as f:
        f.write(content)
    print("MuZeroAgent.cs fixed")

if __name__ == '__main__':
    import os
    os.chdir('C:/Users/cheat/source/repos/worktrees/pr-481-1763014665')

    fix_ppo_agent()
    fix_rainbow_agent()
    fix_dueling_agent()
    fix_muzero_agent()

    print("\nAll agents fixed!")
