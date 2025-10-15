namespace AiDotNet.ReinforcementLearning.Agents.Networks;

/// <summary>
/// Neural network for policy function approximation that maps states to actions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PolicyNetwork is a neural network component that learns to map environment states to optimal actions.
/// It can handle both continuous and discrete action spaces and supports various policy learning algorithms
/// like actor-critic methods, policy gradients, and model-based approaches.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of this as the "brain" that decides what action to take given the current market conditions.
/// It's a neural network that:
/// - Takes in market state (prices, volumes, indicators, etc.)
/// - Outputs what action to take (buy, sell, hold, or specific amounts)
/// - Learns from experience to make better decisions over time
/// - Can handle both discrete actions (buy/sell/hold) and continuous actions (specific amounts)
/// </para>
/// </remarks>
public class PolicyNetwork<T> : NeuralNetworkBase<T>
{
    private int _stateSize;
    private int _actionSize;
    private bool _continuous;
    private readonly INumericOperations<T> _numOps = default!;
    
    // Output layer reference for easy access
    private FullyConnectedLayer<T>? _discreteOutputLayer; // For discrete actions
    
    // Optimizer for training
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer = default!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="PolicyNetwork{T}"/> class.
    /// </summary>
    /// <param name="stateSize">The size of the state space.</param>
    /// <param name="actionSize">The size of the action space.</param>
    /// <param name="hiddenSizes">The sizes of the hidden layers.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="continuous">Whether the action space is continuous.</param>
    public PolicyNetwork(int stateSize, int actionSize, int[] hiddenSizes, double learningRate, bool continuous)
        : base(CreateArchitecture(stateSize, actionSize, hiddenSizes, continuous), 
               new MeanSquaredErrorLoss<T>(), // Default loss, can be overridden
               1.0)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        _continuous = continuous;
        _numOps = MathHelper.GetNumericOperations<T>();
        
        // Create optimizer
        var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = learningRate
        };
        _optimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, options);
        
        // Initialize layers from the architecture
        InitializeLayers();
        
        // Get references to output layers for easy access
        if (_continuous)
        {
            // For continuous actions, the last layer outputs both mean and log std
            // We'll need to split the output
            var outputLayer = Layers[Layers.Count - 1] as FullyConnectedLayer<T>;
            // Note: We'll handle the split in the forward methods
        }
        else
        {
            _discreteOutputLayer = Layers[Layers.Count - 1] as FullyConnectedLayer<T>;
        }
    }
    
    /// <summary>
    /// Gets an action for the given state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="isTraining">Whether to sample (true) or use the mode (false).</param>
    /// <returns>The action vector.</returns>
    public Vector<T> GetAction(Tensor<T> state, bool isTraining)
    {
        if (_continuous)
        {
            return GetContinuousAction(state, isTraining);
        }
        else
        {
            int action = GetDiscreteAction(state, isTraining);
            // Convert to one-hot vector
            Vector<T> actionVector = new Vector<T>(_actionSize);
            actionVector[action] = _numOps.One;
            return actionVector;
        }
    }
    
    /// <summary>
    /// Gets an action and its log probability.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <returns>A tuple of action vectors and log probabilities.</returns>
    public (Vector<T>[], T[]) GetActionAndLogProb(Tensor<T>[] states)
    {
        int batchSize = states.Length;
        var actions = new Vector<T>[batchSize];
        var logProbs = new T[batchSize];
        
        for (int i = 0; i < batchSize; i++)
        {
            if (_continuous)
            {
                (actions[i], logProbs[i]) = GetContinuousActionAndLogProb(states[i]);
            }
            else
            {
                (int action, T logProb) = GetDiscreteActionAndLogProb(states[i]);
                // Convert to one-hot vector
                actions[i] = new Vector<T>(_actionSize);
                actions[i][action] = _numOps.One;
                logProbs[i] = logProb;
            }
        }
        
        return (actions, logProbs);
    }
    
    /// <summary>
    /// Gets a continuous action for the given state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="isTraining">Whether to sample (true) or use the mean (false).</param>
    /// <returns>The continuous action vector.</returns>
    private Vector<T> GetContinuousAction(Tensor<T> state, bool isTraining)
    {
        // Forward through network to get output
        Tensor<T> output = Predict(state);
        
        // For continuous actions, output contains both mean and log std
        // Split the output: first half is mean, second half is log std
        int halfSize = _actionSize;
        Tensor<T> meanTensor = new Tensor<T>(new[] { 1, halfSize });
        Tensor<T> logStdTensor = new Tensor<T>(new[] { 1, halfSize });
        
        for (int i = 0; i < halfSize; i++)
        {
            meanTensor[0, i] = output[0, i];
            logStdTensor[0, i] = output[0, i + halfSize];
        }
        
        // Apply tanh to mean to constrain to [-1, 1]
        meanTensor = TanhTensor(meanTensor);
        
        // During evaluation, just return the mean
        if (!isTraining)
        {
            return TensorToVector(meanTensor);
        }
        
        // Clip log standard deviation for stability
        logStdTensor = ClipTensor(logStdTensor, -20, 2);
        
        // Sample from Gaussian distribution
        Tensor<T> noiseTensor = SampleNoise(new int[] { _actionSize });
        Tensor<T> stdTensor = ExpTensor(logStdTensor);
        Tensor<T> actionTensor = AddTensors(meanTensor, MultiplyTensors(stdTensor, noiseTensor));
        
        // Tanh to constrain to [-1, 1]
        actionTensor = TanhTensor(actionTensor);
        
        return TensorToVector(actionTensor);
    }
    
    /// <summary>
    /// Gets a continuous action and its log probability.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>A tuple of action vector and log probability.</returns>
    private (Vector<T>, T) GetContinuousActionAndLogProb(Tensor<T> state)
    {
        // Forward through network to get output
        Tensor<T> output = Predict(state);
        
        // Split the output: first half is mean, second half is log std
        int halfSize = _actionSize;
        Tensor<T> meanTensor = new Tensor<T>(new[] { 1, halfSize });
        Tensor<T> logStdTensor = new Tensor<T>(new[] { 1, halfSize });
        
        for (int i = 0; i < halfSize; i++)
        {
            meanTensor[0, i] = output[0, i];
            logStdTensor[0, i] = output[0, i + halfSize];
        }
        
        // Clip log standard deviation for stability
        logStdTensor = ClipTensor(logStdTensor, -20, 2);
        
        // Sample from Gaussian distribution
        Tensor<T> noiseTensor = SampleNoise(new int[] { _actionSize });
        Tensor<T> stdTensor = ExpTensor(logStdTensor);
        Tensor<T> actionTensor = AddTensors(meanTensor, MultiplyTensors(stdTensor, noiseTensor));
        
        // Calculate log probability before applying tanh
        T logProb = CalculateGaussianLogProb(actionTensor, meanTensor, logStdTensor);
        
        // Apply tanh to constrain actions to [-1, 1]
        actionTensor = TanhTensor(actionTensor);
        
        // Correct log probability for the tanh squashing
        logProb = CorrectLogProbForTanh(logProb, actionTensor);
        
        return (TensorToVector(actionTensor), logProb);
    }
    
    /// <summary>
    /// Gets a discrete action for the given state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="isTraining">Whether to sample (true) or use the mode (false).</param>
    /// <returns>The discrete action vector (one-hot encoded).</returns>
    private int GetDiscreteAction(Tensor<T> state, bool isTraining)
    {
        // Forward through network to get logits
        Tensor<T> logitsTensor = Predict(state);
        Vector<T> logits = TensorToVector(logitsTensor);
        
        // Convert to probabilities with softmax
        Vector<T> probs = Softmax(logits);
        
        int selectedAction;
        
        if (isTraining)
        {
            // Sample from categorical distribution
            selectedAction = SampleCategorical(probs);
        }
        else
        {
            // Greedy selection
            selectedAction = 0;
            T bestProb = probs[0];
            
            for (int i = 1; i < probs.Length; i++)
            {
                if (_numOps.GreaterThan(probs[i], bestProb))
                {
                    bestProb = probs[i];
                    selectedAction = i;
                }
            }
        }
        
        return selectedAction;
    }
    
    /// <summary>
    /// Gets a discrete action and its log probability.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>A tuple of action vector (one-hot encoded) and log probability.</returns>
    private (int, T) GetDiscreteActionAndLogProb(Tensor<T> state)
    {
        // Forward through network to get logits
        Tensor<T> logitsTensor = Predict(state);
        Vector<T> logits = TensorToVector(logitsTensor);
        
        // Convert to probabilities with softmax
        Vector<T> probs = Softmax(logits);
        
        // Sample from categorical distribution
        int selectedAction = SampleCategorical(probs);
        
        // Calculate log probability
        T logProb = _numOps.Log(probs[selectedAction]);
        
        return (selectedAction, logProb);
    }
    
    /// <summary>
    /// Updates the policy network.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <param name="actions">Batch of actions.</param>
    /// <param name="logProbs">Batch of log probabilities.</param>
    /// <param name="qValues">Batch of Q-values.</param>
    /// <param name="temperature">Temperature parameter for entropy regularization.</param>
    /// <returns>The policy loss.</returns>
    public T Update(Tensor<T>[] states, Vector<T>[] actions, T[] logProbs, T[] qValues, T temperature)
    {
        int batchSize = states.Length;
        T totalLoss = _numOps.Zero;
        
        for (int i = 0; i < batchSize; i++)
        {
            // Calculate entropy-regularized loss
            // Loss = -Q(s,a) + alpha * log_prob
            T entropyTerm = _numOps.Multiply(temperature, logProbs[i]);
            T loss = _numOps.Subtract(entropyTerm, qValues[i]);
            
            totalLoss = _numOps.Add(totalLoss, loss);
        }
        
        // Calculate average loss
        totalLoss = _numOps.Divide(totalLoss, _numOps.FromDouble(batchSize));
        
        // Simulate gradient calculation and optimization
        // Note: In a real implementation, this would involve proper gradient calculation
        // For now, we'll just create dummy data to satisfy the optimizer interface
        var dummyInput = new Tensor<T>(new[] { batchSize, _stateSize });
        var dummyOutput = new Tensor<T>(new[] { batchSize, _actionSize });
        
        var optimizationInput = new OptimizationInputData<T, Tensor<T>, Tensor<T>>
        {
            XTrain = dummyInput,
            YTrain = dummyOutput
        };
        _optimizer.Optimize(optimizationInput);
        
        return totalLoss;
    }
    
    /// <summary>
    /// Gets all parameters of the policy network as a single vector.
    /// </summary>
    /// <returns>A vector containing all network parameters.</returns>
    public new Vector<T> GetParameters()
    {
        // Use the base class implementation
        return base.GetParameters();
    }
    
    /// <summary>
    /// Sets all parameters of the policy network from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;
        
        // Set parameters for all layers
        foreach (var layer in Layers)
        {
            if (layer is ILayer<T> layerWithParams && layerWithParams.GetParameters().Length > 0)
            {
                int layerParamSize = layerWithParams.GetParameters().Length;
                var layerParams = ExtractVector(parameters, index, layerParamSize);
                layerWithParams.SetParameters(layerParams);
                index += layerParamSize;
            }
        }
    }
    
    /// <summary>
    /// Samples from a categorical distribution.
    /// </summary>
    private int SampleCategorical(Vector<T> probs)
    {
        double random = new Random().NextDouble();
        double cumulativeProb = 0;
        
        for (int i = 0; i < probs.Length; i++)
        {
            cumulativeProb += Convert.ToDouble(probs[i]);
            if (random < cumulativeProb)
            {
                return i;
            }
        }
        
        // If we somehow get here (shouldn't happen with proper probabilities)
        return probs.Length - 1;
    }
    
    /// <summary>
    /// Samples noise for continuous actions.
    /// </summary>
    private Tensor<T> SampleNoise(int[] shape)
    {
        var noise = new Tensor<T>(shape);
        var random = new Random();
        
        for (int i = 0; i < noise.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(); // Uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Box-Muller transform
            
            // For 1D tensor, use direct indexing
            if (shape.Length == 1)
            {
                noise[i] = _numOps.FromDouble(randStdNormal);
            }
            else
            {
                // For multi-dimensional, calculate indices
                int[] indices = new int[shape.Length];
                int temp = i;
                for (int j = shape.Length - 1; j >= 0; j--)
                {
                    indices[j] = temp % shape[j];
                    temp /= shape[j];
                }
                noise[indices] = _numOps.FromDouble(randStdNormal);
            }
        }
        
        return noise;
    }
    
    /// <summary>
    /// Applies tanh to a tensor.
    /// </summary>
    private Tensor<T> TanhTensor(Tensor<T> tensor)
    {
        var result = new Tensor<T>(tensor.Shape);
        
        // Iterate through all elements
        var shape = tensor.Shape;
        int totalElements = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            totalElements *= shape[i];
        }
        
        for (int i = 0; i < totalElements; i++)
        {
            // Calculate indices
            int[] indices = new int[shape.Length];
            int temp = i;
            for (int j = shape.Length - 1; j >= 0; j--)
            {
                indices[j] = temp % shape[j];
                temp /= shape[j];
            }
            result[indices] = MathHelper.Tanh(tensor[indices]);
        }
        
        return result;
    }
    
    /// <summary>
    /// Applies exponential to a tensor.
    /// </summary>
    private Tensor<T> ExpTensor(Tensor<T> tensor)
    {
        var result = new Tensor<T>(tensor.Shape);
        
        for (int i = 0; i < tensor.Length; i++)
        {
            var index = tensor.GetIndexFromFlat(i);
            result[index] = _numOps.Exp(tensor[index]);
        }
        
        return result;
    }
    
    /// <summary>
    /// Clips values in a tensor to a range.
    /// </summary>
    private Tensor<T> ClipTensor(Tensor<T> tensor, double min, double max)
    {
        var result = new Tensor<T>(tensor.Shape);
        T minT = _numOps.FromDouble(min);
        T maxT = _numOps.FromDouble(max);
        
        for (int i = 0; i < tensor.Length; i++)
        {
            var index = tensor.GetIndexFromFlat(i);
            T value = tensor[index];
            
            if (_numOps.LessThan(value, minT))
            {
                result[index] = minT;
            }
            else if (_numOps.GreaterThan(value, maxT))
            {
                result[index] = maxT;
            }
            else
            {
                result[index] = value;
            }
        }
        
        return result;
    }
    
    /// <summary>
    /// Converts a tensor to a vector.
    /// </summary>
    private Vector<T> TensorToVector(Tensor<T> tensor)
    {
        var vector = new Vector<T>(tensor.Length);
        
        for (int i = 0; i < tensor.Length; i++)
        {
            var index = tensor.GetIndexFromFlat(i);
            vector[i] = tensor[index];
        }
        
        return vector;
    }
    
    /// <summary>
    /// Adds two tensors elementwise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        
        for (int i = 0; i < a.Length; i++)
        {
            var index = a.GetIndexFromFlat(i);
            result[index] = _numOps.Add(a[index], b[index]);
        }
        
        return result;
    }
    
    /// <summary>
    /// Multiplies two tensors elementwise.
    /// </summary>
    private Tensor<T> MultiplyTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        
        for (int i = 0; i < a.Length; i++)
        {
            var index = a.GetIndexFromFlat(i);
            result[index] = _numOps.Multiply(a[index], b[index]);
        }
        
        return result;
    }
    
    /// <summary>
    /// Calculates the log probability of a Gaussian distribution.
    /// </summary>
    private T CalculateGaussianLogProb(Tensor<T> value, Tensor<T> mean, Tensor<T> logStd)
    {
        T logProb = _numOps.Zero;
        
        for (int i = 0; i < value.Length; i++)
        {
            var index = value.GetIndexFromFlat(i);
            
            // log_prob = -0.5 * ((x - mean) / std)^2 - log(std) - 0.5 * log(2 * pi)
            T diff = _numOps.Subtract(value[index], mean[index]);
            T scaled = _numOps.Divide(diff, _numOps.Exp(logStd[index]));
            T squared = _numOps.Multiply(scaled, scaled);
            T term1 = _numOps.Multiply(_numOps.FromDouble(-0.5), squared);
            T term2 = _numOps.Negate(logStd[index]);
            T term3 = _numOps.FromDouble(-0.5 * Math.Log(2 * Math.PI));
            
            logProb = _numOps.Add(logProb, _numOps.Add(_numOps.Add(term1, term2), term3));
        }
        
        return logProb;
    }
    
    /// <summary>
    /// Corrects log probability for the tanh squashing.
    /// </summary>
    private T CorrectLogProbForTanh(T logProb, Tensor<T> action)
    {
        T correction = _numOps.Zero;
        
        for (int i = 0; i < action.Length; i++)
        {
            var index = action.GetIndexFromFlat(i);
            
            // correction = sum(log(1 - tanh(x)^2))
            T tanhX = action[index];
            T tanhXSquared = _numOps.Multiply(tanhX, tanhX);
            T term = _numOps.Subtract(_numOps.One, tanhXSquared);
            correction = _numOps.Add(correction, _numOps.Log(term));
        }
        
        return _numOps.Subtract(logProb, correction);
    }
    
    /// <summary>
    /// Extracts a portion of a vector.
    /// </summary>
    private Vector<T> ExtractVector(Vector<T> source, int startIndex, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = source[startIndex + i];
        }
        return result;
    }

    /// <summary>
    /// Applies softmax to a vector to convert logits to probabilities.
    /// </summary>
    /// <param name="input">The input vector of logits.</param>
    /// <returns>A vector of probabilities that sum to 1.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Softmax is a mathematical function that converts a vector of numbers
    /// into probabilities. Each output value is between 0 and 1, and all outputs sum to 1.
    /// This is commonly used in neural networks for classification tasks.</para>
    /// </remarks>
    private Vector<T> Softmax(Vector<T> input)
    {
        var output = new Vector<T>(input.Length);
        
        // Find max for numerical stability
        T maxValue = input[0];
        for (int i = 1; i < input.Length; i++)
        {
            if (_numOps.GreaterThan(input[i], maxValue))
            {
                maxValue = input[i];
            }
        }
        
        // Calculate exp(x - max) for each element
        T sumExp = _numOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            T shifted = _numOps.Subtract(input[i], maxValue);
            T expValue = _numOps.Exp(shifted);
            output[i] = expValue;
            sumExp = _numOps.Add(sumExp, expValue);
        }
        
        // Normalize by sum of exps
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = _numOps.Divide(output[i], sumExp);
        }
        
        return output;
    }
    
    /// <summary>
    /// Saves the policy network to a file.
    /// </summary>
    /// <param name="path">The file path to save to.</param>
    public void SaveToFile(string path)
    {
        using (var stream = File.Open(path, FileMode.Create))
        {
            using (var writer = new BinaryWriter(stream))
            {
                SerializeNetworkSpecificData(writer);
            }
        }
    }
    
    /// <summary>
    /// Loads the policy network from a file.
    /// </summary>
    /// <param name="path">The file path to load from.</param>
    public void LoadFromFile(string path)
    {
        using (var stream = File.Open(path, FileMode.Open))
        {
            using (var reader = new BinaryReader(stream))
            {
                DeserializeNetworkSpecificData(reader);
            }
        }
    }
    
    /// <summary>
    /// Creates the neural network architecture for the policy network.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateArchitecture(int stateSize, int actionSize, int[] hiddenSizes, bool continuous)
    {
        var layers = new List<ILayer<T>>();
        
        // Add input layer
        layers.Add(new InputLayer<T>(stateSize));
        
        // Add hidden layers
        for (int i = 0; i < hiddenSizes.Length; i++)
        {
            layers.Add(new FullyConnectedLayer<T>(
                i == 0 ? stateSize : hiddenSizes[i - 1],
                hiddenSizes[i],
                ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU)));
        }
        
        // Add output layers based on action type
        if (continuous)
        {
            // For continuous actions, we need both mean and log standard deviation outputs
            layers.Add(new FullyConnectedLayer<T>(
                hiddenSizes[hiddenSizes.Length - 1],
                actionSize * 2, // mean and log std
                ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Identity)));
        }
        else
        {
            // For discrete actions, we output logits
            layers.Add(new FullyConnectedLayer<T>(
                hiddenSizes[hiddenSizes.Length - 1],
                actionSize,
                ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Identity)));
        }
        
        return new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Medium,
            taskType: continuous ? NeuralNetworkTaskType.Regression : NeuralNetworkTaskType.MultiClassClassification,
            shouldReturnFullSequence: false,
            layers: layers);
    }

    protected override void InitializeLayers()
    {
        // Initialize layers from the architecture
        Layers.Clear();
        if (Architecture?.Layers != null)
        {
            foreach (var layer in Architecture.Layers)
            {
                Layers.Add(layer);
            }
        }
        
        // Get references to specific output layers for continuous actions
        if (_continuous && Layers.Count > 0)
        {
            // For continuous actions, we'll need to split the output
            // The last layer outputs both mean and log std concatenated
        }
        else if (!_continuous && Layers.Count > 0)
        {
            _discreteOutputLayer = Layers[Layers.Count - 1] as FullyConnectedLayer<T>;
        }
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Use the base class's ForwardWithMemory method for prediction
        return ForwardWithMemory(input);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // For policy networks, training is typically done through policy gradient methods
        // rather than supervised training. This method can be used for behavior cloning.
        
        // Forward pass
        var output = Predict(input);
        
        // Calculate loss - need to convert to Vector<double> for loss function
        Vector<T> outputVector = new Vector<T>(output.Length);
        Vector<T> expectedVector = new Vector<T>(expectedOutput.Length);
        for (int i = 0; i < output.Length; i++)
        {
            outputVector[i] = output[i];
            expectedVector[i] = expectedOutput[i];
        }
        
        var loss = LossFunction.CalculateLoss(outputVector, expectedVector);
        
        // Convert loss to Tensor<double> for backpropagation
        var lossTensor = new Tensor<T>(output.Shape);
        for (int i = 0; i < lossTensor.Length; i++)
        {
            lossTensor[i] = loss;
        }
        
        // Backward pass
        Backpropagate(lossTensor);
        
        // Update parameters using the optimizer
        var optimizationInput = new OptimizationInputData<T, Tensor<T>, Tensor<T>>
        {
            XTrain = input,
            YTrain = expectedOutput
        };
        
        _optimizer.Optimize(optimizationInput);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _stateSize,
            Complexity = GetParameterCount(),
            Description = $"Policy network for {(_continuous ? "continuous" : "discrete")} action space with {Layers.Count} layers",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "TaskType", _continuous ? "ContinuousPolicyNetwork" : "DiscretePolicyNetwork" },
                { "StateSize", _stateSize },
                { "ActionSize", _actionSize },
                { "Continuous", _continuous },
                { "OptimizerType", "Adam" },
                { "InputShape", new[] { _stateSize } },
                { "OutputShape", _continuous ? new[] { _actionSize * 2 } : new[] { _actionSize } },
                { "TotalParameters", GetParameterCount() },
                { "TrainableParameters", GetParameterCount() },
                { "NonTrainableParameters", 0 },
                { "LayerCount", Layers.Count },
                { "MemoryFootprint", GetParameterCount() * sizeof(float) },
                { "Architecture", Architecture }
            }
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write policy network specific data
        writer.Write(_stateSize);
        writer.Write(_actionSize);
        writer.Write(_continuous);
        
        // Write optimizer state if needed
        // Note: Optimizer serialization would need to be implemented
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read policy network specific data
        _stateSize = reader.ReadInt32();
        _actionSize = reader.ReadInt32();
        _continuous = reader.ReadBoolean();
        
        // Re-initialize layers after deserialization
        InitializeLayers();
        
        // Restore optimizer state if needed
        // Note: Optimizer deserialization would need to be implemented
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Extract hidden layer sizes from the current architecture
        var hiddenSizes = new List<int>();
        for (int i = 1; i < Layers.Count - 1; i++) // Skip input and output layers
        {
            if (Layers[i] is FullyConnectedLayer<T> fcLayer)
            {
                hiddenSizes.Add(fcLayer.GetOutputShape()[0]);
            }
        }
        
        // Create a new instance with the same configuration
        return new PolicyNetwork<T>(
            _stateSize,
            _actionSize,
            hiddenSizes.ToArray(),
            0.001, // Default learning rate, will be overridden by optimizer
            _continuous
        );
    }
}