namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Neural network model that predicts environment dynamics for MBPO.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class DynamicsModel<T>
{
    private readonly int _stateSize;
    private readonly int _actionSize;
    private readonly bool _probabilistic;
    private readonly INumericOperations<T> _numOps = default!;
    
    // Neural network components
    private readonly FullyConnectedLayer<T>[] _stateInputLayers;
    private readonly FullyConnectedLayer<T>[] _actionInputLayers;
    private readonly FullyConnectedLayer<T>[] _hiddenLayers;
    private readonly FullyConnectedLayer<T> _nextStateOutputLayer = default!;
    private readonly FullyConnectedLayer<T> _rewardOutputLayer = default!;
    private readonly FullyConnectedLayer<T> _terminationOutputLayer = default!;
    
    // For probabilistic models
    private readonly FullyConnectedLayer<T>? _logVarOutputLayer;
    
    // Learning rate for manual gradient updates
    private readonly T _learningRate = default!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="DynamicsModel{T}"/> class.
    /// </summary>
    /// <param name="stateSize">The size of the state space.</param>
    /// <param name="actionSize">The size of the action space.</param>
    /// <param name="hiddenSizes">The sizes of the hidden layers.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="probabilistic">Whether to use a probabilistic model.</param>
    public DynamicsModel(int stateSize, int actionSize, int[] hiddenSizes, double learningRate, bool probabilistic)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        _probabilistic = probabilistic;
        _numOps = MathHelper.GetNumericOperations<T>();
        _learningRate = _numOps.FromDouble(learningRate);
        
        // Create input layers - we use separate layers for state and action for more flexibility
        _stateInputLayers = new FullyConnectedLayer<T>[1];
        _stateInputLayers[0] = new FullyConnectedLayer<T>(
            _stateSize,
            hiddenSizes[0] / 2, // Split the first hidden layer between state and action
            ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU));
        
        _actionInputLayers = new FullyConnectedLayer<T>[1];
        _actionInputLayers[0] = new FullyConnectedLayer<T>(
            _actionSize,
            hiddenSizes[0] / 2, // The other half of the first hidden layer
            ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU));
        
        // Create hidden layers
        _hiddenLayers = new FullyConnectedLayer<T>[hiddenSizes.Length - 1];
        for (int i = 0; i < hiddenSizes.Length - 1; i++)
        {
            _hiddenLayers[i] = new FullyConnectedLayer<T>(
                i == 0 ? hiddenSizes[0] : hiddenSizes[i], // First layer combines state and action inputs
                hiddenSizes[i + 1],
                ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.ReLU));
        }
        
        // Create output layers
        _nextStateOutputLayer = new FullyConnectedLayer<T>(
            hiddenSizes[hiddenSizes.Length - 1],
            _stateSize,
            ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Identity)); // State prediction is unbounded
        
        _rewardOutputLayer = new FullyConnectedLayer<T>(
            hiddenSizes[hiddenSizes.Length - 1],
            1,
            ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Identity)); // Reward prediction is unbounded
        
        _terminationOutputLayer = new FullyConnectedLayer<T>(
            hiddenSizes[hiddenSizes.Length - 1],
            1,
            ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Sigmoid)); // Termination probability between 0 and 1
        
        // For probabilistic models, add log variance output
        if (_probabilistic)
        {
            _logVarOutputLayer = new FullyConnectedLayer<T>(
                hiddenSizes[hiddenSizes.Length - 1],
                _stateSize,
                ActivationFunctionFactory<T>.CreateActivationFunction(ActivationFunction.Identity)); // Log variance can be any value
        }
    }
    
    /// <summary>
    /// Predicts the next state, reward, and termination probability given a state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to take.</param>
    /// <returns>A tuple containing the predicted next state, reward, and whether the episode will terminate.</returns>
    public (Tensor<T>, T, bool) Predict(Tensor<T> state, Vector<T> action)
    {
        // Convert action to tensor if needed
        Tensor<T> actionTensor = Tensor<T>.FromVector(action);
        
        // Forward pass through network
        var nextState = PredictNextState(state, actionTensor, false);
        var reward = PredictReward(state, actionTensor);
        var terminationProb = PredictTermination(state, actionTensor);
        
        // Convert termination probability to boolean
        bool terminated = false;
        if (_numOps.GreaterThan(terminationProb, _numOps.FromDouble(0.5)))
        {
            terminated = true;
        }
        
        return (nextState, reward, terminated);
    }
    
    /// <summary>
    /// Forward method for compatibility. Calls Predict internally.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to take.</param>
    /// <returns>A tuple containing the predicted next state, reward, and termination flag.</returns>
    public (Tensor<T>, T, bool) Forward(Tensor<T> state, Vector<T> action)
    {
        return Predict(state, action);
    }
    
    /// <summary>
    /// Predicts the next state given the current state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to take.</param>
    /// <param name="sampleNoise">Whether to sample from the distribution (for probabilistic models).</param>
    /// <returns>The predicted next state.</returns>
    public Tensor<T> PredictNextState(Tensor<T> state, Tensor<T> action, bool sampleNoise)
    {
        // Forward pass through network
        Tensor<T> stateEncoding = _stateInputLayers[0].Forward(state);
        Tensor<T> actionEncoding = _actionInputLayers[0].Forward(action);
        
        // Concatenate state and action encodings
        Tensor<T> combined = ConcatenateTensors(stateEncoding, actionEncoding);
        
        // Forward through hidden layers
        Tensor<T> hidden = combined;
        foreach (var layer in _hiddenLayers)
        {
            hidden = layer.Forward(hidden);
        }
        
        // Predict next state
        Tensor<T> nextState = _nextStateOutputLayer.Forward(hidden);
        
        // For probabilistic models, add noise based on predicted variance
        if (_probabilistic && sampleNoise && _logVarOutputLayer != null)
        {
            // Predict log variance
            Tensor<T> logVar = _logVarOutputLayer.Forward(hidden);
            
            // Sample noise from distribution
            Tensor<T> noise = SampleNoise(logVar);
            
            // Add noise to prediction
            nextState = AddTensors(nextState, noise);
        }
        
        // Add the prediction to the current state (residual prediction)
        return AddTensors(state, nextState);
    }
    
    /// <summary>
    /// Predicts the reward given the current state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to take.</param>
    /// <returns>The predicted reward.</returns>
    public T PredictReward(Tensor<T> state, Tensor<T> action)
    {
        // Forward pass through network
        Tensor<T> stateEncoding = _stateInputLayers[0].Forward(state);
        Tensor<T> actionEncoding = _actionInputLayers[0].Forward(action);
        
        // Concatenate state and action encodings
        Tensor<T> combined = ConcatenateTensors(stateEncoding, actionEncoding);
        
        // Forward through hidden layers
        Tensor<T> hidden = combined;
        foreach (var layer in _hiddenLayers)
        {
            hidden = layer.Forward(hidden);
        }
        
        // Predict reward
        Tensor<T> rewardTensor = _rewardOutputLayer.Forward(hidden);
        
        // Extract scalar reward
        return rewardTensor.Rank > 0 ? rewardTensor[0] : rewardTensor[0, 0];
    }
    
    /// <summary>
    /// Predicts the termination probability given the current state and action.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to take.</param>
    /// <returns>The predicted termination probability.</returns>
    public T PredictTermination(Tensor<T> state, Tensor<T> action)
    {
        // Forward pass through network
        Tensor<T> stateEncoding = _stateInputLayers[0].Forward(state);
        Tensor<T> actionEncoding = _actionInputLayers[0].Forward(action);
        
        // Concatenate state and action encodings
        Tensor<T> combined = ConcatenateTensors(stateEncoding, actionEncoding);
        
        // Forward through hidden layers
        Tensor<T> hidden = combined;
        foreach (var layer in _hiddenLayers)
        {
            hidden = layer.Forward(hidden);
        }
        
        // Predict termination probability
        Tensor<T> terminationTensor = _terminationOutputLayer.Forward(hidden);
        
        // Extract scalar probability
        return terminationTensor.Rank > 0 ? terminationTensor[0] : terminationTensor[0, 0];
    }
    
    /// <summary>
    /// Trains the model on a batch of experiences.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <param name="actions">Batch of actions.</param>
    /// <param name="rewards">Batch of rewards.</param>
    /// <param name="nextStates">Batch of next states.</param>
    /// <param name="dones">Batch of done flags.</param>
    /// <returns>The training loss.</returns>
    public T Train(
        Tensor<T>[] states,
        Vector<T>[] actions,
        T[] rewards,
        Tensor<T>[] nextStates,
        bool[] dones)
    {
        int batchSize = states.Length;
        T totalLoss = _numOps.Zero;
        
        // Convert actions to tensors
        var actionTensors = new Tensor<T>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            actionTensors[i] = Tensor<T>.FromVector(actions[i]);
        }
        
        // Calculate state deltas (targets are the changes in state)
        var stateDeltas = new Tensor<T>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            stateDeltas[i] = SubtractTensors(nextStates[i], states[i]);
        }
        
        // Process in smaller batches to avoid memory issues
        int miniBatchSize = 32;
        for (int batchStart = 0; batchStart < batchSize; batchStart += miniBatchSize)
        {
            int currentBatchSize = Math.Min(miniBatchSize, batchSize - batchStart);
            
            // Extract mini-batch
            var stateBatch = ExtractBatch(states, batchStart, currentBatchSize);
            var actionBatch = ExtractBatch(actionTensors, batchStart, currentBatchSize);
            var rewardBatch = ExtractBatch(rewards, batchStart, currentBatchSize);
            var doneBatch = ExtractBatch(dones, batchStart, currentBatchSize);
            var stateDeltaBatch = ExtractBatch(stateDeltas, batchStart, currentBatchSize);
            
            // Forward pass and collect losses/gradients
            T batchLoss = _numOps.Zero;
            var stateGradients = new List<Tensor<T>>();
            var rewardGradients = new List<T>();
            var termGradients = new List<T>();
            
            for (int i = 0; i < currentBatchSize; i++)
            {
                // Forward pass to get predictions
                var stateEncoding = _stateInputLayers[0].Forward(stateBatch[i]);
                var actionEncoding = _actionInputLayers[0].Forward(actionBatch[i]);
                var combined = ConcatenateTensors(stateEncoding, actionEncoding);
                
                var hidden = combined;
                foreach (var layer in _hiddenLayers)
                {
                    hidden = layer.Forward(hidden);
                }
                
                // Get predictions from output layers
                var predictedStateDelta = _nextStateOutputLayer.Forward(hidden);
                var predictedRewardTensor = _rewardOutputLayer.Forward(hidden);
                var predictedTermProbTensor = _terminationOutputLayer.Forward(hidden);
                
                // Extract scalar values
                T predictedReward = predictedRewardTensor.Rank > 0 ? predictedRewardTensor[0] : predictedRewardTensor[0, 0];
                T predictedTermProb = predictedTermProbTensor.Rank > 0 ? predictedTermProbTensor[0] : predictedTermProbTensor[0, 0];
                
                // Calculate state prediction loss gradient (MSE gradient)
                var stateGrad = new Tensor<T>(predictedStateDelta.Shape);
                for (int j = 0; j < stateGrad.Length; j++)
                {
                    var idx = stateGrad.GetIndexFromFlat(j);
                    T diff = _numOps.Subtract(predictedStateDelta[idx], stateDeltaBatch[i][idx]);
                    stateGrad[idx] = _numOps.Multiply(_numOps.FromDouble(2.0 / stateGrad.Length), diff);
                }
                stateGradients.Add(stateGrad);
                
                // Calculate reward prediction loss gradient (MSE gradient)
                T rewardDiff = _numOps.Subtract(predictedReward, rewardBatch[i]);
                T rewardGrad = _numOps.Multiply(_numOps.FromDouble(2.0 * 0.1), rewardDiff); // 0.1 is the reward loss weight
                rewardGradients.Add(rewardGrad);
                
                // Calculate termination prediction loss gradient (BCE gradient)
                T termTarget = doneBatch[i] ? _numOps.One : _numOps.Zero;
                T clippedProb = ClipValue(predictedTermProb, 1e-7, 1 - 1e-7);
                T termGrad = _numOps.Divide(
                    _numOps.Subtract(clippedProb, termTarget),
                    _numOps.Multiply(clippedProb, _numOps.Subtract(_numOps.One, clippedProb))
                );
                termGrad = _numOps.Multiply(_numOps.FromDouble(0.1), termGrad); // 0.1 is the termination loss weight
                termGradients.Add(termGrad);
                
                // Calculate losses for monitoring
                T stateLoss = CalculateMSE(predictedStateDelta, stateDeltaBatch[i]);
                T rewardLoss = CalculateScalarMSE(predictedReward, rewardBatch[i]);
                T termLoss = CalculateBCE(predictedTermProb, termTarget);
                
                // Combine losses
                T combinedLoss = _numOps.Zero;
                combinedLoss = _numOps.Add(combinedLoss, stateLoss);
                combinedLoss = _numOps.Add(combinedLoss, _numOps.Multiply(_numOps.FromDouble(0.1), rewardLoss));
                combinedLoss = _numOps.Add(combinedLoss, _numOps.Multiply(_numOps.FromDouble(0.1), termLoss));
                
                batchLoss = _numOps.Add(batchLoss, combinedLoss);
            }
            
            // Calculate average loss for this mini-batch
            batchLoss = _numOps.Divide(batchLoss, _numOps.FromDouble(currentBatchSize));
            
            // Backward pass and optimization
            BackwardPass(stateGradients, rewardGradients, termGradients, currentBatchSize);
            
            // Add to total loss
            totalLoss = _numOps.Add(totalLoss, batchLoss);
        }
        
        // Calculate average loss across all mini-batches
        return _numOps.Divide(totalLoss, _numOps.FromDouble(Math.Ceiling((double)batchSize / miniBatchSize)));
    }
    
    /// <summary>
    /// Performs a backward pass through the network and updates weights.
    /// </summary>
    /// <param name="stateGradients">Gradients with respect to state predictions.</param>
    /// <param name="rewardGradients">Gradients with respect to reward predictions.</param>
    /// <param name="termGradients">Gradients with respect to termination predictions.</param>
    /// <param name="batchSize">Size of the current batch.</param>
    private void BackwardPass(
        List<Tensor<T>> stateGradients,
        List<T> rewardGradients,
        List<T> termGradients,
        int batchSize)
    {
        // Initialize accumulated gradients for each layer
        var stateInputGrads = new List<Tensor<T>>();
        var actionInputGrads = new List<Tensor<T>>();
        var hiddenGrads = new List<Tensor<T>>();
        
        // Process each sample in the batch
        for (int i = 0; i < batchSize; i++)
        {
            // Backward through output layers
            var stateOutGrad = _nextStateOutputLayer.Backward(stateGradients[i]);
            
            // Create single-element tensors for scalar gradients
            var rewardGradTensor = new Tensor<T>(new int[] { 1 });
            rewardGradTensor[0] = rewardGradients[i];
            var rewardOutGrad = _rewardOutputLayer.Backward(rewardGradTensor);
            
            var termGradTensor = new Tensor<T>(new int[] { 1 });
            termGradTensor[0] = termGradients[i];
            var termOutGrad = _terminationOutputLayer.Backward(termGradTensor);
            
            // Combine gradients from all output layers
            var combinedGrad = AddTensors(stateOutGrad, AddTensors(rewardOutGrad, termOutGrad));
            
            // Backward through hidden layers in reverse order
            var currentGrad = combinedGrad;
            for (int j = _hiddenLayers.Length - 1; j >= 0; j--)
            {
                currentGrad = _hiddenLayers[j].Backward(currentGrad);
            }
            
            // Split gradient for state and action input layers
            int stateGradSize = _stateInputLayers[0].GetOutputShape()[0];
            var stateGrad = new Tensor<T>(new int[] { stateGradSize });
            var actionGrad = new Tensor<T>(new int[] { currentGrad.Length - stateGradSize });
            
            for (int j = 0; j < stateGradSize; j++)
            {
                stateGrad[j] = currentGrad[j];
            }
            for (int j = 0; j < actionGrad.Length; j++)
            {
                actionGrad[j] = currentGrad[stateGradSize + j];
            }
            
            // Backward through input layers
            _stateInputLayers[0].Backward(stateGrad);
            _actionInputLayers[0].Backward(actionGrad);
        }
        
        // Update parameters using gradient descent
        UpdateParameters();
    }
    
    /// <summary>
    /// Updates all layer parameters using accumulated gradients.
    /// </summary>
    private void UpdateParameters()
    {
        // Update state input layers
        foreach (var layer in _stateInputLayers)
        {
            layer.UpdateParameters(_learningRate);
        }
        
        // Update action input layers
        foreach (var layer in _actionInputLayers)
        {
            layer.UpdateParameters(_learningRate);
        }
        
        // Update hidden layers
        foreach (var layer in _hiddenLayers)
        {
            layer.UpdateParameters(_learningRate);
        }
        
        // Update output layers
        _nextStateOutputLayer.UpdateParameters(_learningRate);
        _rewardOutputLayer.UpdateParameters(_learningRate);
        _terminationOutputLayer.UpdateParameters(_learningRate);
        
        // Update log variance layer if probabilistic
        if (_probabilistic && _logVarOutputLayer != null)
        {
            _logVarOutputLayer.UpdateParameters(_learningRate);
        }
    }
    
    /// <summary>
    /// Gets all parameters of the model as a single vector.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
    public Vector<T> GetParameters()
    {
        var allParameters = new List<Vector<T>>();
        
        // Add state input layer parameters
        foreach (var layer in _stateInputLayers)
        {
            allParameters.Add(layer.GetParameters());
        }
        
        // Add action input layer parameters
        foreach (var layer in _actionInputLayers)
        {
            allParameters.Add(layer.GetParameters());
        }
        
        // Add hidden layer parameters
        foreach (var layer in _hiddenLayers)
        {
            allParameters.Add(layer.GetParameters());
        }
        
        // Add output layer parameters
        allParameters.Add(_nextStateOutputLayer.GetParameters());
        allParameters.Add(_rewardOutputLayer.GetParameters());
        allParameters.Add(_terminationOutputLayer.GetParameters());
        
        // For probabilistic models, add log variance output parameters
        if (_probabilistic && _logVarOutputLayer != null)
        {
            allParameters.Add(_logVarOutputLayer.GetParameters());
        }
        
        // Combine all parameters
        return ConcatenateVectors(allParameters);
    }
    
    /// <summary>
    /// Sets all parameters of the model from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        int index = 0;
        
        // Set state input layer parameters
        foreach (var layer in _stateInputLayers)
        {
            int layerParamSize = layer.GetParameters().Length;
            var layerParams = ExtractVector(parameters, index, layerParamSize);
            layer.SetParameters(layerParams);
            index += layerParamSize;
        }
        
        // Set action input layer parameters
        foreach (var layer in _actionInputLayers)
        {
            int layerParamSize = layer.GetParameters().Length;
            var layerParams = ExtractVector(parameters, index, layerParamSize);
            layer.SetParameters(layerParams);
            index += layerParamSize;
        }
        
        // Set hidden layer parameters
        foreach (var layer in _hiddenLayers)
        {
            int layerParamSize = layer.GetParameters().Length;
            var layerParams = ExtractVector(parameters, index, layerParamSize);
            layer.SetParameters(layerParams);
            index += layerParamSize;
        }
        
        // Set output layer parameters
        int stateOutputParamSize = _nextStateOutputLayer.GetParameters().Length;
        var stateOutputParams = ExtractVector(parameters, index, stateOutputParamSize);
        _nextStateOutputLayer.SetParameters(stateOutputParams);
        index += stateOutputParamSize;
        
        int rewardOutputParamSize = _rewardOutputLayer.GetParameters().Length;
        var rewardOutputParams = ExtractVector(parameters, index, rewardOutputParamSize);
        _rewardOutputLayer.SetParameters(rewardOutputParams);
        index += rewardOutputParamSize;
        
        int termOutputParamSize = _terminationOutputLayer.GetParameters().Length;
        var termOutputParams = ExtractVector(parameters, index, termOutputParamSize);
        _terminationOutputLayer.SetParameters(termOutputParams);
        index += termOutputParamSize;
        
        // For probabilistic models, set log variance output parameters
        if (_probabilistic && _logVarOutputLayer != null)
        {
            int logVarOutputParamSize = _logVarOutputLayer.GetParameters().Length;
            var logVarOutputParams = ExtractVector(parameters, index, logVarOutputParamSize);
            _logVarOutputLayer.SetParameters(logVarOutputParams);
        }
    }
    
    // Helper methods
    
    /// <summary>
    /// Samples noise based on log variance.
    /// </summary>
    /// <param name="logVar">The log variance tensor.</param>
    /// <returns>A noise tensor.</returns>
    private Tensor<T> SampleNoise(Tensor<T> logVar)
    {
        // Create noise tensor of same shape
        var noise = new Tensor<T>(logVar.Shape);
        
        // Sample standard normal noise
        var random = new Random();
        for (int i = 0; i < noise.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(); // Uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Box-Muller transform
            
            var index = noise.GetIndexFromFlat(i);
            noise[index] = _numOps.FromDouble(randStdNormal);
        }
        
        // Scale noise by standard deviation
        for (int i = 0; i < noise.Length; i++)
        {
            var index = noise.GetIndexFromFlat(i);
            // std = exp(log_var / 2)
            T logStd = _numOps.Multiply(logVar[index], _numOps.FromDouble(0.5));
            T std = _numOps.Exp(logStd);
            noise[index] = _numOps.Multiply(noise[index], std);
        }
        
        return noise;
    }
    
    /// <summary>
    /// Calculates Mean Squared Error between two tensors.
    /// </summary>
    private T CalculateMSE(Tensor<T> predicted, Tensor<T> target)
    {
        T sumSquaredError = _numOps.Zero;
        int count = 0;
        
        for (int i = 0; i < predicted.Length; i++)
        {
            var index = predicted.GetIndexFromFlat(i);
            T diff = _numOps.Subtract(predicted[index], target[index]);
            sumSquaredError = _numOps.Add(sumSquaredError, _numOps.Multiply(diff, diff));
            count++;
        }
        
        return _numOps.Divide(sumSquaredError, _numOps.FromDouble(count));
    }
    
    /// <summary>
    /// Calculates Mean Squared Error between two scalar values.
    /// </summary>
    private T CalculateScalarMSE(T predicted, T target)
    {
        T diff = _numOps.Subtract(predicted, target);
        return _numOps.Multiply(diff, diff);
    }
    
    /// <summary>
    /// Calculates Binary Cross Entropy loss.
    /// </summary>
    private T CalculateBCE(T predicted, T target)
    {
        // Clip prediction to avoid log(0) or log(1)
        predicted = ClipValue(predicted, 1e-7, 1 - 1e-7);
        
        // BCE = -target * log(predicted) - (1 - target) * log(1 - predicted)
        T logPred = _numOps.Log(predicted);
        T logOneMinusPred = _numOps.Log(_numOps.Subtract(_numOps.One, predicted));
        
        T firstTerm = _numOps.Multiply(target, logPred);
        T secondTerm = _numOps.Multiply(
            _numOps.Subtract(_numOps.One, target),
            logOneMinusPred);
            
        return _numOps.Negate(_numOps.Add(firstTerm, secondTerm));
    }
    
    /// <summary>
    /// Calculates KL divergence of log variance to standard normal prior.
    /// </summary>
    private T CalculateKLDivergence(Tensor<T> logVar)
    {
        T sumKL = _numOps.Zero;
        int count = 0;
        
        for (int i = 0; i < logVar.Length; i++)
        {
            var index = logVar.GetIndexFromFlat(i);
            
            // KL = 0.5 * (exp(log_var) + log_var - 1)
            T expLogVar = _numOps.Exp(logVar[index]);
            T term = _numOps.Add(expLogVar, logVar[index]);
            term = _numOps.Subtract(term, _numOps.One);
            term = _numOps.Multiply(_numOps.FromDouble(0.5), term);
            
            sumKL = _numOps.Add(sumKL, term);
            count++;
        }
        
        return _numOps.Divide(sumKL, _numOps.FromDouble(count));
    }
    
    /// <summary>
    /// Clips a value to the specified range.
    /// </summary>
    private T ClipValue(T value, double min, double max)
    {
        T minT = _numOps.FromDouble(min);
        T maxT = _numOps.FromDouble(max);
        
        if (_numOps.LessThan(value, minT))
        {
            return minT;
        }
        if (_numOps.GreaterThan(value, maxT))
        {
            return maxT;
        }
        return value;
    }
    
    /// <summary>
    /// Concatenates two tensors along the first dimension.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        // Simple implementation for 1D or flat tensors
        int aSize = a.Length;
        int bSize = b.Length;
        
        var result = new Tensor<T>(new int[] { aSize + bSize });
        
        for (int i = 0; i < aSize; i++)
        {
            var aIndex = a.GetIndexFromFlat(i);
            result[i] = a[aIndex];
        }
        
        for (int i = 0; i < bSize; i++)
        {
            var bIndex = b.GetIndexFromFlat(i);
            result[aSize + i] = b[bIndex];
        }
        
        return result;
    }
    
    /// <summary>
    /// Adds two tensors elementwise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        // Create result tensor with same shape
        var result = new Tensor<T>(a.Shape);
        
        // Add elements
        for (int i = 0; i < a.Length; i++)
        {
            var index = a.GetIndexFromFlat(i);
            result[index] = _numOps.Add(a[index], b[index]);
        }
        
        return result;
    }
    
    /// <summary>
    /// Subtracts two tensors elementwise.
    /// </summary>
    private Tensor<T> SubtractTensors(Tensor<T> a, Tensor<T> b)
    {
        // Create result tensor with same shape
        var result = new Tensor<T>(a.Shape);
        
        // Subtract elements
        for (int i = 0; i < a.Length; i++)
        {
            var index = a.GetIndexFromFlat(i);
            result[index] = _numOps.Subtract(a[index], b[index]);
        }
        
        return result;
    }
    
    /// <summary>
    /// Extracts a batch of tensors from an array.
    /// </summary>
    private Tensor<T>[] ExtractBatch(Tensor<T>[] array, int startIndex, int batchSize)
    {
        var result = new Tensor<T>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            result[i] = array[startIndex + i];
        }
        return result;
    }
    
    /// <summary>
    /// Extracts a batch of values from an array.
    /// </summary>
    private T[] ExtractBatch(T[] array, int startIndex, int batchSize)
    {
        var result = new T[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            result[i] = array[startIndex + i];
        }
        return result;
    }
    
    /// <summary>
    /// Extracts a batch of booleans from an array.
    /// </summary>
    private bool[] ExtractBatch(bool[] array, int startIndex, int batchSize)
    {
        var result = new bool[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            result[i] = array[startIndex + i];
        }
        return result;
    }
    
    /// <summary>
    /// Concatenates a list of vectors into a single vector.
    /// </summary>
    private Vector<T> ConcatenateVectors(List<Vector<T>> vectors)
    {
        // Calculate total length
        int totalLength = 0;
        foreach (var vector in vectors)
        {
            totalLength += vector.Length;
        }
        
        // Create result vector
        var result = new Vector<T>(totalLength);
        
        // Copy values
        int index = 0;
        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                result[index++] = vector[i];
            }
        }
        
        return result;
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
    /// Saves the dynamics model to a file.
    /// </summary>
    /// <param name="path">The file path to save to.</param>
    public void Save(string path)
    {
        var parameters = GetParameters();
        using (var writer = new BinaryWriter(File.Open(path, FileMode.Create)))
        {
            writer.Write(_stateSize);
            writer.Write(_actionSize);
            writer.Write(_probabilistic);
            writer.Write(_hiddenLayers.Length);
            foreach (var layer in _hiddenLayers)
            {
                writer.Write(layer.GetOutputShape()[0]);
            }
            writer.Write(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                writer.Write(Convert.ToDouble(parameters[i]));
            }
        }
    }
    
    /// <summary>
    /// Loads the dynamics model from a file.
    /// </summary>
    /// <param name="path">The file path to load from.</param>
    public void Load(string path)
    {
        using (var reader = new BinaryReader(File.Open(path, FileMode.Open)))
        {
            int stateSize = reader.ReadInt32();
            int actionSize = reader.ReadInt32();
            bool probabilistic = reader.ReadBoolean();
            int numHiddenLayers = reader.ReadInt32();
            
            if (stateSize != _stateSize || actionSize != _actionSize || probabilistic != _probabilistic)
            {
                throw new InvalidOperationException("Model dimensions or type do not match");
            }
            
            // Skip hidden layer sizes - we already have them
            for (int i = 0; i < numHiddenLayers; i++)
            {
                reader.ReadInt32();
            }
            
            int paramCount = reader.ReadInt32();
            var parameters = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                parameters[i] = _numOps.FromDouble(reader.ReadDouble());
            }
            
            SetParameters(parameters);
        }
    }
}