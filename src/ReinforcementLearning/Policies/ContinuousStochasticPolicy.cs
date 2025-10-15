using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Implements a stochastic policy for continuous action spaces using a neural network.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// This policy outputs means and standard deviations for a Gaussian distribution
    /// over continuous actions, and samples from this distribution to select actions.
    /// It's commonly used in Policy Gradient methods for continuous control tasks.
    /// </para>
    /// </remarks>
    public class ContinuousStochasticPolicy<T> : IStochasticPolicy<Tensor<T>, Vector<T>, T>, IPolicy<Tensor<T>, Vector<T>, T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        private readonly int _stateSize;
        private readonly int _actionSize;
        private readonly List<LayerBase<T>> _commonLayers; // Shared layers
        private readonly List<LayerBase<T>> _meanLayers; // Layers for action means
        private readonly List<LayerBase<T>> _stdDevLayers; // Layers for action standard deviations
        private readonly TanhActivation<T> _finalMeanActivation; // Tanh for bounding action means
        private readonly SoftPlusActivation<T> _finalStdDevActivation; // Softplus for positive std devs
        private readonly Random _random = default!;
        private readonly Vector<T> _actionLowerBound = default!;
        private readonly Vector<T> _actionUpperBound = default!;
        private readonly bool _learnStdDev;
        private readonly T _minStdDev = default!;
        private readonly T _maxStdDev = default!;

        /// <summary>
        /// Gets a value indicating whether the policy is stochastic.
        /// </summary>
        public bool IsStochastic => true;

        /// <summary>
        /// Gets a value indicating whether the policy is for continuous action spaces.
        /// </summary>
        public bool IsContinuous => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="ContinuousStochasticPolicy{T}"/> class.
        /// </summary>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space (dimensionality of the action vector).</param>
        /// <param name="hiddenSizes">The sizes of the hidden layers in the policy network.</param>
        /// <param name="activationFunction">The activation function to use for hidden layers.</param>
        /// <param name="actionLowerBound">The lower bound for each action dimension (typically -1).</param>
        /// <param name="actionUpperBound">The upper bound for each action dimension (typically 1).</param>
        /// <param name="learnStdDev">Whether to learn the standard deviation or use a fixed value.</param>
        /// <param name="initialStdDev">The initial standard deviation for each action dimension.</param>
        /// <param name="minStdDev">The minimum allowed standard deviation.</param>
        /// <param name="maxStdDev">The maximum allowed standard deviation.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        public ContinuousStochasticPolicy(
            int stateSize,
            int actionSize,
            int[] hiddenSizes,
            IActivationFunction<T>? activationFunction = null,
            Vector<T>? actionLowerBound = null,
            Vector<T>? actionUpperBound = null,
            bool learnStdDev = true,
            double initialStdDev = 0.5,
            double minStdDev = 0.01,
            double maxStdDev = 2.0,
            int? seed = null)
        {
            _stateSize = stateSize;
            _actionSize = actionSize;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _learnStdDev = learnStdDev;
            _minStdDev = NumOps.FromDouble(minStdDev);
            _maxStdDev = NumOps.FromDouble(maxStdDev);

            // Default action bounds if not provided
            _actionLowerBound = actionLowerBound ?? Vector<T>.CreateDefault(actionSize, NumOps.FromDouble(-1.0));
            _actionUpperBound = actionUpperBound ?? Vector<T>.CreateDefault(actionSize, NumOps.FromDouble(1.0));

            // Activation functions
            _finalMeanActivation = new TanhActivation<T>();
            _finalStdDevActivation = new SoftPlusActivation<T>();

            // Create the network layers
            _commonLayers = new List<LayerBase<T>>();
            _meanLayers = new List<LayerBase<T>>();
            _stdDevLayers = new List<LayerBase<T>>();

            // Common layers
            int currentSize = stateSize;
            for (int i = 0; i < hiddenSizes.Length - 1; i++)
            {
                _commonLayers.Add(new DenseLayer<T>(currentSize, hiddenSizes[i], activationFunction));
                currentSize = hiddenSizes[i];
            }

            // Final hidden layer for action means
            if (hiddenSizes.Length > 0)
            {
                _meanLayers.Add(new DenseLayer<T>(currentSize, hiddenSizes[hiddenSizes.Length - 1], activationFunction));
                currentSize = hiddenSizes[hiddenSizes.Length - 1];
            }

            // Output layer for action means (no activation, as tanh will be applied separately)
            _meanLayers.Add(new DenseLayer<T>(currentSize, actionSize, new IdentityActivation<T>() as IActivationFunction<T>));

            if (_learnStdDev)
            {
                // Reset current size to the output of common layers
                currentSize = hiddenSizes.Length > 0 ? hiddenSizes[hiddenSizes.Length - 2] : stateSize;

                // Final hidden layer for standard deviations
                if (hiddenSizes.Length > 0)
                {
                    _stdDevLayers.Add(new DenseLayer<T>(currentSize, hiddenSizes[hiddenSizes.Length - 1], activationFunction));
                    currentSize = hiddenSizes[hiddenSizes.Length - 1];
                }

                // Output layer for standard deviations (no activation, as softplus will be applied separately)
                _stdDevLayers.Add(new DenseLayer<T>(currentSize, actionSize, new IdentityActivation<T>() as IActivationFunction<T>));
            }
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <returns>The selected action.</returns>
        public Vector<T> SelectAction(Tensor<T> state)
        {
            // Get action mean and standard deviation
            var (mean, stdDev) = GetActionDistribution(state);

            // Sample action from Gaussian distribution
            return SampleFromGaussian(mean, stdDev);
        }

        /// <summary>
        /// Calculates the policy gradients for actor update.
        /// </summary>
        /// <param name="state">The state.</param>
        /// <param name="action">The action.</param>
        /// <param name="qValue">The Q-value.</param>
        /// <param name="entropyCoefficient">The entropy coefficient (alpha).</param>
        /// <returns>A tuple containing the gradients for action mean and log standard deviation.</returns>
        public (Vector<T> meanGradient, Vector<T> logStdGradient) CalculatePolicyGradients(
            Tensor<T> state, 
            Vector<T> action, 
            T qValue,
            T entropyCoefficient)
        {
            // Get the action distribution parameters
            var (mean, stdDev) = GetActionDistribution(state);
            
            // Initialize gradients
            var meanGradient = new Vector<T>(_actionSize);
            var logStdGradient = new Vector<T>(_actionSize);
            
            // Get the log probability of the action
            T logProb = LogProbability(state, action);
            
            for (int i = 0; i < _actionSize; i++)
            {
                // Compute gradient for the mean
                // For Gaussian policy: d(log p)/d(mean) = (x - mean) / variance
                T diff = NumOps.Subtract(action[i], mean[i]);
                T variance = NumOps.Multiply(stdDev[i], stdDev[i]);
                T meanGrad = NumOps.Divide(diff, variance);
                
                // Compute gradient for the log standard deviation
                // For Gaussian policy: d(log p)/d(log_std) = ((x - mean)^2 / variance - 1)
                T logStdGrad = NumOps.Subtract(
                    NumOps.Divide(NumOps.Multiply(diff, diff), variance),
                    NumOps.One);
                
                // Scale by entropy coefficient for SAC-style updates
                // For maximizing Q - alpha*logP, gradients are:
                // gradient = alpha * gradient(logP) - gradient(Q)
                meanGradient[i] = NumOps.Multiply(entropyCoefficient, meanGrad);
                logStdGradient[i] = NumOps.Multiply(entropyCoefficient, logStdGrad);
                
                // Note: In a real implementation, we would also incorporate the Q-value gradients
                // which would come from the critic network's action gradients.
            }
            
            return (meanGradient, logStdGradient);
        }
        
        /// <summary>
        /// Selects a deterministic action (the mean of the policy) for evaluation.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <returns>The deterministic action.</returns>
        public Vector<T> SelectDeterministicAction(Tensor<T> state)
        {
            // Get the mean of the action distribution without sampling
            var (mean, _) = GetActionDistribution(state);
            return mean;
        }

        /// <summary>
        /// Calculates the log probability of taking a specific action in a given state.
        /// </summary>
        /// <param name="state">The state in which the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <returns>The log probability of the action.</returns>
        public T LogProbability(Tensor<T> state, Vector<T> action)
        {
            var (mean, stdDev) = GetActionDistribution(state);
            
            // Initialize log probability
            T logProb = NumOps.Zero;
            T twoPi = NumOps.FromDouble(2.0 * Math.PI);
            
            // Calculate log probability for each action dimension
            for (int i = 0; i < _actionSize; i++)
            {
                // Gaussian log probability: -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²
                T diff = NumOps.Subtract(action[i], mean[i]);
                T variance = NumOps.Multiply(stdDev[i], stdDev[i]);
                T term1 = NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.Log(twoPi));
                T term2 = NumOps.Negate(NumOps.Log(stdDev[i]));
                T term3 = NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.Divide(NumOps.Multiply(diff, diff), variance));
                logProb = NumOps.Add(logProb, NumOps.Add(NumOps.Add(term1, term2), term3));
            }
            
            return logProb;
        }

        /// <summary>
        /// Updates the policy parameters using the provided gradients.
        /// </summary>
        /// <param name="gradients">The gradients for the policy parameters.</param>
        /// <param name="learningRate">The learning rate for the update.</param>
        public void UpdateParameters((List<Tensor<T>> commonGrads, List<Tensor<T>> meanGrads, List<Tensor<T>> stdDevGrads) gradients, T learningRate)
        {
            // Update common layers
            for (int i = 0; i < _commonLayers.Count; i++)
            {
                if (_commonLayers[i] is DenseLayer<T> denseLayer)
                {
                    denseLayer.UpdateParameters(learningRate);
                }
            }

            // Update mean layers
            for (int i = 0; i < _meanLayers.Count; i++)
            {
                if (_meanLayers[i] is DenseLayer<T> denseLayer)
                {
                    denseLayer.UpdateParameters(learningRate);
                }
            }

            // Update std dev layers if learning standard deviations
            if (_learnStdDev)
            {
                for (int i = 0; i < _stdDevLayers.Count; i++)
                {
                    if (_stdDevLayers[i] is DenseLayer<T> denseLayer)
                    {
                        denseLayer.UpdateParameters(learningRate);
                    }
                }
            }
        }
        
        /// <summary>
        /// Updates the policy parameters using policy gradients with entropy regularization.
        /// </summary>
        /// <param name="policyGradients">A list of state-action-gradient tuples.</param>
        /// <param name="useGradientClipping">Whether to clip gradients to prevent large updates.</param>
        /// <param name="maxGradientNorm">The maximum gradient norm for clipping.</param>
        public void UpdateParameters(List<(Tensor<T> state, Vector<T> meanGradient, Vector<T> logStdGradient)> policyGradients, bool useGradientClipping, T maxGradientNorm)
        {
            if (policyGradients == null || policyGradients.Count == 0)
                return;
            
            // Learning rate for this update
            T learningRate = NumOps.FromDouble(0.001);
            
            // Process each gradient tuple
            foreach (var (state, meanGradient, logStdGradient) in policyGradients)
            {
                // Perform forward pass to set up the network for gradient application
                var (mean, stdDev) = GetActionDistribution(state);
                
                // Apply gradients to mean parameters
                for (int i = 0; i < _meanLayers.Count; i++)
                {
                    if (_meanLayers[i] is DenseLayer<T> denseLayer)
                    {
                        // In a real implementation, we would compute the gradient for each layer
                        // by backpropagating the gradients through the network
                        // For simplicity, we'll just update the final layer directly
                        if (i == _meanLayers.Count - 1)
                        {
                            // Clip gradients if needed
                            Vector<T> clippedGradients = meanGradient;
                            if (useGradientClipping)
                            {
                                T gradNorm = NumOps.Zero;
                                for (int j = 0; j < meanGradient.Length; j++)
                                {
                                    gradNorm = NumOps.Add(gradNorm, 
                                               NumOps.Multiply(meanGradient[j], meanGradient[j]));
                                }
                                gradNorm = NumOps.Sqrt(gradNorm);
                                
                                if (NumOps.GreaterThan(gradNorm, maxGradientNorm))
                                {
                                    // Scale the gradients
                                    T scale = NumOps.Divide(maxGradientNorm, gradNorm);
                                    for (int j = 0; j < meanGradient.Length; j++)
                                    {
                                        clippedGradients[j] = NumOps.Multiply(meanGradient[j], scale);
                                    }
                                }
                            }
                            
                            // Apply gradient to this layer
                            var layerGradients = denseLayer.GetParameterGradients();
                            for (int j = 0; j < layerGradients.Length; j++)
                            {
                                // Here we would compute the actual gradients per weight
                                // This is simplified for demonstration purposes
                                if (j < clippedGradients.Length)
                                {
                                    layerGradients[j] = clippedGradients[j];
                                }
                            }
                            
                            denseLayer.UpdateParameters(learningRate);
                        }
                    }
                }
                
                // Only update standard deviation layers if learning them
                if (_learnStdDev)
                {
                    // Similar logic for standard deviation layers
                    for (int i = 0; i < _stdDevLayers.Count; i++)
                    {
                        if (_stdDevLayers[i] is DenseLayer<T> denseLayer)
                        {
                            // Only update the final layer directly in this simplified example
                            if (i == _stdDevLayers.Count - 1)
                            {
                                // Clip gradients if needed
                                Vector<T> clippedGradients = logStdGradient;
                                if (useGradientClipping)
                                {
                                    T gradNorm = NumOps.Zero;
                                    for (int j = 0; j < logStdGradient.Length; j++)
                                    {
                                        gradNorm = NumOps.Add(gradNorm, 
                                                   NumOps.Multiply(logStdGradient[j], logStdGradient[j]));
                                    }
                                    gradNorm = NumOps.Sqrt(gradNorm);
                                    
                                    if (NumOps.GreaterThan(gradNorm, maxGradientNorm))
                                    {
                                        // Scale the gradients
                                        T scale = NumOps.Divide(maxGradientNorm, gradNorm);
                                        for (int j = 0; j < logStdGradient.Length; j++)
                                        {
                                            clippedGradients[j] = NumOps.Multiply(logStdGradient[j], scale);
                                        }
                                    }
                                }
                                
                                // Apply gradient to this layer
                                var layerGradients = denseLayer.GetParameterGradients();
                                for (int j = 0; j < layerGradients.Length; j++)
                                {
                                    // Here we would compute the actual gradients per weight
                                    // This is simplified for demonstration purposes
                                    if (j < clippedGradients.Length)
                                    {
                                        layerGradients[j] = clippedGradients[j];
                                    }
                                }
                                
                                denseLayer.UpdateParameters(learningRate);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Gets the entropy of the policy for a given state.
        /// </summary>
        /// <param name="state">The state for which to calculate the entropy.</param>
        /// <returns>The entropy value.</returns>
        public T GetEntropy(Tensor<T> state)
        {
            var (_, stdDev) = GetActionDistribution(state);
            
            // Entropy of a Gaussian is 0.5 * log(2πe * σ²)
            T entropy = NumOps.Zero;
            T twoPI_e = NumOps.FromDouble(2.0 * Math.PI * Math.E); // 2πe
            
            for (int i = 0; i < _actionSize; i++)
            {
                T variance = NumOps.Multiply(stdDev[i], stdDev[i]);
                entropy = NumOps.Add(entropy, NumOps.Multiply(NumOps.FromDouble(0.5), 
                           NumOps.Log(NumOps.Multiply(twoPI_e, variance))));
            }
            
            return entropy;
        }

        /// <summary>
        /// Performs a forward pass through the policy network to get the action distribution parameters.
        /// </summary>
        /// <param name="state">The state to evaluate.</param>
        /// <returns>A tuple containing the mean and standard deviation vectors for the action distribution.</returns>
        private (Vector<T> mean, Vector<T> stdDev) GetActionDistribution(Tensor<T> state)
        {
            // Forward pass through the common layers
            var current = state;
            foreach (var layer in _commonLayers)
            {
                current = layer.Forward(current);
            }

            // Forward pass through the mean layers
            var meanOutput = current;
            foreach (var layer in _meanLayers)
            {
                meanOutput = layer.Forward(meanOutput);
            }

            // Ensure the mean output is a vector
            if (meanOutput.Rank != 1)
            {
                throw new InvalidOperationException("Expected mean network output to be a vector");
            }

            // Apply tanh to bound the mean in [-1, 1]
            var meanVector = _finalMeanActivation.Activate(meanOutput.ToVector());

            // Scale the mean to the action bounds
            for (int i = 0; i < _actionSize; i++)
            {
                // Transform from [-1, 1] to [lower, upper]
                T scaledMean = ScaleAction(meanVector[i], i);
                meanVector[i] = scaledMean;
            }

            Vector<T> stdDevVector;

            if (_learnStdDev)
            {
                // Forward pass through the std dev layers
                var stdDevOutput = current;
                foreach (var layer in _stdDevLayers)
                {
                    stdDevOutput = layer.Forward(stdDevOutput);
                }

                // Ensure the std dev output is a vector
                if (stdDevOutput.Rank != 1)
                {
                    throw new InvalidOperationException("Expected std dev network output to be a vector");
                }

                // Apply softplus to ensure positive std devs and clamp to allowed range
                stdDevVector = _finalStdDevActivation.Activate(stdDevOutput.ToVector());
                for (int i = 0; i < _actionSize; i++)
                {
                    stdDevVector[i] = MathHelper.Min(MathHelper.Max(stdDevVector[i], _minStdDev), _maxStdDev);
                }
            }
            else
            {
                // Use fixed standard deviation
                stdDevVector = Vector<T>.CreateDefault(_actionSize, NumOps.FromDouble(0.5));
            }

            return (meanVector, stdDevVector);
        }

        /// <summary>
        /// Samples an action from a Gaussian distribution.
        /// </summary>
        /// <param name="mean">The mean vector of the distribution.</param>
        /// <param name="stdDev">The standard deviation vector of the distribution.</param>
        /// <returns>The sampled action vector.</returns>
        private Vector<T> SampleFromGaussian(Vector<T> mean, Vector<T> stdDev)
        {
            var action = new Vector<T>(mean.Length);

            for (int i = 0; i < mean.Length; i++)
            {
                // Box-Muller transform to sample from standard normal distribution
                double u1 = 1.0 - _random.NextDouble(); // Uniform(0,1) sample
                double u2 = 1.0 - _random.NextDouble(); // Uniform(0,1) sample
                double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // Standard normal sample
                
                // Scale by standard deviation and add mean
                T sample = NumOps.Add(mean[i], NumOps.Multiply(stdDev[i], NumOps.FromDouble(z)));
                
                // Clip to action bounds
                T lowerBound = _actionLowerBound[i];
                T upperBound = _actionUpperBound[i];
                action[i] = MathHelper.Min(MathHelper.Max(sample, lowerBound), upperBound);
            }

            return action;
        }

        /// <summary>
        /// Scales an action from the tanh output range [-1, 1] to the specified action bounds.
        /// </summary>
        /// <param name="actionValue">The action value in the range [-1, 1].</param>
        /// <param name="actionIndex">The index of the action dimension.</param>
        /// <returns>The scaled action value in the range [lowerBound, upperBound].</returns>
        private T ScaleAction(T actionValue, int actionIndex)
        {
            T lowerBound = _actionLowerBound[actionIndex];
            T upperBound = _actionUpperBound[actionIndex];
            
            // Linear scaling from [-1, 1] to [lowerBound, upperBound]
            // x' = lowerBound + (x + 1) * (upperBound - lowerBound) / 2
            return NumOps.Add(lowerBound, 
                  NumOps.Divide(
                    NumOps.Multiply(
                      NumOps.Add(actionValue, NumOps.One), 
                      NumOps.Subtract(upperBound, lowerBound)), 
                    NumOps.FromDouble(2.0)));
        }

        /// <summary>
        /// Gets the parameters of the policy network.
        /// </summary>
        /// <returns>A tuple of parameter lists for common, mean, and stdDev layers.</returns>
        public (List<Vector<T>> common, List<Vector<T>> mean, List<Vector<T>> stdDev) GetParameters()
        {
            var commonParams = new List<Vector<T>>();
            var meanParams = new List<Vector<T>>();
            var stdDevParams = new List<Vector<T>>();

            foreach (var layer in _commonLayers)
            {
                if (layer is DenseLayer<T> denseLayer)
                {
                    commonParams.Add(denseLayer.GetParameters());
                }
            }

            foreach (var layer in _meanLayers)
            {
                if (layer is DenseLayer<T> denseLayer)
                {
                    meanParams.Add(denseLayer.GetParameters());
                }
            }

            if (_learnStdDev)
            {
                foreach (var layer in _stdDevLayers)
                {
                    if (layer is DenseLayer<T> denseLayer)
                    {
                        stdDevParams.Add(denseLayer.GetParameters());
                    }
                }
            }

            return (commonParams, meanParams, stdDevParams);
        }

        /// <summary>
        /// Sets the parameters of the policy network.
        /// </summary>
        /// <param name="parameters">A tuple of parameter lists for common, mean, and stdDev layers.</param>
        public void SetParameters((List<Vector<T>> common, List<Vector<T>> mean, List<Vector<T>> stdDev) parameters)
        {
            var (commonParams, meanParams, stdDevParams) = parameters;

            if (commonParams.Count != _commonLayers.Count)
            {
                throw new ArgumentException($"Expected {_commonLayers.Count} common parameter tensors, got {commonParams.Count}");
            }

            if (meanParams.Count != _meanLayers.Count)
            {
                throw new ArgumentException($"Expected {_meanLayers.Count} mean parameter tensors, got {meanParams.Count}");
            }

            if (_learnStdDev && stdDevParams.Count != _stdDevLayers.Count)
            {
                throw new ArgumentException($"Expected {_stdDevLayers.Count} stdDev parameter tensors, got {stdDevParams.Count}");
            }

            for (int i = 0; i < _commonLayers.Count; i++)
            {
                if (_commonLayers[i] is DenseLayer<T> denseLayer)
                {
                    denseLayer.SetParameters(commonParams[i]);
                }
            }

            for (int i = 0; i < _meanLayers.Count; i++)
            {
                if (_meanLayers[i] is DenseLayer<T> denseLayer)
                {
                    denseLayer.SetParameters(meanParams[i]);
                }
            }

            if (_learnStdDev)
            {
                for (int i = 0; i < _stdDevLayers.Count; i++)
                {
                    if (_stdDevLayers[i] is DenseLayer<T> denseLayer)
                    {
                        denseLayer.SetParameters(stdDevParams[i]);
                    }
                }
            }
        }
        
        /// <summary>
        /// Copies the parameters from another stochastic policy.
        /// </summary>
        /// <param name="source">The source policy from which to copy parameters.</param>
        public void CopyParametersFrom(IStochasticPolicy<Tensor<T>, Vector<T>, T> source)
        {
            if (source is ContinuousStochasticPolicy<T> otherPolicy)
            {
                // Get parameters from the source policy
                var sourceParams = otherPolicy.GetParameters();
                
                // Set our parameters to match
                SetParameters(sourceParams);
            }
            else
            {
                throw new ArgumentException("Source policy must be a ContinuousStochasticPolicy");
            }
        }
        
        /// <summary>
        /// Evaluates the policy for a given state and returns action means and standard deviations.
        /// </summary>
        /// <param name="state">The state to evaluate.</param>
        /// <returns>Tuple of mean and standard deviation as object.</returns>
        public object EvaluatePolicy(Tensor<T> state)
        {
            return GetActionDistribution(state);
        }
        
        /// <summary>
        /// Updates the policy parameters using the provided gradients.
        /// </summary>
        /// <param name="gradients">The gradients for the policy parameters.</param>
        /// <param name="learningRate">The learning rate for the update.</param>
        public void UpdateParameters(object gradients, T learningRate)
        {
            if (gradients is ValueTuple<List<Tensor<T>>, List<Tensor<T>>, List<Tensor<T>>> typedGradients)
            {
                UpdateParameters(typedGradients, learningRate);
            }
            else if (gradients is List<(Tensor<T> state, Vector<T> meanGradient, Vector<T> logStdGradient)> policyGradients)
            {
                UpdateParameters(policyGradients, true, NumOps.FromDouble(1.0));
            }
            else
            {
                throw new ArgumentException("Unsupported gradient format");
            }
        }
        
        /// <summary>
        /// Performs a soft update of parameters from another stochastic policy.
        /// </summary>
        /// <param name="source">The source policy from which to update parameters.</param>
        /// <param name="tau">The soft update factor (between 0 and 1).</param>
        public void SoftUpdate(IStochasticPolicy<Tensor<T>, Vector<T>, T> source, T tau)
        {
            if (source is ContinuousStochasticPolicy<T> otherPolicy)
            {
                // Get parameters from both policies
                var (commonParams, meanParams, stdDevParams) = GetParameters();
                var (otherCommonParams, otherMeanParams, otherStdDevParams) = otherPolicy.GetParameters();
                
                // Ensure the parameter lists have the same length
                if (commonParams.Count != otherCommonParams.Count ||
                    meanParams.Count != otherMeanParams.Count ||
                    (_learnStdDev && stdDevParams.Count != otherStdDevParams.Count))
                {
                    throw new ArgumentException("Policy parameter structures don't match for soft update");
                }
                
                // Perform soft update on common layer parameters
                for (int i = 0; i < commonParams.Count; i++)
                {
                    var targetParams = commonParams[i];
                    var sourceParams = otherCommonParams[i];
                    
                    for (int j = 0; j < targetParams.Length; j++)
                    {
                        // target_params = (1 - tau) * target_params + tau * source_params
                        targetParams[j] = NumOps.Add(
                            NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), targetParams[j]),
                            NumOps.Multiply(tau, sourceParams[j]));
                    }
                }
                
                // Perform soft update on mean layer parameters
                for (int i = 0; i < meanParams.Count; i++)
                {
                    var targetParams = meanParams[i];
                    var sourceParams = otherMeanParams[i];
                    
                    for (int j = 0; j < targetParams.Length; j++)
                    {
                        targetParams[j] = NumOps.Add(
                            NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), targetParams[j]),
                            NumOps.Multiply(tau, sourceParams[j]));
                    }
                }
                
                // Perform soft update on std dev layer parameters if learning them
                if (_learnStdDev)
                {
                    for (int i = 0; i < stdDevParams.Count; i++)
                    {
                        var targetParams = stdDevParams[i];
                        var sourceParams = otherStdDevParams[i];
                        
                        for (int j = 0; j < targetParams.Length; j++)
                        {
                            targetParams[j] = NumOps.Add(
                                NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), targetParams[j]),
                                NumOps.Multiply(tau, sourceParams[j]));
                        }
                    }
                }
                
                // Apply the updated parameters to our network
                SetParameters((commonParams, meanParams, stdDevParams));
            }
            else
            {
                throw new ArgumentException("Source policy must be a ContinuousStochasticPolicy");
            }
        }
        
        /// <summary>
        /// Gets all parameters of the stochastic policy as a flattened vector.
        /// </summary>
        /// <returns>A vector containing all parameters of the policy.</returns>
        Vector<T> IStochasticPolicy<Tensor<T>, Vector<T>, T>.GetParameters()
        {
            var allParameters = new List<T>();
            var (commonParams, meanParams, stdDevParams) = GetParameters();
            
            // Flatten common parameters
            foreach (var paramVector in commonParams)
            {
                for (int i = 0; i < paramVector.Length; i++)
                {
                    allParameters.Add(paramVector[i]);
                }
            }
            
            // Flatten mean parameters
            foreach (var paramVector in meanParams)
            {
                for (int i = 0; i < paramVector.Length; i++)
                {
                    allParameters.Add(paramVector[i]);
                }
            }
            
            // Flatten stdDev parameters if learned
            if (_learnStdDev)
            {
                foreach (var paramVector in stdDevParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
            }
            
            return new Vector<T>([.. allParameters]);
        }
        
        /// <summary>
        /// Sets the parameters of the stochastic policy from a flattened vector.
        /// </summary>
        /// <param name="parameters">The parameters to set.</param>
        void IStochasticPolicy<Tensor<T>, Vector<T>, T>.SetParameters(Vector<T> parameters)
        {
            var (commonParams, meanParams, stdDevParams) = GetParameters();
            int offset = 0;
            
            // Reconstruct common parameters
            var newCommonParams = new List<Vector<T>>();
            foreach (var paramVector in commonParams)
            {
                var newParams = new Vector<T>(paramVector.Length);
                for (int i = 0; i < paramVector.Length; i++)
                {
                    newParams[i] = parameters[offset++];
                }
                newCommonParams.Add(newParams);
            }
            
            // Reconstruct mean parameters
            var newMeanParams = new List<Vector<T>>();
            foreach (var paramVector in meanParams)
            {
                var newParams = new Vector<T>(paramVector.Length);
                for (int i = 0; i < paramVector.Length; i++)
                {
                    newParams[i] = parameters[offset++];
                }
                newMeanParams.Add(newParams);
            }
            
            // Reconstruct stdDev parameters if learned
            var newStdDevParams = new List<Vector<T>>();
            if (_learnStdDev)
            {
                foreach (var paramVector in stdDevParams)
                {
                    var newParams = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newParams[i] = parameters[offset++];
                    }
                    newStdDevParams.Add(newParams);
                }
            }
            
            // Check if we used all parameters
            if (offset != parameters.Length)
            {
                throw new ArgumentException($"Parameter count mismatch. Expected {offset} parameters, got {parameters.Length}");
            }
            
            // Apply the parameters
            SetParameters((newCommonParams, newMeanParams, newStdDevParams));
        }
    }
}