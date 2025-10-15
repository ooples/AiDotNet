global using AiDotNet.ReinforcementLearning.Policies;

namespace AiDotNet.ReinforcementLearning.Agents
{
    /// <summary>
    /// Implements the REINFORCE algorithm (Monte Carlo Policy Gradient) for reinforcement learning.
    /// </summary>
    /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// REINFORCE is a policy gradient method that directly optimizes the policy by estimating
    /// the gradient of expected returns using Monte Carlo sampling. It collects complete episodes
    /// before updating the policy, making it suitable for episodic tasks.
    /// </para>
    /// </remarks>
    public class REINFORCEAgent<TState, TAction, T> : AgentBase<TState, TAction, T>
        where TState : Tensor<T>
    {
        private readonly IPolicy<TState, TAction, T> _policy = default!;
        private readonly IValueFunction<TState, T>? _baseline;
        private readonly T _learningRate = default!;
        private readonly T _baselineLearningRate = default!;
        private readonly T _entropyCoefficient = default!;
        private readonly bool _useBaseline;
        private readonly bool _normalizeReturns;
        private readonly bool _standardizeRewards;

        private readonly List<(TState state, TAction action, T reward, bool done)> _episode;
        private readonly List<(TState, TAction, T, TState, bool)> _memory;
        private readonly RandomNumberGenerator<T> _rng = default!;

        /// <summary>
        /// Initializes a new instance of the <see cref="REINFORCEAgent{TState, TAction, T}"/> class.
        /// </summary>
        /// <param name="options">The options for the REINFORCE algorithm.</param>
        public REINFORCEAgent(PolicyGradientOptions options)
            : base(options.Gamma, 0.0, options.BatchSize, options.Seed) // REINFORCE doesn't use tau
        {
            _learningRate = NumOps.FromDouble(options.PolicyLearningRate);
            _baselineLearningRate = NumOps.FromDouble(options.ValueLearningRate);
            _entropyCoefficient = NumOps.FromDouble(options.EntropyCoefficient);
            _useBaseline = options.UseBaseline;
            _normalizeReturns = options.NormalizeAdvantages;
            _standardizeRewards = options.StandardizeRewards;
            _episode = new List<(TState, TAction, T, bool)>();
            _memory = new List<(TState, TAction, T, TState, bool)>();
            _rng = new RandomNumberGenerator<T>(options.Seed);

            // Create policy based on action space type
            if (!options.IsContinuous)
            {
                // Discrete action space
                if (typeof(TAction) != typeof(int))
                {
                    throw new ArgumentException("For discrete action spaces, TAction must be int");
                }

                _policy = (IPolicy<TState, TAction, T>)new DiscreteStochasticPolicy<T>(
                    options.StateSize,
                    options.ActionSize,
                    options.PolicyNetworkArchitecture,
                    ActivationFunctionFactory<T>.CreateActivationFunction(options.PolicyActivationFunction),
                    options.Seed);
            }
            else
            {
                // Continuous action space
                if (typeof(TAction) != typeof(Vector<T>))
                {
                    throw new ArgumentException("For continuous action spaces, TAction must be Vector<T>");
                }

                _policy = (IPolicy<TState, TAction, T>)new ContinuousStochasticPolicy<T>(
                    options.StateSize,
                    options.ActionSize,
                    options.PolicyNetworkArchitecture,
                    ActivationFunctionFactory<T>.CreateActivationFunction(options.PolicyActivationFunction),
                    null, // Default action bounds [-1, 1]
                    null,
                    true, // Learn standard deviations
                    options.InitialPolicyStdDev,
                    0.01, // Min std dev
                    2.0,  // Max std dev
                    options.Seed);
            }

            // Create baseline value function if using baseline
            if (_useBaseline)
            {
                _baseline = new ValueNetwork<T>(
                    options.StateSize,
                    options.ValueNetworkArchitecture,
                    options.ValueActivationFunction,
                    options.Seed);
            }
            else
            {
                _baseline = null;
            }
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="isTraining">A flag indicating whether the agent is in training mode.</param>
        /// <returns>The selected action.</returns>
        public override TAction SelectAction(TState state, bool isTraining = true)
        {
            // Always use the policy to select actions
            return _policy.SelectAction(state);
        }

        /// <summary>
        /// Updates the agent's knowledge based on an experience tuple.
        /// </summary>
        /// <param name="state">The state before the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <param name="reward">The reward received after taking the action.</param>
        /// <param name="nextState">The state after the action was taken.</param>
        /// <param name="done">A flag indicating whether the episode ended after this action.</param>
        public override void Learn(TState state, TAction action, T reward, TState nextState, bool done)
        {
            if (!IsTraining)
                return;

            // Store the transition in the episode buffer
            _episode.Add((state, action, reward, done));

            // If the episode is complete, process it
            if (done)
            {
                ProcessEpisode();
            }
        }

        /// <summary>
        /// Processes a completed episode for learning.
        /// </summary>
        private void ProcessEpisode()
        {
            if (_episode.Count == 0)
                return;

            // Calculate returns for each step in the episode
            var returns = CalculateReturns();

            // Standardize rewards if enabled
            if (_standardizeRewards)
            {
                StandardizeReturns(returns);
            }

            // Process baseline value function for advantage estimation
            Vector<T>? advantages = null;
            if (_useBaseline && _baseline != null)
            {
                advantages = new Vector<T>(_episode.Count);
                var states = new TState[_episode.Count];
                var targetValues = new Vector<T>(_episode.Count);

                for (int i = 0; i < _episode.Count; i++)
                {
                    states[i] = _episode[i].state;
                    targetValues[i] = returns[i];
                }

                // Update the baseline value function
                _baseline.Update(states, targetValues);

                // Calculate advantages
                for (int i = 0; i < _episode.Count; i++)
                {
                    T baselineValue = _baseline.PredictValue(_episode[i].state);
                    advantages[i] = NumOps.Subtract(returns[i], baselineValue);
                }

                // Normalize advantages if enabled
                if (_normalizeReturns)
                {
                    NormalizeVector(advantages);
                }
            }
            else
            {
                // If no baseline, use returns directly as advantages
                advantages = returns;
                
                // Normalize returns if enabled
                if (_normalizeReturns)
                {
                    NormalizeVector(advantages);
                }
            }

            // Update policy using the calculated advantages
            UpdatePolicy(advantages);

            // Clear the episode buffer
            _episode.Clear();
        }

        /// <summary>
        /// Calculates returns for each step in the episode using discounted rewards.
        /// </summary>
        /// <returns>A vector of returns for each step in the episode.</returns>
        private Vector<T> CalculateReturns()
        {
            var returns = new Vector<T>(_episode.Count);
            T discountedReturn = NumOps.Zero;

            // Calculate returns in reverse order
            for (int i = _episode.Count - 1; i >= 0; i--)
            {
                T reward = _episode[i].reward;
                
                // G_t = r_t + gamma * G_{t+1}
                discountedReturn = NumOps.Add(reward, NumOps.Multiply(Gamma, discountedReturn));
                returns[i] = discountedReturn;
            }

            return returns;
        }

        /// <summary>
        /// Standardizes returns by subtracting the mean and dividing by the standard deviation.
        /// </summary>
        /// <param name="returns">The vector of returns to standardize.</param>
        private void StandardizeReturns(Vector<T> returns)
        {
            NormalizeVector(returns);
        }

        /// <summary>
        /// Normalizes a vector by subtracting the mean and dividing by the standard deviation.
        /// </summary>
        /// <param name="vector">The vector to normalize.</param>
        private void NormalizeVector(Vector<T> vector)
        {
            // Calculate mean
            T sum = NumOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                sum = NumOps.Add(sum, vector[i]);
            }
            T mean = NumOps.Divide(sum, NumOps.FromDouble(vector.Length));

            // Calculate standard deviation
            T sumSquaredDiff = NumOps.Zero;
            for (int i = 0; i < vector.Length; i++)
            {
                T diff = NumOps.Subtract(vector[i], mean);
                sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
            }
            T stdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(vector.Length)));

            // Add small epsilon to avoid division by zero
            T epsilon = NumOps.FromDouble(1e-8);
            if (NumOps.LessThan(stdDev, epsilon))
            {
                stdDev = epsilon;
            }

            // Normalize
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = NumOps.Divide(NumOps.Subtract(vector[i], mean), stdDev);
            }
        }

        /// <summary>
        /// Updates the policy based on the calculated advantages.
        /// </summary>
        /// <param name="advantages">The advantages used for policy gradient estimation.</param>
        private void UpdatePolicy(Vector<T> advantages)
        {
            List<object> gradients = new List<object>();

            // Process each transition in the episode
            for (int i = 0; i < _episode.Count; i++)
            {
                var (state, action, _, _) = _episode[i];
                T advantage = advantages[i];

                // Calculate log probability of the action
                T logProb = _policy.LogProbability(state, action);

                // Calculate policy gradient
                T gradientScale = advantage;

                // Add entropy term if enabled
                if (NumOps.GreaterThan(_entropyCoefficient, NumOps.Zero))
                {
                    T entropy = _policy.GetEntropy(state);
                    gradientScale = NumOps.Add(gradientScale, NumOps.Multiply(_entropyCoefficient, entropy));
                }

                // Store the gradient information
                gradients.Add((state, action, gradientScale, logProb));
            }

            // Apply gradients to update the policy
            _policy.UpdateParameters(gradients, _learningRate);
            
            // Set LastLoss to track training progress
            // For REINFORCE, we'll use the average advantage magnitude as a proxy for loss
            T totalAdvantage = NumOps.Zero;
            for (int i = 0; i < advantages.Length; i++)
            {
                totalAdvantage = NumOps.Add(totalAdvantage, NumOps.Abs(advantages[i]));
            }
            LastLoss = NumOps.Divide(totalAdvantage, NumOps.FromDouble(advantages.Length));
        }

        /// <summary>
        /// Trains the agent on a batch of experiences.
        /// </summary>
        /// <param name="states">The states experienced.</param>
        /// <param name="actions">The actions taken.</param>
        /// <param name="rewards">The rewards received.</param>
        /// <param name="nextStates">The next states reached.</param>
        /// <param name="dones">Whether each transition led to a terminal state.</param>
        /// <returns>The loss value from training.</returns>
        public T Train(TState[] states, TAction[] actions, T[] rewards, TState[] nextStates, bool[] dones)
        {
            // REINFORCE typically works with complete episodes, so we'll process these as a batch
            T totalLoss = NumOps.Zero;
            
            // Add each transition to the episode buffer
            for (int i = 0; i < states.Length; i++)
            {
                _episode.Add((states[i], actions[i], rewards[i], dones[i]));
                
                // If this is a terminal state, update the policy
                if (dones[i])
                {
                    ProcessEpisode();
                    totalLoss = NumOps.Add(totalLoss, LastLoss);
                }
            }
            
            // Return average loss
            return NumOps.Divide(totalLoss, NumOps.FromDouble(states.Length));
        }

        /// <summary>
        /// Gets the parameters of the agent.
        /// </summary>
        /// <returns>A vector containing all parameters.</returns>
        public Vector<T> GetParameters()
        {
            var allParameters = new List<T>();
            
            // Get policy parameters
            if (_policy is DiscreteStochasticPolicy<T> discretePolicy)
            {
                var policyParams = discretePolicy.GetParameters();
                foreach (var paramVector in policyParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
            }
            else if (_policy is ContinuousStochasticPolicy<T> continuousPolicy)
            {
                var (commonParams, meanParams, stdDevParams) = continuousPolicy.GetParameters();
                
                // Add common layer parameters
                foreach (var paramVector in commonParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
                
                // Add mean layer parameters
                foreach (var paramVector in meanParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
                
                // Add std dev layer parameters
                foreach (var paramVector in stdDevParams)
                {
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        allParameters.Add(paramVector[i]);
                    }
                }
            }
            
            // Get baseline parameters if using baseline
            if (_useBaseline && _baseline != null)
            {
                var baselineParams = _baseline.GetParameters();
                for (int i = 0; i < baselineParams.Length; i++)
                {
                    allParameters.Add(baselineParams[i]);
                }
            }
            
            return new Vector<T>([.. allParameters]);
        }

        /// <summary>
        /// Sets the parameters of the agent.
        /// </summary>
        /// <param name="parameters">A vector containing all parameters.</param>
        public void SetParameters(Vector<T> parameters)
        {
            int index = 0;
            
            // Set policy parameters
            if (_policy is DiscreteStochasticPolicy<T> discretePolicy)
            {
                var policyParams = discretePolicy.GetParameters();
                var newPolicyParams = new List<Vector<T>>();
                
                foreach (var paramVector in policyParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newPolicyParams.Add(newVector);
                }
                
                discretePolicy.SetParameters(newPolicyParams);
            }
            else if (_policy is ContinuousStochasticPolicy<T> continuousPolicy)
            {
                var (commonParams, meanParams, stdDevParams) = continuousPolicy.GetParameters();
                var newCommonParams = new List<Vector<T>>();
                var newMeanParams = new List<Vector<T>>();
                var newStdDevParams = new List<Vector<T>>();
                
                // Extract common layer parameters
                foreach (var paramVector in commonParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newCommonParams.Add(newVector);
                }
                
                // Extract mean layer parameters
                foreach (var paramVector in meanParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newMeanParams.Add(newVector);
                }
                
                // Extract std dev layer parameters
                foreach (var paramVector in stdDevParams)
                {
                    var newVector = new Vector<T>(paramVector.Length);
                    for (int i = 0; i < paramVector.Length; i++)
                    {
                        newVector[i] = parameters[index++];
                    }
                    newStdDevParams.Add(newVector);
                }
                
                continuousPolicy.SetParameters((newCommonParams, newMeanParams, newStdDevParams));
            }
            
            // Set baseline parameters if using baseline
            if (_useBaseline && _baseline != null)
            {
                var baselineParams = _baseline.GetParameters();
                var newBaselineParams = new Vector<T>(baselineParams.Length);
                for (int i = 0; i < baselineParams.Length; i++)
                {
                    newBaselineParams[i] = parameters[index++];
                }
                _baseline.SetParameters(newBaselineParams);
            }
        }


        /// <summary>
        /// Private helper class to generate random numbers of type T.
        /// </summary>
        /// <typeparam name="TNum">The numeric type.</typeparam>
        private class RandomNumberGenerator<TNum>
        {
            /// <summary>
            /// Gets the numeric operations for type TNum.
            /// </summary>
            protected INumericOperations<TNum> NumOps => MathHelper.GetNumericOperations<TNum>();
            private readonly Random _random = default!;

            public RandomNumberGenerator(int? seed = null)
            {
                _random = seed.HasValue ? new Random(seed.Value) : new Random();
            }

            public TNum NextDouble()
            {
                return NumOps.FromDouble(_random.NextDouble());
            }

            public TNum NextGaussian()
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return NumOps.FromDouble(z);
            }
        }

        /// <summary>
        /// Private implementation of a value function network.
        /// </summary>
        /// <typeparam name="TNum">The numeric type.</typeparam>
        private class ValueNetwork<TNum> : IValueFunction<TState, TNum>
        {
            /// <summary>
            /// Gets the numeric operations for type TNum.
            /// </summary>
            protected INumericOperations<TNum> NumOps => MathHelper.GetNumericOperations<TNum>();
            private readonly List<LayerBase<TNum>> _layers = default!;
            private readonly TNum _learningRate = default!;

            public ValueNetwork(int stateSize, int[] hiddenSizes, ActivationFunction activation, int? seed = null)
            {
                _layers = new List<LayerBase<TNum>>();
                _learningRate = NumOps.FromDouble(0.001);

                // Input layer to first hidden layer
                int inputSize = stateSize;
                for (int i = 0; i < hiddenSizes.Length; i++)
                {
                    _layers.Add(new DenseLayer<TNum>(inputSize, hiddenSizes[i], ActivationFunctionFactory<TNum>.CreateActivationFunction(activation)));
                    inputSize = hiddenSizes[i];
                }

                // Output layer (single value)
                _layers.Add(new DenseLayer<TNum>(inputSize, 1, ActivationFunctionFactory<TNum>.CreateActivationFunction(ActivationFunction.Identity)));
            }

            public TNum PredictValue(TState state)
            {
                Tensor<TNum> output = state as Tensor<TNum> ?? throw new InvalidCastException("State must be a Tensor<TNum>");
                foreach (var layer in _layers)
                {
                    output = layer.Forward(output);
                }
                return output.ToVector()[0];
            }

            public Vector<TNum> PredictValues(TState[] states)
            {
                var values = new Vector<TNum>(states.Length);
                for (int i = 0; i < states.Length; i++)
                {
                    values[i] = PredictValue(states[i]);
                }
                return values;
            }

            public TNum Update(TState[] states, Vector<TNum> targets)
            {
                TNum totalLoss = NumOps.Zero;

                for (int i = 0; i < states.Length; i++)
                {
                    // Forward pass
                    Tensor<TNum> output = states[i] as Tensor<TNum> ?? throw new InvalidCastException("State must be a Tensor<TNum>");
                    foreach (var layer in _layers)
                    {
                        output = layer.Forward(output);
                    }

                    // Compute loss
                    TNum prediction = output.ToVector()[0];
                    TNum error = NumOps.Subtract(prediction, targets[i]);
                    totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(error, error));

                    // Backward pass (simple gradient for MSE loss)
                    Tensor<TNum> gradient = new Tensor<TNum>(new[] { 1 });
                    gradient[0] = NumOps.Multiply(NumOps.FromDouble(2.0), error);

                    foreach (var layer in _layers.AsEnumerable().Reverse())
                    {
                        if (layer is DenseLayer<TNum> denseLayer)
                        {
                            gradient = denseLayer.Backward(gradient);
                            denseLayer.UpdateParameters(_learningRate);
                        }
                    }
                }

                return NumOps.Divide(totalLoss, NumOps.FromDouble(states.Length));
            }

            public Vector<TNum> GetParameters()
            {
                var parameters = new List<TNum>();

                foreach (var layer in _layers)
                {
                    if (layer is DenseLayer<TNum> denseLayer)
                    {
                        var layerParams = denseLayer.GetParameters();
                        for (int i = 0; i < layerParams.Length; i++)
                        {
                            parameters.Add(layerParams[i]);
                        }
                    }
                }

                return new Vector<TNum>([.. parameters]);
            }

            public void SetParameters(Vector<TNum> parameters)
            {
                int index = 0;

                foreach (var layer in _layers)
                {
                    if (layer is DenseLayer<TNum> denseLayer)
                    {
                        var layerParams = denseLayer.GetParameters();
                        var newParams = new Vector<TNum>(layerParams.Length);
                        
                        for (int i = 0; i < layerParams.Length; i++)
                        {
                            newParams[i] = parameters[index++];
                        }
                        
                        denseLayer.SetParameters(newParams);
                    }
                }
            }

            public void CopyParametersFrom(IValueFunction<TState, TNum> source)
            {
                var parameters = source.GetParameters();
                SetParameters(parameters);
            }

            public void SoftUpdate(IValueFunction<TState, TNum> source, TNum tau)
            {
                // Soft update: target = tau * source + (1 - tau) * target
                var sourceParams = source.GetParameters();
                var targetParams = GetParameters();
                
                if (sourceParams.Length != targetParams.Length)
                {
                    throw new InvalidOperationException("Parameter vectors must have the same length for soft update");
                }
                
                // Apply soft update: targetParams = (1 - tau) * targetParams + tau * sourceParams
                for (int i = 0; i < targetParams.Length; i++)
                {
                    targetParams[i] = NumOps.Add(
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), targetParams[i]),
                        NumOps.Multiply(tau, sourceParams[i])
                    );
                }
                
                SetParameters(targetParams);
            }
        }
    }
}