using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Implements a stochastic policy for discrete action spaces using a neural network.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// This policy outputs a probability distribution over discrete actions and
    /// samples from this distribution to select actions. It's commonly used in
    /// Policy Gradient methods like REINFORCE and A2C.
    /// </para>
    /// </remarks>
    public class DiscreteStochasticPolicy<T> : IPolicy<Tensor<T>, int, T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        private readonly int _stateSize;
        private readonly int _actionSize;
        private readonly List<LayerBase<T>> _layers = default!;
        private readonly SoftmaxActivation<T> _softmax = default!;
        private readonly Random _random = default!;

        /// <summary>
        /// Gets a value indicating whether the policy is stochastic.
        /// </summary>
        public bool IsStochastic => true;

        /// <summary>
        /// Gets a value indicating whether the policy is for continuous action spaces.
        /// </summary>
        public bool IsContinuous => false;

        /// <summary>
        /// Initializes a new instance of the <see cref="DiscreteStochasticPolicy{T}"/> class.
        /// </summary>
        /// <param name="stateSize">The size of the state space.</param>
        /// <param name="actionSize">The size of the action space (number of possible actions).</param>
        /// <param name="hiddenSizes">The sizes of the hidden layers in the policy network.</param>
        /// <param name="activationFunction">The activation function to use for hidden layers.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        public DiscreteStochasticPolicy(
            int stateSize,
            int actionSize,
            int[] hiddenSizes,
            IActivationFunction<T>? activationFunction = null,
            int? seed = null)
        {
            _stateSize = stateSize;
            _actionSize = actionSize;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _softmax = new SoftmaxActivation<T>();

            // Create the network layers
            _layers = [];

            // Input layer
            int inputSize = stateSize;

            // Hidden layers
            foreach (int hiddenSize in hiddenSizes)
            {
                _layers.Add(new DenseLayer<T>(inputSize, hiddenSize, activationFunction));
                inputSize = hiddenSize;
            }

            // Output layer (no activation, as softmax will be applied separately)
            _layers.Add(new DenseLayer<T>(inputSize, actionSize, new IdentityActivation<T>() as IActivationFunction<T>));
        }

        /// <summary>
        /// Selects an action based on the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <returns>The selected action.</returns>
        public int SelectAction(Tensor<T> state)
        {
            // Get action probabilities
            var actionProbabilities = GetActionProbabilities(state);

            // Sample action based on probabilities
            return SampleFromDistribution(actionProbabilities);
        }

        /// <summary>
        /// Evaluates the policy for a given state and returns action probabilities.
        /// </summary>
        /// <param name="state">The state to evaluate.</param>
        /// <returns>A vector of probabilities for each action.</returns>
        public object EvaluatePolicy(Tensor<T> state)
        {
            return GetActionProbabilities(state);
        }

        /// <summary>
        /// Calculates the log probability of taking a specific action in a given state.
        /// </summary>
        /// <param name="state">The state in which the action was taken.</param>
        /// <param name="action">The action that was taken.</param>
        /// <returns>The log probability of the action.</returns>
        public T LogProbability(Tensor<T> state, int action)
        {
            var probabilities = GetActionProbabilities(state);
            T probability = probabilities[action];
            
            // Clamp probability to avoid log(0)
            T minProb = NumOps.FromDouble(1e-10);
            probability = MathHelper.Max(probability, minProb);
            
            return NumOps.Log(probability);
        }

        /// <summary>
        /// Updates the policy parameters using the provided gradients.
        /// </summary>
        /// <param name="gradients">The gradients for the policy parameters.</param>
        /// <param name="learningRate">The learning rate for the update.</param>
        public void UpdateParameters(object gradients, T learningRate)
        {
            if (gradients is not List<Tensor<T>> grads)
            {
                throw new ArgumentException("Gradients must be a List<Tensor<T>>");
            }

            if (grads.Count != _layers.Count)
            {
                throw new ArgumentException($"Expected {_layers.Count} gradients, got {grads.Count}");
            }

            for (int i = 0; i < _layers.Count; i++)
            {
                if (_layers[i] is DenseLayer<T> denseLayer)
                {
                    denseLayer.UpdateParameters(learningRate);
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
            var probabilities = GetActionProbabilities(state);
            T entropy = NumOps.Zero;
            T minProb = NumOps.FromDouble(1e-10);

            for (int i = 0; i < probabilities.Length; i++)
            {
                T prob = MathHelper.Max(probabilities[i], minProb);
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(prob, NumOps.Log(prob)));
            }

            return entropy;
        }

        /// <summary>
        /// Performs a forward pass through the policy network to get action probabilities.
        /// </summary>
        /// <param name="state">The state to evaluate.</param>
        /// <returns>A vector of probabilities for each action.</returns>
        private Vector<T> GetActionProbabilities(Tensor<T> state)
        {
            // Forward pass through the neural network
            var current = state;
            foreach (var layer in _layers)
            {
                current = layer.Forward(current);
            }

            // Ensure the output is a vector
            if (current.Rank != 1)
            {
                throw new InvalidOperationException("Expected network output to be a vector");
            }

            // Apply softmax to get probabilities
            return _softmax.Activate(current.ToVector());
        }

        /// <summary>
        /// Samples an action from a probability distribution.
        /// </summary>
        /// <param name="probabilities">The probability distribution over actions.</param>
        /// <returns>The sampled action.</returns>
        private int SampleFromDistribution(Vector<T> probabilities)
        {
            double randomValue = _random.NextDouble();
            double cumulativeProbability = 0.0;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulativeProbability += Convert.ToDouble(probabilities[i]);
                if (randomValue < cumulativeProbability)
                {
                    return i;
                }
            }

            // Fallback to the last action (should rarely happen due to rounding errors)
            return probabilities.Length - 1;
        }

        /// <summary>
        /// Gets the parameters of the policy network.
        /// </summary>
        /// <returns>A list of parameter tensors for each layer.</returns>
        public List<Vector<T>> GetParameters()
        {
            var parameters = new List<Vector<T>>();
            foreach (var layer in _layers)
            {
                if (layer is DenseLayer<T> denseLayer)
                {
                    parameters.Add(denseLayer.GetParameters());
                }
            }

            return parameters;
        }

        /// <summary>
        /// Sets the parameters of the policy network.
        /// </summary>
        /// <param name="parameters">A list of parameter tensors for each layer.</param>
        public void SetParameters(List<Vector<T>> parameters)
        {
            if (parameters.Count != _layers.Count)
            {
                throw new ArgumentException($"Expected {_layers.Count} parameter tensors, got {parameters.Count}");
            }

            for (int i = 0; i < _layers.Count; i++)
            {
                if (_layers[i] is DenseLayer<T> denseLayer)
                {
                    denseLayer.SetParameters(parameters[i]);
                }
            }
        }
    }
}