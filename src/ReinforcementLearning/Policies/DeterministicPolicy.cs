using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Deterministic policy for continuous action spaces.
    /// Directly outputs actions without sampling from a distribution.
    /// Commonly used in DDPG, TD3, and other deterministic policy gradient methods.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DeterministicPolicy<T> : PolicyBase<T>
    {
        private readonly NeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;
        private readonly bool _useTanhSquashing;

        /// <summary>
        /// Initializes a new instance of the DeterministicPolicy class.
        /// </summary>
        /// <param name="policyNetwork">The neural network that outputs actions.</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="explorationStrategy">The exploration strategy for training.</param>
        /// <param name="useTanhSquashing">Whether to apply tanh squashing to bound actions to [-1, 1].</param>
        /// <param name="random">Optional random number generator.</param>
        public DeterministicPolicy(
            NeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            bool useTanhSquashing = true,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(policyNetwork);
            _policyNetwork = policyNetwork;
            Guard.NotNull(explorationStrategy);
            _explorationStrategy = explorationStrategy;
            _actionSize = actionSize;
            _useTanhSquashing = useTanhSquashing;
        }

        /// <summary>
        /// Selects a deterministic action from the policy network.
        /// </summary>
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            ValidateState(state, nameof(state));

            // Get deterministic action from network
            var stateTensor = Tensor<T>.FromVector(state);
            var actionTensor = _policyNetwork.Predict(stateTensor);
            var action = actionTensor.ToVector();

            ValidateActionSize(_actionSize, action.Length, nameof(action));

            // Apply tanh squashing if enabled
            if (_useTanhSquashing)
            {
                for (int i = 0; i < action.Length; i++)
                {
                    double actionValue = NumOps.ToDouble(action[i]);
                    action[i] = NumOps.FromDouble(Math.Tanh(actionValue));
                }
            }

            if (training)
            {
                // Apply exploration noise during training
                return _explorationStrategy.GetExplorationAction(state, action, _actionSize, _random);
            }

            return action;
        }

        /// <summary>
        /// Computes log probability for a deterministic policy.
        /// This returns a constant (zero) since deterministic policies have delta distribution.
        /// </summary>
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            // Deterministic policies have infinite log probability at the selected action
            // and negative infinity elsewhere. In practice, we return zero or handle specially.
            // For compatibility with policy gradient methods, return zero.
            return NumOps.Zero;
        }

        /// <summary>
        /// Gets the neural networks used by this policy.
        /// </summary>
        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        /// <summary>
        /// Resets the exploration strategy.
        /// </summary>
        public override void Reset()
        {
            _explorationStrategy.Reset();
        }

        /// <summary>
        /// Disposes of policy resources.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                // Cleanup resources if needed
            }
            base.Dispose(disposing);
        }
    }
}
