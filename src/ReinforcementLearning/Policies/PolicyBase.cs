using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Abstract base class for policy implementations.
    /// Provides common functionality for numeric operations, random number generation, and resource management.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public abstract class PolicyBase<T> : ModelBase<T, Vector<T>, Vector<T>>, IPolicy<T>
    {
        // NumOps inherited from ModelBase

        /// <summary>
        /// Random number generator for stochastic policies.
        /// </summary>
        protected readonly Random _random;

        /// <summary>
        /// Tracks whether the object has been disposed.
        /// </summary>
        protected bool _disposed;

        /// <summary>
        /// Initializes a new instance of the PolicyBase class.
        /// </summary>
        /// <param name="random">Optional random number generator. If null, a new instance will be created.</param>
        protected PolicyBase(Random? random = null)
        {
            _random = random ?? RandomHelper.CreateSecureRandom();
            _disposed = false;
        }

        /// <summary>
        /// Selects an action given the current state.
        /// </summary>
        /// <param name="state">The current state observation.</param>
        /// <param name="training">Whether the agent is training (enables exploration).</param>
        /// <returns>The selected action vector.</returns>
        public abstract Vector<T> SelectAction(Vector<T> state, bool training = true);

        /// <summary>
        /// Computes the log probability of a given action in a given state.
        /// Used by policy gradient methods (PPO, A2C, etc.).
        /// </summary>
        /// <param name="state">The state observation.</param>
        /// <param name="action">The action taken.</param>
        /// <returns>The log probability of the action.</returns>
        public abstract T ComputeLogProb(Vector<T> state, Vector<T> action);

        /// <summary>
        /// Gets the neural networks used by this policy.
        /// </summary>
        /// <returns>A read-only list of neural networks.</returns>
        public abstract IReadOnlyList<INeuralNetwork<T>> GetNetworks();

        /// <summary>
        /// Resets any internal state (e.g., for recurrent policies, exploration noise).
        /// </summary>
        public virtual void Reset()
        {
            // Base implementation - derived classes can override
        }

        /// <summary>
        /// Validates that an action vector has the expected size.
        /// </summary>
        /// <param name="expected">The expected action size.</param>
        /// <param name="actual">The actual action size.</param>
        /// <param name="paramName">The parameter name for error reporting.</param>
        /// <exception cref="ArgumentException">Thrown when action size doesn't match expected size.</exception>
        protected void ValidateActionSize(int expected, int actual, string paramName)
        {
            if (actual != expected)
            {
                throw new ArgumentException(
                    string.Format("Action size mismatch. Expected {0}, got {1}.", expected, actual),
                    paramName);
            }
        }

        /// <summary>
        /// Validates that a state vector is not null and has positive size.
        /// </summary>
        /// <param name="state">The state vector to validate.</param>
        /// <param name="paramName">The parameter name for error reporting.</param>
        /// <exception cref="ArgumentNullException">Thrown when state is null.</exception>
        /// <exception cref="ArgumentException">Thrown when state has invalid size.</exception>
        protected void ValidateState(Vector<T> state, string paramName)
        {
            if (state is null)
            {
                throw new ArgumentNullException(paramName);
            }
            if (state.Length <= 0)
            {
                throw new ArgumentException("State must have positive size.", paramName);
            }
        }

        #region ModelBase Overrides

        /// <summary>
        /// Predicts an action for the given state (inference mode).
        /// </summary>
        public override Vector<T> Predict(Vector<T> input) => SelectAction(input, training: false);

        /// <summary>
        /// Training is handled by RL algorithms, not directly on the policy.
        /// </summary>
        public override void Train(Vector<T> input, Vector<T> expectedOutput) { }

        /// <inheritdoc />
        public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

        /// <inheritdoc />
        public override Vector<T> GetParameters()
        {
            var networks = GetNetworks();
            if (networks.Count == 0)
                return new Vector<T>(0);

            // Collect parameters from all networks
            var allParams = new List<T>();
            foreach (var network in networks)
            {
                var p = network.GetParameters();
                for (int i = 0; i < p.Length; i++)
                    allParams.Add(p[i]);
            }

            return new Vector<T>(allParams.ToArray());
        }

        /// <inheritdoc />
        public override void SetParameters(Vector<T> parameters) { }

        /// <inheritdoc />
        public override IFullModel<T, Vector<T>, Vector<T>> WithParameters(Vector<T> parameters)
        {
            var copy = DeepCopy();
            copy.SetParameters(parameters);
            return copy;
        }

        /// <inheritdoc />
        public override IFullModel<T, Vector<T>, Vector<T>> DeepCopy()
            => (PolicyBase<T>)MemberwiseClone();

        #endregion

        /// <summary>
        /// Releases the unmanaged resources used by the policy and optionally releases the managed resources.
        /// </summary>
        /// <param name="disposing">True to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    // Derived classes can override to dispose their own resources
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Releases all resources used by the policy.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
