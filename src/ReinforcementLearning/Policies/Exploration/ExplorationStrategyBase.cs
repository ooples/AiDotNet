using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Abstract base class for exploration strategy implementations.
    /// Provides common functionality for noise generation and action clamping.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public abstract class ExplorationStrategyBase<T> : IExplorationStrategy<T>
    {
        /// <summary>
        /// Numeric operations helper for type-agnostic calculations.
        /// </summary>
        protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Modifies or replaces the policy's action for exploration.
        /// </summary>
        /// <param name="state">The current state.</param>
        /// <param name="policyAction">The action suggested by the policy.</param>
        /// <param name="actionSpaceSize">The number of possible actions.</param>
        /// <param name="random">Random number generator for stochastic exploration.</param>
        /// <returns>The action to take after applying exploration.</returns>
        public abstract Vector<T> GetExplorationAction(
            Vector<T> state,
            Vector<T> policyAction,
            int actionSpaceSize,
            Random random);

        /// <summary>
        /// Updates internal parameters (e.g., epsilon decay, noise reduction).
        /// Called after each training step.
        /// </summary>
        public abstract void Update();

        /// <summary>
        /// Resets internal state (e.g., for new episodes or training sessions).
        /// </summary>
        public virtual void Reset()
        {
            // Base implementation - derived classes can override
        }

        /// <summary>
        /// Generates a standard normal random sample using the Box-Muller transform.
        /// </summary>
        /// <param name="random">Random number generator.</param>
        /// <returns>A sample from the standard normal distribution N(0, 1).</returns>
        protected T BoxMullerSample(Random random)
        {
            // Box-Muller transform for Gaussian sampling
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            double normalSample = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return NumOps.FromDouble(normalSample);
        }

        /// <summary>
        /// Clamps all elements of an action vector to a specified range.
        /// Compatible with net462 (does not use Math.Clamp).
        /// </summary>
        /// <param name="action">The action vector to clamp.</param>
        /// <param name="min">The minimum value (default: -1.0).</param>
        /// <param name="max">The maximum value (default: 1.0).</param>
        /// <returns>A new vector with clamped values.</returns>
        protected Vector<T> ClampAction(Vector<T> action, double min = -1.0, double max = 1.0)
        {
            var clampedAction = new Vector<T>(action.Length);
            for (int i = 0; i < action.Length; i++)
            {
                double value = NumOps.ToDouble(action[i]);
                // Math.Clamp not available in net462
                double clamped = Math.Max(min, Math.Min(max, value));
                clampedAction[i] = NumOps.FromDouble(clamped);
            }
            return clampedAction;
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
    }
}
