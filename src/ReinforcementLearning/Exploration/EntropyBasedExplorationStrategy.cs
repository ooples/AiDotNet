using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Exploration
{
    /// <summary>
    /// Implements entropy-based exploration for both discrete and continuous action spaces.
    /// </summary>
    /// <typeparam name="TAction">The type used to represent actions.</typeparam>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// Entropy-based exploration encourages the agent to maintain policy entropy (randomness)
    /// during training, which helps with exploration. This is commonly used in algorithms like
    /// SAC (Soft Actor-Critic) and PPO with entropy regularization.
    /// </para>
    /// </remarks>
    public class EntropyBasedExplorationStrategy<TAction, T> : IExplorationStrategy<TAction, T>
        where TAction : notnull
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        private readonly bool _isContinuous;
        private readonly T _initialTemperature = default!;
        private readonly T _minTemperature = default!;
        private readonly T _decayRate = default!;
        private readonly long _decaySteps;
        private readonly Random _random = default!;
        private readonly Func<T, TAction, TAction> _applyTemperature = default!;

        /// <summary>
        /// Gets the current exploration rate (temperature parameter).
        /// </summary>
        public T ExplorationRate { get; private set; }

        /// <summary>
        /// Gets a value indicating whether the strategy is suitable for continuous action spaces.
        /// </summary>
        public bool IsContinuous => _isContinuous;

        /// <summary>
        /// Initializes a new instance of the <see cref="EntropyBasedExplorationStrategy{TAction, T}"/> class for discrete actions.
        /// </summary>
        /// <param name="initialTemperature">The initial temperature parameter.</param>
        /// <param name="minTemperature">The minimum temperature after decay.</param>
        /// <param name="decayRate">The rate at which the temperature decays.</param>
        /// <param name="decaySteps">The number of steps over which to decay the temperature.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        /// <returns>A new entropy-based exploration strategy for discrete actions.</returns>
        public static EntropyBasedExplorationStrategy<Vector<T>, T> ForContinuousActions(
            double initialTemperature = 1.0,
            double minTemperature = 0.1, 
            double decayRate = 0.9999,
            long decaySteps = 100000,
            int? seed = null)
        {
            return new EntropyBasedExplorationStrategy<Vector<T>, T>(
                true,
                initialTemperature,
                minTemperature,
                decayRate,
                decaySteps,
                ApplyTemperatureToVector,
                seed);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="EntropyBasedExplorationStrategy{TAction, T}"/> class for discrete actions.
        /// </summary>
        /// <param name="initialTemperature">The initial temperature parameter.</param>
        /// <param name="minTemperature">The minimum temperature after decay.</param>
        /// <param name="decayRate">The rate at which the temperature decays.</param>
        /// <param name="decaySteps">The number of steps over which to decay the temperature.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        /// <returns>A new entropy-based exploration strategy for discrete actions.</returns>
        public static EntropyBasedExplorationStrategy<int, T> ForDiscreteActions(
            double initialTemperature = 1.0,
            double minTemperature = 0.1,
            double decayRate = 0.9999,
            long decaySteps = 100000,
            int? seed = null)
        {
            return new EntropyBasedExplorationStrategy<int, T>(
                false,
                initialTemperature,
                minTemperature,
                decayRate,
                decaySteps,
                (_, action) => action,  // For discrete actions, we don't modify the action directly
                seed);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="EntropyBasedExplorationStrategy{TAction, T}"/> class.
        /// </summary>
        /// <param name="isContinuous">Whether this strategy is for continuous action spaces.</param>
        /// <param name="initialTemperature">The initial temperature parameter.</param>
        /// <param name="minTemperature">The minimum temperature after decay.</param>
        /// <param name="decayRate">The rate at which the temperature decays.</param>
        /// <param name="decaySteps">The number of steps over which to decay the temperature.</param>
        /// <param name="applyTemperature">Function to apply temperature to an action.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        private EntropyBasedExplorationStrategy(
            bool isContinuous,
            double initialTemperature,
            double minTemperature,
            double decayRate,
            long decaySteps,
            Func<T, TAction, TAction> applyTemperature,
            int? seed = null)
        {
            _isContinuous = isContinuous;
            _initialTemperature = NumOps.FromDouble(initialTemperature);
            _minTemperature = NumOps.FromDouble(minTemperature);
            _decayRate = NumOps.FromDouble(decayRate);
            _decaySteps = decaySteps;
            _applyTemperature = applyTemperature;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            ExplorationRate = _initialTemperature;
        }

        /// <summary>
        /// Applies the exploration strategy to potentially modify an action.
        /// </summary>
        /// <param name="action">The original action selected by the policy.</param>
        /// <param name="step">The current training step, used to adjust exploration parameters over time.</param>
        /// <returns>The potentially modified action after applying exploration.</returns>
        public TAction ApplyExploration(TAction action, long step)
        {
            // Decay temperature based on the current step
            Decay(step);

            // Apply temperature to the action
            return _applyTemperature(ExplorationRate, action);
        }

        /// <summary>
        /// Applies temperature to a vector action by adding scaled Gaussian noise.
        /// </summary>
        /// <param name="temperature">The temperature parameter.</param>
        /// <param name="action">The original action vector.</param>
        /// <returns>The action with added noise proportional to the temperature.</returns>
        private static Vector<T> ApplyTemperatureToVector(T temperature, Vector<T> action)
        {
            // Create a new vector for the noisy action
            var noisyAction = new Vector<T>(action.Length);
            
            // Create random generator
            Random random = new Random();
            
            // Add noise to each dimension proportional to temperature
            for (int i = 0; i < action.Length; i++)
            {
                // Generate Gaussian noise
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                
                // Scale noise by temperature
                T scaledNoise = NumOps.Multiply(NumOps.FromDouble(noise), temperature);
                
                // Add noise to action
                noisyAction[i] = NumOps.Add(action[i], scaledNoise);
            }
            
            return noisyAction;
        }

        /// <summary>
        /// Decays the exploration rate according to the strategy's schedule.
        /// </summary>
        /// <param name="step">The current training step.</param>
        public void Decay(long step)
        {
            if (step < _decaySteps)
            {
                // Exponential decay
                ExplorationRate = NumOps.Multiply(ExplorationRate, _decayRate);
                
                // Ensure temperature doesn't go below minimum
                ExplorationRate = MathHelper.Max(ExplorationRate, _minTemperature);
            }
        }

        /// <summary>
        /// Resets the exploration parameters to their initial values.
        /// </summary>
        public void Reset()
        {
            ExplorationRate = _initialTemperature;
        }

        /// <summary>
        /// Gets a value indicating whether the exploration is active.
        /// </summary>
        /// <param name="step">The current training step.</param>
        /// <returns>True if exploration is still active at the current step, otherwise false.</returns>
        public bool IsActive(long step)
        {
            Decay(step);
            return NumOps.GreaterThan(ExplorationRate, _minTemperature);
        }
    }
}