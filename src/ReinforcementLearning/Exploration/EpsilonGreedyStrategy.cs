using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Exploration
{
    /// <summary>
    /// Implements epsilon-greedy exploration strategy for discrete action spaces.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// The epsilon-greedy strategy selects a random action with probability epsilon,
    /// and the greedy action (the one with highest estimated value) with probability 1-epsilon.
    /// Epsilon typically decays over time to shift from exploration to exploitation.
    /// </para>
    /// </remarks>
    public class EpsilonGreedyStrategy<T> : IExplorationStrategy<int, T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        private readonly T _initialEpsilon = default!;
        private readonly T _finalEpsilon = default!;
        private readonly T _decayRate = default!;
        private readonly int _actionSize;
        private readonly Random _random = default!;
        private readonly T _minStepsForDecay = default!;

        /// <summary>
        /// Gets the current exploration rate (epsilon).
        /// </summary>
        public T ExplorationRate { get; private set; }

        /// <summary>
        /// Gets a value indicating whether the strategy is suitable for continuous action spaces.
        /// </summary>
        public bool IsContinuous => false;

        /// <summary>
        /// Initializes a new instance of the <see cref="EpsilonGreedyStrategy{T}"/> class.
        /// </summary>
        /// <param name="actionSize">The number of possible actions.</param>
        /// <param name="initialEpsilon">The initial exploration rate.</param>
        /// <param name="finalEpsilon">The final exploration rate after decay.</param>
        /// <param name="decayRate">The rate at which epsilon decays.</param>
        /// <param name="minStepsForDecay">The minimum number of steps required for complete decay.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        public EpsilonGreedyStrategy(
            int actionSize,
            double initialEpsilon = 1.0,
            double finalEpsilon = 0.01,
            double decayRate = 0.995,
            double minStepsForDecay = 10000,
            int? seed = null)
        {
            _actionSize = actionSize;
            _initialEpsilon = NumOps.FromDouble(initialEpsilon);
            _finalEpsilon = NumOps.FromDouble(finalEpsilon);
            _decayRate = NumOps.FromDouble(decayRate);
            _minStepsForDecay = NumOps.FromDouble(minStepsForDecay);
            ExplorationRate = _initialEpsilon;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Applies the exploration strategy to potentially modify an action.
        /// </summary>
        /// <param name="action">The original action selected by the policy.</param>
        /// <param name="step">The current training step, used to adjust exploration parameters over time.</param>
        /// <returns>The potentially modified action after applying exploration.</returns>
        public int ApplyExploration(int action, long step)
        {
            // Decay epsilon based on the current step
            Decay(step);

            // With probability epsilon, select a random action
            if (_random.NextDouble() < Convert.ToDouble(ExplorationRate))
            {
                return _random.Next(_actionSize);
            }

            // Otherwise, return the original action
            return action;
        }

        /// <summary>
        /// Decays the exploration rate according to the strategy's schedule.
        /// </summary>
        /// <param name="step">The current training step.</param>
        public void Decay(long step)
        {
            if (step < Convert.ToInt64(_minStepsForDecay))
            {
                // Linear decay
                T stepT = NumOps.FromDouble(step);
                T progress = NumOps.Divide(stepT, _minStepsForDecay);
                ExplorationRate = NumOps.Subtract(_initialEpsilon, 
                                 NumOps.Multiply(progress, 
                                 NumOps.Subtract(_initialEpsilon, _finalEpsilon)));
            }
            else
            {
                // Exponential decay thereafter
                T decaySteps = NumOps.FromDouble(step - Convert.ToInt64(_minStepsForDecay));
                T decayFactor = NumOps.Power(_decayRate, decaySteps);
                ExplorationRate = NumOps.Add(_finalEpsilon, 
                                NumOps.Multiply(NumOps.Subtract(_initialEpsilon, _finalEpsilon), 
                                decayFactor));
            }

            // Ensure epsilon doesn't go below the final value
            ExplorationRate = MathHelper.Max(ExplorationRate, _finalEpsilon);
        }

        /// <summary>
        /// Resets the exploration parameters to their initial values.
        /// </summary>
        public void Reset()
        {
            ExplorationRate = _initialEpsilon;
        }

        /// <summary>
        /// Gets a value indicating whether the exploration is active.
        /// </summary>
        /// <param name="step">The current training step.</param>
        /// <returns>True if exploration is still active at the current step, otherwise false.</returns>
        public bool IsActive(long step)
        {
            Decay(step);
            return NumOps.GreaterThan(ExplorationRate, NumOps.FromDouble(0.0001));
        }
    }
}