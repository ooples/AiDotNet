using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for the Rainbow DQN algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type used for computations.</typeparam>
    public class RainbowDQNOptions<T> : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets a value indicating whether to use noisy networks for exploration.
        /// </summary>
        public bool UseNoisyNetworks { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the standard deviation for noisy layer initialization.
        /// </summary>
        public T NoisySigma { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether to use prioritized experience replay.
        /// </summary>
        public new bool UsePrioritizedReplay { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the alpha parameter for prioritized experience replay.
        /// Controls how much prioritization is used (0 = no prioritization, 1 = full prioritization).
        /// </summary>
        public T PrioritizedAlpha { get; set; }
        
        /// <summary>
        /// Gets or sets the initial beta parameter for prioritized experience replay.
        /// Controls how much importance sampling correction is used (0 = no correction, 1 = full correction).
        /// </summary>
        public T PrioritizedBeta { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether to use multi-step learning with n-step returns.
        /// </summary>
        public bool UseMultiStepLearning { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the number of steps to use for n-step returns.
        /// </summary>
        public int NSteps { get; set; } = 3;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use distributional RL (C51 algorithm).
        /// </summary>
        public bool UseDistributionalRL { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the number of atoms in the categorical distribution.
        /// </summary>
        public int AtomCount { get; set; } = 51;
        
        /// <summary>
        /// Gets or sets the minimum value in the support range.
        /// </summary>
        public T ValueRangeMin { get; set; }
        
        /// <summary>
        /// Gets or sets the maximum value in the support range.
        /// </summary>
        public T ValueRangeMax { get; set; }

        /// <summary>
        /// Gets or sets the discount factor for future rewards.
        /// </summary>
        /// <remarks>
        /// Value between 0 and 1 that determines how much future rewards are valued.
        /// </remarks>
        public double DiscountFactor { get; set; } = 0.99;

        /// <summary>
        /// Gets or sets the exploration fraction.
        /// </summary>
        /// <remarks>
        /// Fraction of the total training steps over which the exploration rate is annealed.
        /// </remarks>
        public double ExplorationFraction { get; set; } = 0.1;

        /// <summary>
        /// Initializes a new instance of the <see cref="RainbowDQNOptions{T}"/> class.
        /// </summary>
        public RainbowDQNOptions()
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            NoisySigma = numOps.FromDouble(0.5);
            PrioritizedAlpha = numOps.FromDouble(0.6);
            PrioritizedBeta = numOps.FromDouble(0.4);
            ValueRangeMin = numOps.FromDouble(-10);
            ValueRangeMax = numOps.FromDouble(10);
        }
    }
}