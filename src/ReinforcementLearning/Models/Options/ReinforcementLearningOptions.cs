using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Base options class for reinforcement learning algorithms.
    /// </summary>
    public abstract class ReinforcementLearningOptions : ModelOptions
    {
        /// <summary>
        /// Gets or sets the size of the state space (number of dimensions in the state).
        /// </summary>
        public int StateSize { get; set; }

        /// <summary>
        /// Gets or sets the size of the action space (number of possible actions for discrete spaces,
        /// or number of dimensions for continuous action spaces).
        /// </summary>
        public int ActionSize { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the action space is continuous.
        /// </summary>
        public bool IsContinuous { get; set; }

        /// <summary>
        /// Gets or sets the discount factor (gamma) for future rewards.
        /// </summary>
        public double Gamma { get; set; } = 0.99;

        /// <summary>
        /// Gets or sets the learning rate for the algorithm.
        /// </summary>
        public double LearningRate { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the capacity of the replay buffer.
        /// </summary>
        public int ReplayBufferCapacity { get; set; } = 100000;

        /// <summary>
        /// Gets or sets the batch size for training.
        /// </summary>
        public int BatchSize { get; set; } = 64;

        /// <summary>
        /// Gets or sets the number of training epochs per update.
        /// </summary>
        public int EpochsPerUpdate { get; set; } = 1;

        /// <summary>
        /// Gets or sets a value indicating whether to use prioritized experience replay.
        /// </summary>
        public bool UsePrioritizedReplay { get; set; } = false;

        /// <summary>
        /// Gets or sets the initial exploration rate.
        /// </summary>
        public double InitialExplorationRate { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets the final exploration rate after decay.
        /// </summary>
        public double FinalExplorationRate { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets the number of steps to update the target network.
        /// </summary>
        public int TargetUpdateFrequency { get; set; } = 1000;

        /// <summary>
        /// Gets or sets the number of environment steps to take before starting training.
        /// </summary>
        public int WarmupSteps { get; set; } = 1000;

        /// <summary>
        /// Gets or sets the soft update factor (tau) for target networks.
        /// </summary>
        public double Tau { get; set; } = 0.005;

        /// <summary>
        /// Gets or sets a value indicating whether to clip gradients during training.
        /// </summary>
        public bool ClipGradients { get; set; } = true;

        /// <summary>
        /// Gets or sets the maximum gradient norm for gradient clipping.
        /// </summary>
        public double MaxGradientNorm { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets a value indicating whether to normalize observations.
        /// </summary>
        public bool NormalizeObservations { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to normalize rewards.
        /// </summary>
        public bool NormalizeRewards { get; set; } = false;
    }
}