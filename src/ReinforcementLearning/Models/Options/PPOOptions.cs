namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for the Proximal Policy Optimization (PPO) algorithm.
    /// </summary>
    public class PPOOptions<T> : ActorCriticOptions<T>
    {
        /// <summary>
        /// Gets or sets the clip parameter for PPO.
        /// </summary>
        /// <remarks>
        /// Controls how far the new policy can deviate from the old policy.
        /// Typical values are between 0.1 and 0.3.
        /// </remarks>
        public double ClipParameter { get; set; } = 0.2;

        /// <summary>
        /// Gets or sets a value indicating whether to use adaptive KL penalty.
        /// </summary>
        /// <remarks>
        /// If true, an adaptive KL penalty will be used to maintain an appropriate
        /// distance between the old and new policies.
        /// </remarks>
        public bool UseKLPenalty { get; set; } = false;

        /// <summary>
        /// Gets or sets the target KL divergence for adaptive KL penalty.
        /// </summary>
        public double TargetKL { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets the KL penalty coefficient.
        /// </summary>
        public double KLCoefficient { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets the number of epochs to train on each batch of data.
        /// </summary>
        /// <remarks>
        /// PPO reuses each batch of data for multiple training epochs.
        /// </remarks>
        public int EpochsPerBatch { get; set; } = 10;

        /// <summary>
        /// Gets or sets the size of mini-batches for training.
        /// </summary>
        /// <remarks>
        /// Each update epoch will split the collected data into mini-batches of this size.
        /// </remarks>
        public int MinibatchSize { get; set; } = 64;

        /// <summary>
        /// Gets or sets a value indicating whether to use value function clipping.
        /// </summary>
        /// <remarks>
        /// If true, the value function update will be clipped similar to the policy update.
        /// </remarks>
        public bool UseValueClipping { get; set; } = true;

        /// <summary>
        /// Gets or sets the clip parameter for the value function.
        /// </summary>
        public double ValueClipParameter { get; set; } = 0.2;

        /// <summary>
        /// Gets or sets a value indicating whether to use early stopping based on approximate KL.
        /// </summary>
        /// <remarks>
        /// If true, training will stop early if the KL divergence gets too high.
        /// </remarks>
        public bool UseEarlyStoppingKL { get; set; } = true;

        /// <summary>
        /// Gets or sets the threshold for early stopping based on KL divergence.
        /// </summary>
        public double EarlyStoppingKLThreshold { get; set; } = 0.015;

        /// <summary>
        /// Gets or sets the number of steps to collect for each PPO update.
        /// </summary>
        /// <remarks>
        /// This overrides the StepsPerUpdate from ActorCriticOptions.
        /// PPO typically uses larger batch sizes (1000-10000 steps).
        /// </remarks>
        public new int StepsPerUpdate { get; set; } = 2048;

        /// <summary>
        /// Gets or sets a value indicating whether to use decay for the clip parameter.
        /// </summary>
        public bool UseClipDecay { get; set; } = false;

        /// <summary>
        /// Gets or sets the final clip parameter value after decay.
        /// </summary>
        public double FinalClipParameter { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets the number of updates over which to decay the clip parameter.
        /// </summary>
        public int ClipDecayUpdates { get; set; } = 1000;

        /// <summary>
        /// Gets or sets a value indicating whether to use adaptive learning rate.
        /// </summary>
        public bool UseAdaptiveLearningRate { get; set; } = false;

        /// <summary>
        /// Gets or sets the factor by which to decrease the learning rate in adaptive mode.
        /// </summary>
        public double LearningRateDecreaseFactor { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets the threshold for learning rate adaptation based on KL divergence.
        /// </summary>
        public double KLLearningRateThreshold { get; set; } = 0.03;

        /// <summary>
        /// Gets or sets a value indicating whether to normalize observations online.
        /// </summary>
        /// <remarks>
        /// If true, observation normalization statistics will be updated online.
        /// </remarks>
        public bool UpdateNormalizationOnline { get; set; } = true;
    }
}