namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.
    /// </summary>
    public class TD3Options : DDPGOptions
    {
        /// <summary>
        /// Gets or sets the policy update frequency.
        /// </summary>
        /// <remarks>
        /// The actor will be updated once for every this many critic updates.
        /// This delay helps to reduce variance and stabilize training.
        /// </remarks>
        public int PolicyUpdateFrequency { get; set; } = 2;

        /// <summary>
        /// Gets or sets the target policy noise scale.
        /// </summary>
        /// <remarks>
        /// The scale of noise added to target actions for smoothing.
        /// </remarks>
        public double TargetPolicyNoiseScale { get; set; } = 0.2;

        /// <summary>
        /// Gets or sets the target policy noise clip value.
        /// </summary>
        /// <remarks>
        /// The maximum amount by which the target policy noise is clipped.
        /// </remarks>
        public double TargetPolicyNoiseClip { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets a value indicating whether to use the minimum of the two critic values.
        /// </summary>
        /// <remarks>
        /// If true, the minimum of the two critic values will be used for the target,
        /// which can help prevent overestimation of Q-values.
        /// </remarks>
        public bool UseMinimumQValue { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use the same network structure for both critics.
        /// </summary>
        public bool UseSharedCriticStructure { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use a separate optimizer for each critic.
        /// </summary>
        public bool UseSeparateCriticOptimizers { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use different initializations for the two critics.
        /// </summary>
        public bool UseDifferentCriticInitializations { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to normalize critic gradients separately.
        /// </summary>
        public bool NormalizeCriticGradientsSeparately { get; set; } = false;

        /// <summary>
        /// Gets or sets a value indicating whether to use the average of the two critics for actor updates.
        /// </summary>
        /// <remarks>
        /// If false, only the first critic will be used for actor updates.
        /// </remarks>
        public bool UseAverageCriticForActorUpdate { get; set; } = false;

        /// <summary>
        /// Gets or sets a value indicating whether to use additional exploration noise during warm-up.
        /// </summary>
        public bool UseExtraWarmupNoise { get; set; } = true;

        /// <summary>
        /// Gets or sets the scale of extra warm-up noise.
        /// </summary>
        public double ExtraWarmupNoiseScale { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets a value indicating whether to use a delayed target network update.
        /// </summary>
        /// <remarks>
        /// If true, the target networks will be updated only once every TargetUpdateFrequency steps.
        /// If false, soft updates will be performed at every step.
        /// </remarks>
        public bool UseDelayedTargetUpdate { get; set; } = false;

        /// <summary>
        /// Gets or sets the target network update frequency if using delayed updates.
        /// </summary>
        public new int TargetUpdateFrequency { get; set; } = 10;

        /// <summary>
        /// Gets or sets a value indicating whether to clip the critic values during the actor update.
        /// </summary>
        public bool ClipCriticValueDuringActorUpdate { get; set; } = false;

        /// <summary>
        /// Gets or sets the maximum critic value during actor update.
        /// </summary>
        public double MaxCriticValueDuringActorUpdate { get; set; } = 100.0;

        /// <summary>
        /// Initializes a new instance of the <see cref="TD3Options"/> class.
        /// </summary>
        public TD3Options()
        {
            // TD3 requires continuous action spaces
            IsContinuous = true;
            
            // TD3 typically uses a smaller tau than DDPG
            Tau = 0.005;
            
            // TD3 typically uses a larger batch size
            BatchSize = 256;
            
            // TD3 uses a slower actor learning rate
            ActorLearningRate = 0.0001;
            
            // TD3 often uses Gaussian noise instead of OU noise
            UseGaussianNoise = true;
            OUNoiseTheta = 0.15;
            OUNoiseSigma = 0.1;
            
            // TD3 typically does not use prioritized replay by default
            UsePrioritizedReplay = false;
        }
    }
}