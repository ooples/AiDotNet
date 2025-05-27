using AiDotNet.Enums;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for configuring the Rainbow DQN (Deep Q-Network) reinforcement learning algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Rainbow DQN combines several improvements to the standard DQN algorithm:
    /// - Double DQN: Reduces overestimation by using two networks for action selection and evaluation
    /// - Dueling Networks: Separates state value and action advantage estimation
    /// - Prioritized Experience Replay: Focuses training on important transitions
    /// - Noisy Networks: Adds parametric noise to weights for better exploration
    /// - Multi-step Learning: Uses n-step returns for faster reward propagation
    /// - Distributional RL (C51): Models the full distribution of returns instead of just the mean
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Rainbow DQN is a powerful reinforcement learning algorithm that combines multiple
    /// enhancements to traditional DQN. It's particularly effective for complex environments
    /// where the agent needs to make decisions based on high-dimensional inputs.
    /// Think of it as a "best of all worlds" approach that can learn more efficiently
    /// and achieve better performance than standard DQN.
    /// </para>
    /// </remarks>
    public class RainbowDQNOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets a value indicating whether to use Double DQN.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Double DQN helps reduce overestimation of Q-values by using the online network
        /// to select actions and the target network to evaluate those actions.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This prevents the agent from being too optimistic about future rewards,
        /// which can lead to more stable and reliable learning.
        /// </para>
        /// </remarks>
        public bool UseDoubleDQN { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use Dueling DQN architecture.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Dueling DQN separates the value and advantage streams, which can lead to better
        /// policy evaluation and more efficient learning.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This helps the agent distinguish between the value of being in a state (how good is this situation?)
        /// and the advantage of taking specific actions in that state (which action is best here?).
        /// </para>
        /// </remarks>
        public bool UseDuelingDQN { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use Noisy Networks for exploration.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Noisy networks add parametric noise to the weights, which can help with exploration.
        /// When enabled, epsilon-greedy exploration is typically disabled as exploration is handled by the network itself.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// Instead of randomly selecting actions occasionally (epsilon-greedy), noisy networks
        /// add randomness directly into the decision-making process. This often leads to more
        /// intelligent exploration that adapts over time.
        /// </para>
        /// </remarks>
        public bool UseNoisyNetworks { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the initial sigma value for Noisy Networks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Controls the initial amount of noise in the noisy layers.
        /// Typical values range from 0.1 to 0.5.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// Higher values mean more exploration initially. The noise naturally
        /// decreases as the agent learns what works.
        /// </para>
        /// </remarks>
        public double NoisyNetworksSigma { get; set; } = 0.5;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use factorized Gaussian noise in Noisy Networks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Factorized noise reduces the number of independent noise variables,
        /// making the approach more computationally efficient for large networks.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is a technical optimization that makes noisy networks run faster without
        /// significantly changing their behavior. It's generally recommended to keep this enabled.
        /// </para>
        /// </remarks>
        public bool UseFactorizedNoiseInNoisyNetworks { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use Prioritized Experience Replay.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Prioritized experience replay gives higher priority to experiences from which
        /// the agent can learn more (typically those with large errors).
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This helps the agent learn more efficiently by focusing on the most informative
        /// experiences. For example, it might pay more attention to rare or surprising
        /// situations where its predictions were very wrong.
        /// </para>
        /// </remarks>
        public new bool UsePrioritizedReplay { get; set; } = true;

        /// <summary>
        /// Gets or sets the prioritized replay alpha parameter.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Controls the amount of prioritization in the prioritized replay buffer.
        /// A value of 0 means no prioritization, while a value of 1 means full prioritization.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// Higher values mean the agent focuses more strongly on surprising experiences.
        /// The default of 0.6 provides a good balance for most environments.
        /// </para>
        /// </remarks>
        public double PrioritizedReplayAlpha { get; set; } = 0.6;

        /// <summary>
        /// Gets or sets the prioritized replay beta initial value.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Controls the amount of importance sampling correction. Starts at this value
        /// and anneals to 1.0 over the course of training.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is a technical parameter that helps ensure the agent's learning stays
        /// balanced even when it's focusing more on certain experiences. It should generally
        /// start at a moderate value and increase during training.
        /// </para>
        /// </remarks>
        public double PrioritizedReplayBetaInitial { get; set; } = 0.4;

        /// <summary>
        /// Gets or sets the prioritized replay beta annealing steps.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Number of steps over which to anneal beta from its initial value to 1.0.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This determines how quickly the beta parameter increases to its final value of 1.0.
        /// Longer annealing helps maintain learning stability in complex environments.
        /// </para>
        /// </remarks>
        public int PrioritizedReplayBetaSteps { get; set; } = 100000;

        /// <summary>
        /// Gets or sets a value indicating whether to use Multi-step Returns.
        /// </summary>
        /// <remarks>
        /// <para>
        /// N-step returns look ahead multiple steps when calculating the target value,
        /// which can help propagate rewards faster but may increase variance.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This helps the agent learn more quickly by looking ahead multiple steps
        /// when evaluating its actions. It's particularly helpful in environments where
        /// rewards are delayed or sparse.
        /// </para>
        /// </remarks>
        public bool UseMultiStepReturns { get; set; } = true;

        /// <summary>
        /// Gets or sets the number of steps for N-step returns.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Determines how many steps to look ahead when calculating returns.
        /// Higher values can speed up learning but may introduce more variance.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is how many steps into the future the agent looks when evaluating its actions.
        /// 3-5 steps is often a good balance between speed and stability.
        /// </para>
        /// </remarks>
        public int NSteps { get; set; } = 3;

        /// <summary>
        /// Gets or sets a value indicating whether to use Distributional RL (C51 algorithm).
        /// </summary>
        /// <remarks>
        /// <para>
        /// Distributional RL models the full distribution of returns rather than just the expected value.
        /// The C51 algorithm uses a categorical distribution with a fixed number of atoms.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// Instead of just predicting the average reward, the agent predicts the entire range
        /// of possible rewards and their probabilities. This gives the agent much richer
        /// information about uncertainty and risk, often leading to better decisions.
        /// </para>
        /// </remarks>
        public bool UseDistributionalRL { get; set; } = true;

        /// <summary>
        /// Gets or sets the number of atoms in the categorical distribution for C51.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This parameter determines the granularity of the return distribution.
        /// More atoms provide a finer-grained distribution but require more computational resources.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is like deciding how many "buckets" to split possible returns into.
        /// 51 atoms is the default from the original C51 paper and works well for most environments.
        /// </para>
        /// </remarks>
        public int AtomCount { get; set; } = 51;

        /// <summary>
        /// Gets or sets the minimum value in the support range for distributional RL.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The minimum possible value in the return distribution.
        /// Should be set based on the environment's reward scale.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is the lower bound of possible returns the agent considers.
        /// Should be set to a value that's unlikely to be exceeded in the negative direction.
        /// </para>
        /// </remarks>
        public double ValueRangeMin { get; set; } = -10.0;

        /// <summary>
        /// Gets or sets the maximum value in the support range for distributional RL.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The maximum possible value in the return distribution.
        /// Should be set based on the environment's reward scale.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is the upper bound of possible returns the agent considers.
        /// Should be set to a value that's unlikely to be exceeded in the positive direction.
        /// </para>
        /// </remarks>
        public double ValueRangeMax { get; set; } = 10.0;

        /// <summary>
        /// Gets or sets the discount factor (gamma) for future rewards.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Determines how much the agent values future rewards compared to immediate rewards.
        /// Range: [0, 1]. 0 means only immediate rewards matter, 1 means future rewards are valued equally.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how "forward-thinking" the agent is. Higher values (e.g., 0.99) make
        /// the agent consider long-term consequences, while lower values make it more short-sighted.
        /// </para>
        /// </remarks>
        public double DiscountFactor { get; set; } = 0.99;

        /// <summary>
        /// Gets or sets the network architecture for the Q-network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Specifies the hidden layer sizes for the neural network.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This defines the structure of the neural network's "brain". Larger numbers and more
        /// layers mean a more powerful model that can potentially learn more complex patterns,
        /// but might also be slower to train.
        /// </para>
        /// </remarks>
        public int[] NetworkArchitecture { get; set; } = new int[] { 256, 256 };

        /// <summary>
        /// Gets or sets the activation function for the Q-network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The activation function to use in the hidden layers of the neural network.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is a mathematical function that adds non-linearity to the network.
        /// ReLU is a good default for most reinforcement learning tasks.
        /// </para>
        /// </remarks>
        public ActivationFunction ActivationFunction { get; set; } = ActivationFunction.ReLU;

        /// <summary>
        /// Gets or sets a value indicating whether to clip rewards.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When enabled, rewards are clipped to the range [-1, 1], which can help with
        /// learning stability in environments with very large or varied rewards.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This normalizes rewards to prevent extreme values from destabilizing learning.
        /// Useful in environments where reward magnitude varies greatly.
        /// </para>
        /// </remarks>
        public bool ClipRewards { get; set; } = false;

        /// <summary>
        /// Gets or sets the frequency of updates to the target network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// How many online network updates between each target network update.
        /// Only used when UseSoftTargetUpdate is false.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This controls how often the target network is updated to match the online network.
        /// Less frequent updates provide more stability but slower adaptation.
        /// </para>
        /// </remarks>
        public new int TargetUpdateFrequency { get; set; } = 1000;

        /// <summary>
        /// Gets or sets the exploration fraction.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Fraction of the total training steps over which the exploration rate is annealed.
        /// Only used when UseNoisyNetworks is false.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This determines how quickly the agent transitions from exploration (trying new things)
        /// to exploitation (using what it has learned). Only relevant when not using noisy networks.
        /// </para>
        /// </remarks>
        public double ExplorationFraction { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets a value indicating whether to use soft update for the target network.
        /// </summary>
        /// <remarks>
        /// <para>
        /// If true, the target network is updated using a soft update with the tau parameter.
        /// If false, the target network is updated by copying the online network every TargetNetworkUpdateFrequency steps.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// Soft updates gradually blend the target network toward the online network,
        /// which can provide more stability. Hard updates copy the entire network periodically.
        /// </para>
        /// </remarks>
        public bool UseSoftTargetUpdate { get; set; } = false;

        /// <summary>
        /// Gets or sets the type of optimizer to use for training.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The optimization algorithm used to update the neural network weights.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b>
        /// This is the algorithm that adjusts the network's weights during learning.
        /// Adam is a good default choice that works well across many environments.
        /// </para>
        /// </remarks>
        public OptimizerType OptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets the maximum number of training steps.
        /// </summary>
        /// <remarks>
        /// This is used for annealing parameters like exploration rate over time.
        /// </remarks>
        public int MaxTrainingSteps { get; set; } = 1000000;
    }
}