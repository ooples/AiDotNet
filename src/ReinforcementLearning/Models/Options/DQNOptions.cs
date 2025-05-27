using AiDotNet.Enums;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for the Deep Q-Network (DQN) algorithm.
    /// </summary>
    public class DQNOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets a value indicating whether to use Double DQN.
        /// </summary>
        /// <remarks>
        /// Double DQN helps reduce overestimation of Q-values by using the online network
        /// to select actions and the target network to evaluate those actions.
        /// </remarks>
        public bool UseDoubleDQN { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use Dueling DQN.
        /// </summary>
        /// <remarks>
        /// Dueling DQN separates the value and advantage streams, which can lead to better
        /// policy evaluation.
        /// </remarks>
        public bool UseDuelingDQN { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use Noisy Networks.
        /// </summary>
        /// <remarks>
        /// Noisy networks add parametric noise to the weights, which can help with exploration.
        /// When enabled, epsilon-greedy exploration is disabled as exploration is handled by the network itself.
        /// </remarks>
        public bool UseNoisyNetworks { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the initial sigma value for Noisy Networks.
        /// </summary>
        /// <remarks>
        /// Controls the initial amount of noise in the noisy layers.
        /// Typical values range from 0.1 to 0.5.
        /// </remarks>
        public double NoisyNetworksSigma { get; set; } = 0.5;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use factorized Gaussian noise in Noisy Networks.
        /// </summary>
        /// <remarks>
        /// Factorized noise reduces the number of independent noise variables,
        /// making the approach more computationally efficient for large networks.
        /// </remarks>
        public bool UseFactorizedNoiseInNoisyNetworks { get; set; } = true;

        /// <summary>
        /// Gets or sets the prioritized replay alpha parameter.
        /// </summary>
        /// <remarks>
        /// Controls the amount of prioritization in the prioritized replay buffer.
        /// A value of 0 means no prioritization, while a value of 1 means full prioritization.
        /// </remarks>
        public double PrioritizedReplayAlpha { get; set; } = 0.6;

        /// <summary>
        /// Gets or sets the prioritized replay beta initial value.
        /// </summary>
        /// <remarks>
        /// Controls the amount of importance sampling correction. Starts at this value
        /// and anneals to 1.0 over the course of training.
        /// </remarks>
        public double PrioritizedReplayBetaInitial { get; set; } = 0.4;

        /// <summary>
        /// Gets or sets the prioritized replay beta annealing steps.
        /// </summary>
        /// <remarks>
        /// Number of steps over which to anneal beta from its initial value to 1.0.
        /// </remarks>
        public int PrioritizedReplayBetaSteps { get; set; } = 100000;

        /// <summary>
        /// Gets or sets the network architecture for the Q-network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] NetworkArchitecture { get; set; } = new int[] { 256, 256 };

        /// <summary>
        /// Gets or sets the activation function for the Q-network.
        /// </summary>
        public ActivationFunction ActivationFunction { get; set; } = ActivationFunction.ReLU;

        /// <summary>
        /// Gets or sets a value indicating whether to use N-step returns.
        /// </summary>
        /// <remarks>
        /// N-step returns can help propagate rewards faster but may increase variance.
        /// </remarks>
        public bool UseNStepReturns { get; set; } = false;

        /// <summary>
        /// Gets or sets the number of steps for N-step returns.
        /// </summary>
        public int NSteps { get; set; } = 3;

        /// <summary>
        /// Gets or sets a value indicating whether to clip rewards.
        /// </summary>
        public bool ClipRewards { get; set; } = false;

        /// <summary>
        /// Gets or sets the frequency of updates to the target network.
        /// </summary>
        /// <remarks>
        /// How many online network updates between each target network update.
        /// </remarks>
        public int TargetNetworkUpdateFrequency { get; set; } = 1000;

        /// <summary>
        /// Gets or sets the exploration fraction.
        /// </summary>
        /// <remarks>
        /// Fraction of the total training steps over which the exploration rate is annealed.
        /// </remarks>
        public double ExplorationFraction { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets the type of optimizer to use for training.
        /// </summary>
        public OptimizerType OptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets a value indicating whether to use soft update for the target network.
        /// </summary>
        /// <remarks>
        /// If true, the target network is updated using a soft update with the tau parameter.
        /// If false, the target network is updated by copying the online network every TargetNetworkUpdateFrequency steps.
        /// </remarks>
        public bool UseSoftTargetUpdate { get; set; } = false;

        /// <summary>
        /// Gets or sets the maximum number of training steps.
        /// </summary>
        /// <remarks>
        /// This is used for annealing the exploration rate over time.
        /// </remarks>
        public int MaxSteps { get; set; } = 1000000;

        /// <summary>
        /// Gets or sets the discount factor for future rewards.
        /// </summary>
        /// <remarks>
        /// Value between 0 and 1 that determines how much future rewards are valued.
        /// </remarks>
        public double DiscountFactor { get; set; } = 0.99;

        /// <summary>
        /// Gets or sets the starting value for epsilon in epsilon-greedy exploration.
        /// </summary>
        public double EpsilonStart { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets the ending value for epsilon in epsilon-greedy exploration.
        /// </summary>
        public double EpsilonEnd { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets the decay rate for epsilon.
        /// </summary>
        public double EpsilonDecay { get; set; } = 0.995;

        /// <summary>
        /// Gets or sets the hidden layer sizes for the Q-network.
        /// </summary>
        public int[] HiddenLayerSizes { get; set; } = new int[] { 256, 256 };

        /// <summary>
        /// Gets or sets the delta value for Huber loss.
        /// </summary>
        /// <remarks>
        /// Used to clip the TD error for stability.
        /// </remarks>
        public double HuberDelta { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets the maximum number of training steps (alias for MaxSteps).
        /// </summary>
        public int MaxTrainingSteps => MaxSteps;
    }
}