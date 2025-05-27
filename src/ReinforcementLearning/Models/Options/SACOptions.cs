using AiDotNet.Enums;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for the Soft Actor-Critic (SAC) algorithm.
    /// </summary>
    public class SACOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the network architecture for the actor network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] ActorNetworkArchitecture { get; set; } = new int[] { 256, 256 };

        /// <summary>
        /// Gets or sets the network architecture for the critic network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] CriticNetworkArchitecture { get; set; } = new int[] { 256, 256 };

        /// <summary>
        /// Gets or sets the activation function for the actor network.
        /// </summary>
        public ActivationFunction ActorActivationFunction { get; set; } = ActivationFunction.ReLU;

        /// <summary>
        /// Gets or sets the activation function for the critic network.
        /// </summary>
        public ActivationFunction CriticActivationFunction { get; set; } = ActivationFunction.ReLU;

        /// <summary>
        /// Gets or sets the learning rate for the actor network.
        /// </summary>
        public double ActorLearningRate { get; set; } = 0.0003;

        /// <summary>
        /// Gets or sets the learning rate for the critic network.
        /// </summary>
        public double CriticLearningRate { get; set; } = 0.0003;

        /// <summary>
        /// Gets or sets the learning rate for the entropy coefficient (alpha) if automatic tuning is enabled.
        /// </summary>
        public double EntropyLearningRate { get; set; } = 0.0003;

        /// <summary>
        /// Gets or sets the batch size for training.
        /// </summary>
        public new int BatchSize { get; set; } = 256;

        /// <summary>
        /// Gets or sets the soft update factor (tau) for target networks.
        /// </summary>
        /// <remarks>
        /// Controls how quickly the target networks are updated to match the main networks.
        /// </remarks>
        public new double Tau { get; set; } = 0.005;

        /// <summary>
        /// Gets or sets the initial entropy coefficient (alpha).
        /// </summary>
        /// <remarks>
        /// Controls the importance of entropy in the policy. Higher values result in more exploration.
        /// </remarks>
        public double InitialEntropyCoefficient { get; set; } = 0.2;

        /// <summary>
        /// Gets or sets a value indicating whether to automatically tune the entropy coefficient.
        /// </summary>
        /// <remarks>
        /// If true, the entropy coefficient will be adjusted to achieve a target entropy.
        /// </remarks>
        public bool AutoTuneEntropyCoefficient { get; set; } = true;

        /// <summary>
        /// Gets or sets the target entropy for automatic entropy tuning.
        /// </summary>
        /// <remarks>
        /// If null, the target entropy will be set to -ActionSize (for continuous actions).
        /// This is a good default for many continuous control problems.
        /// </remarks>
        public double? TargetEntropy { get; set; } = null;

        /// <summary>
        /// Gets or sets a value indicating whether to use separate networks for Q1 and Q2.
        /// </summary>
        /// <remarks>
        /// If true, two completely separate critic networks will be used.
        /// If false, they will share some layers.
        /// </remarks>
        public bool UseSeparateQNetworks { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use state-dependent exploration.
        /// </summary>
        /// <remarks>
        /// If true, the policy will predict both mean and standard deviation for each action.
        /// If false, a fixed standard deviation will be used.
        /// </remarks>
        public bool UseStateDependentExploration { get; set; } = true;

        /// <summary>
        /// Gets or sets the optimizer type for the actor network.
        /// </summary>
        public OptimizerType ActorOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets the optimizer type for the critic network.
        /// </summary>
        public OptimizerType CriticOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets the optimizer type for the entropy parameter.
        /// </summary>
        public OptimizerType EntropyOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets a value indicating whether to use prioritized experience replay.
        /// </summary>
        public new bool UsePrioritizedReplay { get; set; } = false;

        /// <summary>
        /// Gets or sets the prioritized replay alpha parameter.
        /// </summary>
        public double PrioritizedReplayAlpha { get; set; } = 0.6;

        /// <summary>
        /// Gets or sets the prioritized replay beta initial value.
        /// </summary>
        public double PrioritizedReplayBetaInitial { get; set; } = 0.4;

        /// <summary>
        /// Gets or sets the prioritized replay beta annealing steps.
        /// </summary>
        public int PrioritizedReplayBetaSteps { get; set; } = 100000;

        /// <summary>
        /// Gets or sets a value indicating whether to use squashed Gaussian policy.
        /// </summary>
        /// <remarks>
        /// A squashed Gaussian policy applies a tanh transform to the Gaussian samples,
        /// which bounds the actions to [-1, 1] while maintaining differentiability.
        /// </remarks>
        public bool UseSquashedGaussianPolicy { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use gradient clipping.
        /// </summary>
        public bool UseGradientClipping { get; set; } = true;

        /// <summary>
        /// Gets or sets the maximum gradient norm for gradient clipping.
        /// </summary>
        public new double MaxGradientNorm { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets the warm-up steps before starting training.
        /// </summary>
        /// <remarks>
        /// The agent will collect this many steps of experience with random actions
        /// before starting to train.
        /// </remarks>
        public int WarmUpSteps { get; set; } = 10000;

        /// <summary>
        /// Gets or sets a value indicating whether to use a fixed standard deviation for exploration during warm-up.
        /// </summary>
        public bool UseFixedWarmUpExploration { get; set; } = true;

        /// <summary>
        /// Gets or sets the standard deviation for fixed warm-up exploration.
        /// </summary>
        public double WarmUpExplorationStdDev { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets a value indicating whether to update the networks after each environment step.
        /// </summary>
        /// <remarks>
        /// If false, the networks will be updated every `TrainingFrequency` steps.
        /// </remarks>
        public bool UpdateAfterEachStep { get; set; } = true;

        /// <summary>
        /// Gets or sets the frequency of training updates if not updating after each step.
        /// </summary>
        public int TrainingFrequency { get; set; } = 1;

        /// <summary>
        /// Gets or sets the number of training batches to process for each update.
        /// </summary>
        public int GradientsStepsPerUpdate { get; set; } = 1;

        /// <summary>
        /// Gets or sets a value indicating whether to clip log probabilities.
        /// </summary>
        public bool ClipLogProbs { get; set; } = true;

        /// <summary>
        /// Gets or sets the minimum log probability for clipping.
        /// </summary>
        public double MinLogProb { get; set; } = -20.0;

        /// <summary>
        /// Gets or sets the maximum log probability for clipping.
        /// </summary>
        public double MaxLogProb { get; set; } = 2.0;

        /// <summary>
        /// Initializes a new instance of the <see cref="SACOptions"/> class.
        /// </summary>
        public SACOptions()
        {
            // SAC is designed for continuous action spaces
            IsContinuous = true;
            
            // SAC uses soft target updates
            Tau = 0.005;
            
            // SAC typically uses a discount factor slightly less than 1
            Gamma = 0.99;
        }
    }
}