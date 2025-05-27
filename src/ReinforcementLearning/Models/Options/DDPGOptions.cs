using AiDotNet.Enums;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for the Deep Deterministic Policy Gradient (DDPG) algorithm.
    /// </summary>
    public class DDPGOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the network architecture for the actor network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] ActorNetworkArchitecture { get; set; } = new int[] { 400, 300 };

        /// <summary>
        /// Gets or sets the network architecture for the critic network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] CriticNetworkArchitecture { get; set; } = new int[] { 400, 300 };

        /// <summary>
        /// Gets or sets the activation function for the actor network.
        /// </summary>
        public ActivationFunction ActorActivationFunction { get; set; } = ActivationFunction.ReLU;

        /// <summary>
        /// Gets or sets the final activation function for the actor network.
        /// </summary>
        /// <remarks>
        /// This is applied to the output layer of the actor to bound the actions.
        /// For example, Tanh for actions in the range [-1, 1].
        /// </remarks>
        public ActivationFunction ActorFinalActivationFunction { get; set; } = ActivationFunction.Tanh;

        /// <summary>
        /// Gets or sets the activation function for the critic network.
        /// </summary>
        public ActivationFunction CriticActivationFunction { get; set; } = ActivationFunction.ReLU;

        /// <summary>
        /// Gets or sets the learning rate for the actor network.
        /// </summary>
        public double ActorLearningRate { get; set; } = 0.0001;

        /// <summary>
        /// Gets or sets the learning rate for the critic network.
        /// </summary>
        public double CriticLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the soft update factor (tau) for target networks.
        /// </summary>
        /// <remarks>
        /// Controls how quickly the target networks are updated to match the main networks.
        /// Small values (e.g., 0.001-0.01) result in slower, more stable updates.
        /// </remarks>
        public new double Tau { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the Ornstein-Uhlenbeck noise theta parameter.
        /// </summary>
        /// <remarks>
        /// Controls the mean reversion rate in the noise process.
        /// Higher values cause the noise to revert to the mean faster.
        /// </remarks>
        public double OUNoiseTheta { get; set; } = 0.15;

        /// <summary>
        /// Gets or sets the Ornstein-Uhlenbeck noise sigma parameter.
        /// </summary>
        /// <remarks>
        /// Controls the volatility of the noise process.
        /// Higher values result in more exploration.
        /// </remarks>
        public double OUNoiseSigma { get; set; } = 0.2;

        /// <summary>
        /// Gets or sets a value indicating whether to use Gaussian noise instead of Ornstein-Uhlenbeck.
        /// </summary>
        public bool UseGaussianNoise { get; set; } = false;

        /// <summary>
        /// Gets or sets the standard deviation for Gaussian action noise.
        /// </summary>
        public double GaussianNoiseStdDev { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets a value indicating whether to use parameter noise.
        /// </summary>
        /// <remarks>
        /// Parameter noise adds noise to the network parameters instead of the actions.
        /// This can lead to more consistent exploration.
        /// </remarks>
        public bool UseParameterNoise { get; set; } = false;

        /// <summary>
        /// Gets or sets the initial standard deviation for parameter noise.
        /// </summary>
        public double ParameterNoiseInitialStdDev { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets a value indicating whether to use prioritized experience replay.
        /// </summary>
        public new bool UsePrioritizedReplay { get; set; } = true;

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
        /// Gets or sets the batch size for training.
        /// </summary>
        public new int BatchSize { get; set; } = 64;

        /// <summary>
        /// Gets or sets the noise decay rate.
        /// </summary>
        /// <remarks>
        /// Controls how quickly the exploration noise decays over time.
        /// </remarks>
        public double NoiseDecayRate { get; set; } = 0.9999;

        /// <summary>
        /// Gets or sets the minimum noise scale.
        /// </summary>
        /// <remarks>
        /// The noise scale will not decay below this value.
        /// </remarks>
        public double MinNoiseScale { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets a value indicating whether to use Layer Normalization.
        /// </summary>
        public bool UseLayerNormalization { get; set; } = false;

        /// <summary>
        /// Gets or sets a value indicating whether to use gradient clipping.
        /// </summary>
        public bool UseGradientClipping { get; set; } = true;


        /// <summary>
        /// Gets or sets a value indicating whether to use L2 regularization.
        /// </summary>
        public bool UseL2Regularization { get; set; } = true;

        /// <summary>
        /// Gets or sets the L2 regularization coefficient.
        /// </summary>
        public double L2RegularizationCoefficient { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets the warm-up steps before starting training.
        /// </summary>
        /// <remarks>
        /// The agent will collect this many steps of experience with random actions
        /// before starting to train.
        /// </remarks>
        public int WarmUpSteps { get; set; } = 10000;

        /// <summary>
        /// Gets or sets the optimizer type for the actor network.
        /// </summary>
        public OptimizerType ActorOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets the optimizer type for the critic network.
        /// </summary>
        public OptimizerType CriticOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets a value indicating whether to use a separate target critic for the actor update.
        /// </summary>
        public bool UseSeparateTargetCriticForActorUpdate { get; set; } = true;

        /// <summary>
        /// Gets or sets the actor hidden layer sizes (alias for ActorNetworkArchitecture).
        /// </summary>
        public int[] ActorHiddenLayers => ActorNetworkArchitecture;

        /// <summary>
        /// Gets or sets the critic hidden layer sizes (alias for CriticNetworkArchitecture).
        /// </summary>
        public int[] CriticHiddenLayers => CriticNetworkArchitecture;
    }
}