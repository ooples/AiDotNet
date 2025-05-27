using AiDotNet.Enums;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for Actor-Critic algorithms (A2C, A3C).
    /// </summary>
    public class ActorCriticOptions<T> : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets the network architecture for the actor (policy) network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] ActorNetworkArchitecture { get; set; } = new int[] { 64, 64 };

        /// <summary>
        /// Gets or sets the network architecture for the critic (value) network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] CriticNetworkArchitecture { get; set; } = new int[] { 64, 64 };

        /// <summary>
        /// Gets or sets a value indicating whether to use shared layers for actor and critic.
        /// </summary>
        /// <remarks>
        /// If true, the actor and critic will share the initial layers of the network.
        /// </remarks>
        public bool UseSharedLayers { get; set; } = true;

        /// <summary>
        /// Gets or sets the number of shared layers if using shared layers.
        /// </summary>
        public int NumSharedLayers { get; set; } = 1;

        /// <summary>
        /// Gets or sets the activation function for the actor network.
        /// </summary>
        public IActivationFunction<T> ActorActivationFunction { get; set; } = new TanhActivation<T>();

        /// <summary>
        /// Gets or sets the activation function for the critic network.
        /// </summary>
        public IActivationFunction<T> CriticActivationFunction { get; set; } = new TanhActivation<T>();

        /// <summary>
        /// Gets or sets the entropy coefficient for the policy.
        /// </summary>
        /// <remarks>
        /// Adding an entropy term to the objective encourages exploration.
        /// Higher values result in more exploration.
        /// </remarks>
        public double EntropyCoefficient { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets the value loss coefficient.
        /// </summary>
        /// <remarks>
        /// Controls the relative weight of the value function loss in the overall loss.
        /// </remarks>
        public double ValueLossCoefficient { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets the learning rate for the actor network.
        /// </summary>
        public double ActorLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the learning rate for the critic network.
        /// </summary>
        public double CriticLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the optimizer type for the actor network.
        /// </summary>
        public OptimizerType ActorOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets the optimizer type for the critic network.
        /// </summary>
        public OptimizerType CriticOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets a value indicating whether to normalize the advantage function.
        /// </summary>
        /// <remarks>
        /// Normalizing advantages can help reduce training instability.
        /// </remarks>
        public bool NormalizeAdvantages { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use generalized advantage estimation (GAE).
        /// </summary>
        /// <remarks>
        /// GAE provides a way to reduce variance while maintaining an acceptable level of bias.
        /// </remarks>
        public bool UseGAE { get; set; } = true;

        /// <summary>
        /// Gets or sets the GAE lambda parameter.
        /// </summary>
        /// <remarks>
        /// Controls the bias-variance tradeoff in advantage estimation. Higher values
        /// (closer to 1) introduce less bias but more variance.
        /// </remarks>
        public double GAELambda { get; set; } = 0.95;

        /// <summary>
        /// Gets or sets the number of steps to collect before updating the networks.
        /// </summary>
        public int StepsPerUpdate { get; set; } = 5;

        /// <summary>
        /// Gets or sets a value indicating whether to use n-step returns.
        /// </summary>
        public bool UseNStepReturns { get; set; } = true;

        /// <summary>
        /// Gets or sets the number of steps for n-step returns.
        /// </summary>
        public int NSteps { get; set; } = 5;

        /// <summary>
        /// Gets or sets a value indicating whether to use asynchronous updates (A3C).
        /// </summary>
        /// <remarks>
        /// If true, multiple workers will collect experiences and update the networks asynchronously.
        /// </remarks>
        public bool UseAsynchronousUpdates { get; set; } = false;

        /// <summary>
        /// Gets or sets the number of parallel workers for asynchronous updates.
        /// </summary>
        public int NumWorkers { get; set; } = 4;

        /// <summary>
        /// Gets or sets the maximum gradient norm for gradient clipping.
        /// </summary>
        public new double MaxGradientNorm { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets the standard deviation of the initial policy (for continuous action spaces).
        /// </summary>
        public double InitialPolicyStdDev { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets a value indicating whether to learn the standard deviation of the policy.
        /// </summary>
        /// <remarks>
        /// If true, the policy will output both mean and standard deviation for continuous actions.
        /// If false, a fixed standard deviation will be used.
        /// </remarks>
        public bool LearnPolicyStdDev { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to standardize rewards.
        /// </summary>
        public bool StandardizeRewards { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to use a target network for the critic.
        /// </summary>
        public bool UseCriticTargetNetwork { get; set; } = false;

        /// <summary>
        /// Gets or sets the soft update factor (tau) for the critic target network.
        /// </summary>
        public double CriticTargetUpdateTau { get; set; } = 0.005;
    }
}