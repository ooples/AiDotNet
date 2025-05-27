using AiDotNet.Enums;

namespace AiDotNet.ReinforcementLearning.Models.Options
{
    /// <summary>
    /// Options for the Policy Gradient algorithms (REINFORCE).
    /// </summary>
    public class PolicyGradientOptions : ReinforcementLearningOptions
    {
        /// <summary>
        /// Gets or sets a value indicating whether to use a baseline for variance reduction.
        /// </summary>
        /// <remarks>
        /// A baseline (typically the value function) helps reduce the variance of gradient estimates.
        /// </remarks>
        public bool UseBaseline { get; set; } = true;

        /// <summary>
        /// Gets or sets the network architecture for the policy network.
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] PolicyNetworkArchitecture { get; set; } = new int[] { 64, 64 };

        /// <summary>
        /// Gets or sets the network architecture for the value network (if baseline is used).
        /// </summary>
        /// <remarks>
        /// Specifies the hidden layer sizes for the neural network.
        /// </remarks>
        public int[] ValueNetworkArchitecture { get; set; } = new int[] { 64, 64 };

        /// <summary>
        /// Gets or sets the activation function for the policy network.
        /// </summary>
        public ActivationFunction PolicyActivationFunction { get; set; } = ActivationFunction.Tanh;

        /// <summary>
        /// Gets or sets the activation function for the value network.
        /// </summary>
        public ActivationFunction ValueActivationFunction { get; set; } = ActivationFunction.Tanh;

        /// <summary>
        /// Gets or sets the entropy coefficient for the policy.
        /// </summary>
        /// <remarks>
        /// Adding an entropy term to the objective encourages exploration.
        /// Higher values result in more exploration.
        /// </remarks>
        public double EntropyCoefficient { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets the learning rate for the policy network.
        /// </summary>
        public double PolicyLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the learning rate for the value network.
        /// </summary>
        public double ValueLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the optimizer type for the policy network.
        /// </summary>
        public OptimizerType PolicyOptimizerType { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets the optimizer type for the value network.
        /// </summary>
        public OptimizerType ValueOptimizerType { get; set; } = OptimizerType.Adam;

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
        public bool UseGAE { get; set; } = false;

        /// <summary>
        /// Gets or sets the GAE lambda parameter.
        /// </summary>
        /// <remarks>
        /// Controls the bias-variance tradeoff in advantage estimation. Higher values
        /// (closer to 1) introduce less bias but more variance.
        /// </remarks>
        public double GAELambda { get; set; } = 0.95;

        /// <summary>
        /// Gets or sets the number of steps to collect before updating the policy.
        /// </summary>
        public int StepsPerUpdate { get; set; } = 2048;

        /// <summary>
        /// Gets or sets the minimum episode length before allowing updates.
        /// </summary>
        public int MinEpisodeLength { get; set; } = 50;

        /// <summary>
        /// Gets or sets a value indicating whether to standardize rewards.
        /// </summary>
        public bool StandardizeRewards { get; set; } = true;

        /// <summary>
        /// Gets or sets the standard deviation of the initial policy (for continuous action spaces).
        /// </summary>
        public double InitialPolicyStdDev { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets a value indicating whether to use a fixed standard deviation for the policy.
        /// </summary>
        public bool UseFixedPolicyStdDev { get; set; } = false;

        /// <summary>
        /// Gets or sets a value indicating whether to reuse past trajectories for multiple updates.
        /// </summary>
        public bool ReuseTrajectories { get; set; } = false;

        /// <summary>
        /// Gets or sets the number of times to reuse each trajectory.
        /// </summary>
        public int TrajectoryReuseCount { get; set; } = 3;
    }
}