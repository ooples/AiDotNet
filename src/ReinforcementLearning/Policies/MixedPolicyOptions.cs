using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for mixed discrete and continuous policies: <see cref="MixedPolicy{T}"/>'s
    /// <see cref="MixedPolicy{T}(MixedPolicyOptions{T})"/> constructor builds its networks from them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A mixed (parameterized) action space pairs one categorical choice with real-valued parameters, as in
    /// parameterized-action RL (Masson, Ranchod and Konidaris, 2016; Xiong et al., 2018, P-DQN). The policy samples a
    /// one-hot discrete action from softmax logits and a Gaussian continuous action from a mean and log standard
    /// deviation per dimension; the action vector is [discrete one-hot, continuous values].
    /// </para>
    /// <para><b>For Beginners:</b> Use this when each step makes a choice and sets some amounts, such as "which tool"
    /// plus "how hard to press".
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MixedPolicyOptions<T> : ModelOptions
    {
        /// <summary>Gets or sets the number of features in an observation (the networks' input width).</summary>
        /// <value>Must be positive; 0 (the default) means unset.</value>
        public int StateSize { get; set; } = 0;

        /// <summary>Gets or sets the number of discrete choices (the softmax head's width).</summary>
        /// <value>Must be positive.</value>
        public int DiscreteActionSize { get; set; } = 0;

        /// <summary>Gets or sets the number of continuous action dimensions.</summary>
        /// <value>Must be positive; the continuous head outputs a mean and a log standard deviation for each, 2 × this.</value>
        public int ContinuousActionSize { get; set; } = 0;

        /// <summary>Gets or sets the width of each hidden layer, in order (of the trunk when <see cref="SharedFeatures"/>).</summary>
        /// <value>Defaults to [256, 256]; each layer is dense with ReLU.</value>
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };

        /// <summary>Gets or sets the loss the networks are trained with when they are fitted directly.</summary>
        /// <value>Defaults to mean squared error.</value>
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

        /// <summary>Gets or sets the exploration applied to the sampled discrete action during training.</summary>
        /// <value>Defaults to epsilon-greedy.</value>
        public IExplorationStrategy<T> DiscreteExplorationStrategy { get; set; } = new EpsilonGreedyExploration<T>();

        /// <summary>Gets or sets the exploration applied to the sampled continuous action during training.</summary>
        /// <value>Defaults to Gaussian noise.</value>
        public IExplorationStrategy<T> ContinuousExplorationStrategy { get; set; } = new GaussianNoiseExploration<T>();

        /// <summary>
        /// Gets or sets whether both heads read one shared feature trunk instead of two separate networks.
        /// </summary>
        /// <value>
        /// False (the default) builds two networks. True builds one network whose output is
        /// [DiscreteActionSize logits, ContinuousActionSize means, ContinuousActionSize log standard deviations], so both
        /// heads learn from the same features and the state is evaluated once per call.
        /// </value>
        public bool SharedFeatures { get; set; } = false;

        /// <summary>Gets or sets the seed of the policy's action sampling; null samples from a secure random source.</summary>
        public new int? Seed { get; set; } = null;
    }
}
