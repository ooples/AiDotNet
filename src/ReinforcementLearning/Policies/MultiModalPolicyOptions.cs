using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for multi-modal mixture-of-Gaussians policies: <see cref="MultiModalPolicy{T}"/>'s
    /// <see cref="MultiModalPolicy{T}(MultiModalPolicyOptions{T})"/> constructor builds its network from them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A mixture density network (Bishop, 1994) head: for each of <see cref="NumComponents"/> components the network
    /// outputs a mixing logit, a mean per action dimension and a log standard deviation per dimension, so a state can
    /// have several distinct good actions instead of one Gaussian averaging between them.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when there can be more than one right answer (go around an obstacle on the
    /// left or on the right). A single Gaussian would average the two and drive into the obstacle.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MultiModalPolicyOptions<T> : ModelOptions
    {
        /// <summary>Gets or sets the number of features in an observation (the network's input width).</summary>
        /// <value>Must be positive; 0 (the default) means unset.</value>
        public int StateSize { get; set; } = 0;

        /// <summary>Gets or sets the number of action dimensions.</summary>
        /// <value>Must be positive.</value>
        public int ActionSize { get; set; } = 0;

        /// <summary>Gets or sets the number of Gaussian components in the mixture.</summary>
        /// <value>Defaults to 3; the network outputs NumComponents × (1 + 2 × ActionSize) values.</value>
        public int NumComponents { get; set; } = 3;

        /// <summary>Gets or sets the width of each hidden layer of the policy network, in order.</summary>
        /// <value>Defaults to [256, 256]; each layer is dense with ReLU.</value>
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };

        /// <summary>Gets or sets the loss the policy network is trained with when it is fitted directly.</summary>
        /// <value>Defaults to mean squared error.</value>
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

        /// <summary>Gets or sets the exploration applied to sampled actions during training.</summary>
        /// <value>Defaults to none: sampling from the mixture already explores.</value>
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new NoExploration<T>();

    }
}
