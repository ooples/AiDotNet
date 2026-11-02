using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for Beta distribution policies: <see cref="BetaPolicy{T}"/>'s
    /// <see cref="BetaPolicy{T}(BetaPolicyOptions{T})"/> constructor builds its network from them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A Beta policy (Chou, Maturana and Scherer, 2017) samples each action dimension from a Beta(α, β) distribution on
    /// [0, 1] and rescales it to [<see cref="ActionMin"/>, <see cref="ActionMax"/>]. Unlike a clipped Gaussian it puts no
    /// probability mass outside the bounds, so bounded controls need no clipping bias.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when every action has a hard minimum and maximum (a throttle from 0 to 1, a
    /// valve from closed to open). The network predicts two shape numbers per action, and the policy draws actions that
    /// always land inside the allowed range.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class BetaPolicyOptions<T> : ModelOptions
    {
        /// <summary>Gets or sets the number of features in an observation (the network's input width).</summary>
        /// <value>Must be positive; 0 (the default) means unset.</value>
        public int StateSize { get; set; } = 0;

        /// <summary>Gets or sets the number of action dimensions.</summary>
        /// <value>Must be positive; the network outputs α and β for each dimension, 2 × ActionSize values.</value>
        public int ActionSize { get; set; } = 0;

        /// <summary>Gets or sets the width of each hidden layer of the policy network, in order.</summary>
        /// <value>Defaults to [256, 256], the actor of SAC and TD3; each layer is dense with ReLU.</value>
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };

        /// <summary>Gets or sets the loss the policy network is trained with when it is fitted directly.</summary>
        /// <value>Defaults to mean squared error.</value>
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

        /// <summary>Gets or sets the exploration applied to sampled actions during training.</summary>
        /// <value>Defaults to none: sampling from the Beta distribution already explores.</value>
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new NoExploration<T>();

        /// <summary>Gets or sets the lower bound of every action dimension.</summary>
        /// <value>Defaults to 0; must be below <see cref="ActionMax"/>.</value>
        public double ActionMin { get; set; } = 0.0;

        /// <summary>Gets or sets the upper bound of every action dimension.</summary>
        /// <value>Defaults to 1; must be above <see cref="ActionMin"/>.</value>
        public double ActionMax { get; set; } = 1.0;

    }
}
