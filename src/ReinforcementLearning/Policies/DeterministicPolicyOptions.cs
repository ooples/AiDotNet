using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for deterministic policies: <see cref="DeterministicPolicy{T}"/>'s
    /// <see cref="DeterministicPolicy{T}(DeterministicPolicyOptions{T})"/> constructor builds its network from them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The actor of DDPG (Lillicrap et al., 2016) and TD3 (Fujimoto et al., 2018): the network maps a state straight to
    /// one action, and exploration comes from noise added to it during training.
    /// </para>
    /// <para><b>For Beginners:</b> The policy always gives the same action for the same situation; during training a
    /// little noise is added so it tries nearby actions too.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DeterministicPolicyOptions<T> : ModelOptions
    {
        /// <summary>Gets or sets the number of features in an observation (the network's input width).</summary>
        /// <value>Must be positive; 0 (the default) means unset.</value>
        public int StateSize { get; set; } = 0;

        /// <summary>Gets or sets the number of action dimensions (the network's output width).</summary>
        /// <value>Must be positive.</value>
        public int ActionSize { get; set; } = 0;

        /// <summary>Gets or sets the width of each hidden layer of the policy network, in order.</summary>
        /// <value>Defaults to [256, 256], TD3's actor; each layer is dense with ReLU.</value>
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };

        /// <summary>Gets or sets the loss the policy network is trained with when it is fitted directly.</summary>
        /// <value>Defaults to mean squared error.</value>
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

        /// <summary>Gets or sets the exploration noise added to actions during training.</summary>
        /// <value>
        /// With no explicit strategy, each read creates fresh Ornstein-Uhlenbeck noise with θ = 0.15 and σ = 0.2, DDPG's choice, sized to
        /// <see cref="ActionSize"/>. The default used to be that noise built for one action, which threw on the first
        /// exploratory step of any policy with more than one action dimension.
        /// </value>
        public IExplorationStrategy<T>? ExplorationStrategy
        {
            get => _explorationStrategy ?? new OrnsteinUhlenbeckNoise<T>(Math.Max(1, ActionSize));
            set => _explorationStrategy = value;
        }

        private IExplorationStrategy<T>? _explorationStrategy;

        /// <summary>Gets or sets whether actions pass through tanh, bounding them to [-1, 1].</summary>
        /// <value>Defaults to true, as DDPG's and TD3's actors do.</value>
        public bool UseTanhSquashing { get; set; } = true;

    }
}
