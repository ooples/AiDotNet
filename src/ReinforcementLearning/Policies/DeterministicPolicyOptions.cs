using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for deterministic policies.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DeterministicPolicyOptions<T> : ModelOptions
    {
        public int StateSize { get; set; } = 0;
        public int ActionSize { get; set; } = 0;
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
        private IExplorationStrategy<T>? _explorationStrategy;

        /// <summary>
        /// The exploration strategy. Unless one is assigned, every read returns NEW Ornstein-Uhlenbeck noise sized to
        /// <see cref="ActionSize"/> (1 while that is unset): the noise carries state, so two policies built from one
        /// options instance must each get their own, and setting ActionSize after construction still gives noise of
        /// the right size. An assigned strategy is returned as is.
        /// </summary>
        public IExplorationStrategy<T> ExplorationStrategy
        {
            get => _explorationStrategy ?? new OrnsteinUhlenbeckNoise<T>(actionSize: Math.Max(1, ActionSize));
            set => _explorationStrategy = value ?? throw new ArgumentNullException(nameof(value));
        }
        public bool UseTanhSquashing { get; set; } = true;
    }
}
