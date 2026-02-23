using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for mixed discrete and continuous policies.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MixedPolicyOptions<T> : ModelOptions
    {
        public int StateSize { get; set; } = 0;
        public int DiscreteActionSize { get; set; } = 0;
        public int ContinuousActionSize { get; set; } = 0;
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
        public IExplorationStrategy<T> DiscreteExplorationStrategy { get; set; } = new EpsilonGreedyExploration<T>();
        public IExplorationStrategy<T> ContinuousExplorationStrategy { get; set; } = new GaussianNoiseExploration<T>();
        public bool SharedFeatures { get; set; } = false;
        public new int? Seed { get; set; } = null;
    }
}
