using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for discrete policies.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DiscretePolicyOptions<T>
    {
        public int StateSize { get; set; } = 0;
        public int ActionSize { get; set; } = 0;
        public int[] HiddenLayers { get; set; } = new int[] { 128, 128 };
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new EpsilonGreedyExploration<T>();
        public int? Seed { get; set; } = null;
    }
}
