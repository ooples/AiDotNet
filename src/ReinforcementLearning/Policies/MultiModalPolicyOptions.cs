using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for multi-modal mixture of Gaussians policies.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MultiModalPolicyOptions<T> : ModelOptions
    {
        public int StateSize { get; set; } = 0;
        public int ActionSize { get; set; } = 0;
        public int NumComponents { get; set; } = 3;
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new NoExploration<T>();
        public new int? Seed { get; set; } = null;
    }
}
