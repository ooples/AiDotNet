using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Configuration options for continuous policies.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ContinuousPolicyOptions<T>
    {
        public int StateSize { get; set; } = 0;
        public int ActionSize { get; set; } = 0;
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };
        public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new GaussianNoiseExploration<T>();
        public bool UseTanhSquashing { get; set; } = false;
        public int? Seed { get; set; } = null;
    }
}
