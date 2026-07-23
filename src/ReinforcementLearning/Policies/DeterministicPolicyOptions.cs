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
        public IExplorationStrategy<T> ExplorationStrategy { get; set; } = new OrnsteinUhlenbeckNoise<T>(actionSize: 1);
        public bool UseTanhSquashing { get; set; } = true;
        public new int? Seed { get; set; } = null;
    }
}
