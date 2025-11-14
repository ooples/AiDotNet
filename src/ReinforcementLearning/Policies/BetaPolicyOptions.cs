using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    public class BetaPolicyOptions<T>
    {
        public int StateSize { get; set; } = 0;
        public int ActionSize { get; set; } = 0;
        public int[] HiddenLayers { get; set; } = new int[] { 256, 256 };
        public ILossFunction<T>? LossFunction { get; set; } = null;
        public IExplorationStrategy<T>? ExplorationStrategy { get; set; } = null;
        public double ActionMin { get; set; } = 0.0;
        public double ActionMax { get; set; } = 1.0;
        public int? Seed { get; set; } = null;
    }
}
