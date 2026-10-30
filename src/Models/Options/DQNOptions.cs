using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Deep Q-Network (DQN) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DQNOptions<T> : ModelOptions
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }

    /// <summary>
    /// Discount factor gamma applied to future rewards in the Q-learning target. Paper value: 0.99.
    /// </summary>
    [PaperDefault(0.99, Source = "Mnih et al. 2015 (Nature 518), Extended Data Table 1: discount factor")]
    public T DiscountFactor { get; set; }

    /// <summary>
    /// Initial epsilon of the epsilon-greedy exploration schedule. Paper value: 1.0.
    /// </summary>
    [PaperDefault(1.0, Source = "Mnih et al. 2015 (Nature 518), Extended Data Table 1: initial exploration")]
    public double EpsilonStart { get; set; } = 1.0;
    public double EpsilonEnd { get; set; } = 0.01;
    public double EpsilonDecay { get; set; } = 0.995;
    public int BatchSize { get; set; } = 64;
    public int ReplayBufferSize { get; set; } = 100000;
    public int TargetUpdateFrequency { get; set; } = 1000;
    public int WarmupSteps { get; set; } = 1000;
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Optimizer for Q-network training. Default: Adam (per Mnih et al. 2015).
    /// </summary>
    public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; set; }

    public List<int> HiddenLayers { get; set; } = new List<int> { 128, 128 };

    public DQNOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
