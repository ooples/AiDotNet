using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DDPG agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DDPGOptions<T> : ModelOptions
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T ActorLearningRate { get; set; }
    public T CriticLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T TargetUpdateTau { get; set; }
    public ILossFunction<T> CriticLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    public int BatchSize { get; set; } = 64;
    public int ReplayBufferSize { get; set; } = 1000000;
    public int WarmupSteps { get; set; } = 1000;
    public double ExplorationNoise { get; set; } = 0.1;
    public List<int> ActorHiddenLayers { get; set; } = new List<int> { 400, 300 };
    public List<int> CriticHiddenLayers { get; set; } = new List<int> { 400, 300 };

    public DDPGOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ActorLearningRate = numOps.FromDouble(0.0001);
        CriticLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        TargetUpdateTau = numOps.FromDouble(0.001);
    }
}
