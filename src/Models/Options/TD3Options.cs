using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TD3 agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TD3Options<T>
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T ActorLearningRate { get; set; }
    public T CriticLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T TargetUpdateTau { get; set; }
    public ILossFunction<T> CriticLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    public int BatchSize { get; set; } = 256;
    public int ReplayBufferSize { get; set; } = 1000000;
    public int WarmupSteps { get; set; } = 25000;
    public int PolicyUpdateFrequency { get; set; } = 2;
    public double ExplorationNoise { get; set; } = 0.1;
    public double TargetPolicyNoise { get; set; } = 0.2;
    public double TargetNoiseClip { get; set; } = 0.5;
    public List<int> ActorHiddenLayers { get; set; } = new List<int> { 256, 256 };
    public List<int> CriticHiddenLayers { get; set; } = new List<int> { 256, 256 };
    public int? Seed { get; set; }

    public TD3Options()
    {
        var numOps = NumericOperations<T>.Instance;
        ActorLearningRate = numOps.FromDouble(0.001);
        CriticLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        TargetUpdateTau = numOps.FromDouble(0.005);
    }
}
