using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TD3 agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TD3Options<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public T ActorLearningRate { get; init; }
    public T CriticLearningRate { get; init; }
    public T TargetUpdateTau { get; init; }
    public ILossFunction<T> CriticLossFunction { get; init; } = new MeanSquaredErrorLoss<T>();
    public int PolicyUpdateFrequency { get; init; } = 2;
    public double ExplorationNoise { get; init; } = 0.1;
    public double TargetPolicyNoise { get; init; } = 0.2;
    public double TargetNoiseClip { get; init; } = 0.5;
    public List<int> ActorHiddenLayers { get; init; } = new List<int> { 256, 256 };
    public List<int> CriticHiddenLayers { get; init; } = new List<int> { 256, 256 };
    public int WarmupSteps { get; init; } = 25000;

    public TD3Options()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ActorLearningRate = numOps.FromDouble(0.001);
        CriticLearningRate = numOps.FromDouble(0.001);
        TargetUpdateTau = numOps.FromDouble(0.005);
    }
}
