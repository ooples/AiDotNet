using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.DDPG;

/// <summary>
/// Configuration options for DDPG agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DDPGOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public T ActorLearningRate { get; init; }
    public T CriticLearningRate { get; init; }
    public T DiscountFactor { get; init; }
    public T TargetUpdateTau { get; init; }
    public ILossFunction<T> CriticLossFunction { get; init; }
    public int BatchSize { get; init; } = 64;
    public int ReplayBufferSize { get; init; } = 1000000;
    public int WarmupSteps { get; init; } = 1000;
    public double ExplorationNoise { get; init; } = 0.1;
    public int[] ActorHiddenLayers { get; init; } = new[] { 400, 300 };
    public int[] CriticHiddenLayers { get; init; } = new[] { 400, 300 };
    public int? Seed { get; init; }

    public static DDPGOptions<T> Default(int stateSize, int actionSize, T actorLr, T criticLr, T gamma)
    {
        var numOps = NumericOperations<T>.Instance;
        return new DDPGOptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            ActorLearningRate = actorLr,
            CriticLearningRate = criticLr,
            DiscountFactor = gamma,
            TargetUpdateTau = numOps.FromDouble(0.001),
            CriticLossFunction = new MeanSquaredError<T>()
        };
    }
}
