using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.TD3;

/// <summary>
/// Configuration options for TD3 agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TD3Options<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public T ActorLearningRate { get; init; }
    public T CriticLearningRate { get; init; }
    public T DiscountFactor { get; init; }
    public T TargetUpdateTau { get; init; }
    public ILossFunction<T> CriticLossFunction { get; init; }
    public int BatchSize { get; init; } = 256;
    public int ReplayBufferSize { get; init; } = 1000000;
    public int WarmupSteps { get; init; } = 25000;

    /// <summary>
    /// Delay policy updates: update actor every N critic updates.
    /// </summary>
    public int PolicyUpdateFrequency { get; init; } = 2;

    /// <summary>
    /// Standard deviation for exploration noise.
    /// </summary>
    public double ExplorationNoise { get; init; } = 0.1;

    /// <summary>
    /// Standard deviation for target policy smoothing noise.
    /// </summary>
    public double TargetPolicyNoise { get; init; } = 0.2;

    /// <summary>
    /// Clip range for target policy smoothing noise.
    /// </summary>
    public double TargetNoiseClip { get; init; } = 0.5;

    public int[] ActorHiddenLayers { get; init; } = new[] { 256, 256 };
    public int[] CriticHiddenLayers { get; init; } = new[] { 256, 256 };
    public int? Seed { get; init; }

    public static TD3Options<T> Default(int stateSize, int actionSize, T actorLr, T criticLr, T gamma)
    {
        var numOps = NumericOperations<T>.Instance;
        return new TD3Options<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            ActorLearningRate = actorLr,
            CriticLearningRate = criticLr,
            DiscountFactor = gamma,
            TargetUpdateTau = numOps.FromDouble(0.005),
            CriticLossFunction = new MeanSquaredError<T>()
        };
    }
}
