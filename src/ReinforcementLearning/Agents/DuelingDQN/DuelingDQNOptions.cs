using AiDotNet.LossFunctions;

namespace AiDotNet.ReinforcementLearning.Agents.DuelingDQN;

/// <summary>
/// Configuration options for Dueling DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DuelingDQNOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public T LearningRate { get; init; }
    public T DiscountFactor { get; init; }
    public ILossFunction<T> LossFunction { get; init; }
    public double EpsilonStart { get; init; } = 1.0;
    public double EpsilonEnd { get; init; } = 0.01;
    public double EpsilonDecay { get; init; } = 0.995;
    public int BatchSize { get; init; } = 32;
    public int ReplayBufferSize { get; init; } = 10000;
    public int TargetUpdateFrequency { get; init; } = 1000;
    public int WarmupSteps { get; init; } = 1000;
    public int[] SharedLayers { get; init; } = new[] { 128 };
    public int[] ValueStreamLayers { get; init; } = new[] { 128 };
    public int[] AdvantageStreamLayers { get; init; } = new[] { 128 };
    public int? Seed { get; init; }

    public static DuelingDQNOptions<T> Default(int stateSize, int actionSize, T learningRate, T gamma)
    {
        return new DuelingDQNOptions<T>
        {
            StateSize = stateSize,
            ActionSize = actionSize,
            LearningRate = learningRate,
            DiscountFactor = gamma,
            LossFunction = new MeanSquaredError<T>()
        };
    }
}
