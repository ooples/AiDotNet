using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Dueling DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DuelingDQNOptions<T>
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredError<T>();
    public double EpsilonStart { get; set; } = 1.0;
    public double EpsilonEnd { get; set; } = 0.01;
    public double EpsilonDecay { get; set; } = 0.995;
    public int BatchSize { get; set; } = 32;
    public int ReplayBufferSize { get; set; } = 10000;
    public int TargetUpdateFrequency { get; set; } = 1000;
    public int WarmupSteps { get; set; } = 1000;
    public List<int> SharedLayers { get; set; } = [128];
    public List<int> ValueStreamLayers { get; set; } = [128];
    public List<int> AdvantageStreamLayers { get; set; } = [128];
    public int? Seed { get; set; }

    public DuelingDQNOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        LearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
