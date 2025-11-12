using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Rainbow DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Rainbow DQN combines six extensions to DQN:
/// 1. Double Q-learning: Reduces overestimation bias
/// 2. Dueling networks: Separates value and advantage streams
/// 3. Prioritized replay: Samples important experiences more frequently
/// 4. Multi-step learning: Uses n-step returns for better credit assignment
/// 5. Distributional RL: Learns full distribution of returns (C51)
/// 6. Noisy networks: Parameter noise for exploration
/// </remarks>
public class RainbowDQNOptions<T>
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredError<T>();

    // Epsilon-greedy parameters (optional, can use noisy networks instead)
    public double EpsilonStart { get; set; } = 1.0;
    public double EpsilonEnd { get; set; } = 0.01;
    public double EpsilonDecay { get; set; } = 0.995;
    public bool UseNoisyNetworks { get; set; } = true;

    // Standard DQN parameters
    public int BatchSize { get; set; } = 32;
    public int ReplayBufferSize { get; set; } = 100000;
    public int TargetUpdateFrequency { get; set; } = 1000;
    public int WarmupSteps { get; set; } = 10000;

    // Dueling network architecture
    public List<int> SharedLayers { get; set; } = [128];
    public List<int> ValueStreamLayers { get; set; } = [128];
    public List<int> AdvantageStreamLayers { get; set; } = [128];

    // Prioritized experience replay parameters
    public bool UsePrioritizedReplay { get; set; } = true;
    public double PriorityAlpha { get; set; } = 0.6;
    public double PriorityBeta { get; set; } = 0.4;
    public double PriorityBetaIncrement { get; set; } = 0.001;
    public double PriorityEpsilon { get; set; } = 1e-6;

    // Multi-step learning parameters
    public int NSteps { get; set; } = 3;

    // Distributional RL (C51) parameters
    public bool UseDistributional { get; set; } = true;
    public int NumAtoms { get; set; } = 51;
    public double VMin { get; set; } = -10.0;
    public double VMax { get; set; } = 10.0;

    // Noisy networks parameters
    public double NoisyNetSigma { get; set; } = 0.5;

    public int? Seed { get; set; }

    public RainbowDQNOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        LearningRate = numOps.FromDouble(0.0001);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
