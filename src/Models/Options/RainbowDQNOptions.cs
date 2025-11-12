using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

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
public class RainbowDQNOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; }
    public int ActionSize { get; init; }
    public bool UseNoisyNetworks { get; init; } = true;

    // Dueling network architecture
    public List<int> SharedLayers { get; init; } = [128];
    public List<int> ValueStreamLayers { get; init; } = [128];
    public List<int> AdvantageStreamLayers { get; init; } = [128];

    // Prioritized experience replay parameters (base has UsePrioritizedReplay)
    public double PriorityAlpha { get; init; } = 0.6;
    public double PriorityBeta { get; init; } = 0.4;
    public double PriorityBetaIncrement { get; init; } = 0.001;
    public double PriorityEpsilon { get; init; } = 1e-6;

    // Multi-step learning parameters
    public int NSteps { get; init; } = 3;

    // Distributional RL (C51) parameters
    public bool UseDistributional { get; init; } = true;
    public int NumAtoms { get; init; } = 51;
    public double VMin { get; init; } = -10.0;
    public double VMax { get; init; } = 10.0;

    // Noisy networks parameters
    public double NoisyNetSigma { get; init; } = 0.5;

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
