using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for QMIX agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// QMIX factorizes joint action-values into per-agent values using a mixing network.
/// This enables decentralized execution while maintaining centralized training.
/// </para>
/// <para><b>For Beginners:</b>
/// QMIX solves multi-agent problems by learning individual Q-values for each agent,
/// then combining them with a "mixing network" that ensures the team's joint action
/// is consistent with individual actions.
///
/// Key features:
/// - **Value Factorization**: Decomposes team value into agent values
/// - **Mixing Network**: Combines agent Q-values monotonically
/// - **Decentralized Execution**: Each agent acts independently
/// - **Discrete Actions**: Value-based method for discrete action spaces
///
/// Think of it like: Each team member estimates their contribution, and a coach
/// (mixing network) combines these to determine the team's overall performance.
///
/// Famous for: StarCraft micromanagement, cooperative games
/// </para>
/// </remarks>
public class QMIXOptions<T>
{
    public int NumAgents { get; set; }
    public int StateSize { get; set; }  // Per-agent observation size
    public int ActionSize { get; set; }  // Per-agent action size
    public int GlobalStateSize { get; set; }  // Global state for mixing network
    public T LearningRate { get; set; }
    public T DiscountFactor { get; set; }

    // QMIX-specific
    public double EpsilonStart { get; set; } = 1.0;
    public double EpsilonEnd { get; set; } = 0.05;
    public double EpsilonDecay { get; set; } = 0.9995;
    public int BatchSize { get; set; } = 32;
    public int ReplayBufferSize { get; set; } = 5000;
    public int TargetUpdateFrequency { get; set; } = 200;

    // Network architectures
    public List<int> AgentHiddenLayers { get; set; } = [64];
    public List<int> MixingHiddenLayers { get; set; } = [32];
    public int? Seed { get; set; }

    public QMIXOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        LearningRate = numOps.FromDouble(0.0005);
        DiscountFactor = numOps.FromDouble(0.99);
    }
}
