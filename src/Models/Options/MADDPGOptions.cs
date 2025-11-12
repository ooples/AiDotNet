using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Multi-Agent DDPG (MADDPG) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MADDPG extends DDPG to multi-agent settings with centralized training and
/// decentralized execution. Critics observe all agents during training.
/// </para>
/// <para><b>For Beginners:</b>
/// MADDPG allows multiple agents to learn together in shared environments.
/// During training, agents can "see" what others are doing (centralized critics),
/// but during execution, each agent acts independently (decentralized actors).
///
/// Key features:
/// - **Centralized Training**: Critics see all agents' observations and actions
/// - **Decentralized Execution**: Actors only use their own observations
/// - **Continuous Actions**: Based on DDPG for continuous control
/// - **Cooperative or Competitive**: Works for both settings
///
/// Think of it like: Team sports where players practice together (centralized)
/// but during the game each player makes their own decisions (decentralized).
///
/// Examples: Robot coordination, traffic control, multi-player games
/// </para>
/// </remarks>
public class MADDPGOptions<T>
{
    public int NumAgents { get; set; }
    public int StateSize { get; set; }  // Per-agent state size
    public int ActionSize { get; set; }  // Per-agent action size
    public T ActorLearningRate { get; set; }
    public T CriticLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T TargetUpdateTau { get; set; }

    // MADDPG-specific
    public int BatchSize { get; set; } = 64;
    public int ReplayBufferSize { get; set; } = 1000000;
    public int WarmupSteps { get; set; } = 10000;
    public double ExplorationNoise { get; set; } = 0.1;

    public List<int> ActorHiddenLayers { get; set; } = [128, 128];
    public List<int> CriticHiddenLayers { get; set; } = [128, 128];
    public int? Seed { get; set; }

    public MADDPGOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        ActorLearningRate = numOps.FromDouble(0.0001);
        CriticLearningRate = numOps.FromDouble(0.001);
        DiscountFactor = numOps.FromDouble(0.99);
        TargetUpdateTau = numOps.FromDouble(0.001);
    }
}
