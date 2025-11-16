using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

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
public class MADDPGOptions<T> : ReinforcementLearningOptions<T>
{
    public int NumAgents { get; init; }
    public int StateSize { get; init; }  // Per-agent state size
    public int ActionSize { get; init; }  // Per-agent action size
    public T ActorLearningRate { get; init; }
    public T CriticLearningRate { get; init; }
    public T TargetUpdateTau { get; init; }

    // MADDPG-specific
    public double ExplorationNoise { get; init; } = 0.1;

    public List<int> ActorHiddenLayers { get; init; } = new List<int> { 128, 128 };
    public List<int> CriticHiddenLayers { get; init; } = new List<int> { 128, 128 };

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }

    public MADDPGOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ActorLearningRate = numOps.FromDouble(0.0001);
        CriticLearningRate = numOps.FromDouble(0.001);
        TargetUpdateTau = numOps.FromDouble(0.001);
    }

    /// <summary>
    /// Validates that required properties are set.
    /// </summary>
    public void Validate()
    {
        if (NumAgents <= 0)
            throw new ArgumentException("NumAgents must be greater than 0", nameof(NumAgents));
        if (StateSize <= 0)
            throw new ArgumentException("StateSize must be greater than 0", nameof(StateSize));
        if (ActionSize <= 0)
            throw new ArgumentException("ActionSize must be greater than 0", nameof(ActionSize));
    }
}
