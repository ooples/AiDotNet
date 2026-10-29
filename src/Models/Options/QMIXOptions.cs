using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

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
public class QMIXOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of cooperating agents.</summary>
    public int NumAgents { get; init; }
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }  // Per-agent observation size
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }  // Per-agent action size
    /// <summary>Number of features in the global state the mixing network's hypernetworks condition on.</summary>
    public int GlobalStateSize { get; init; }  // Global state for mixing network

    // Network architectures
    /// <summary>Hidden layer sizes of each agent's Q-network.</summary>
    public List<int> AgentHiddenLayers { get; init; } = new List<int> { 64 };
    /// <summary>Hidden layer sizes of the mixing network that combines per-agent Q-values.</summary>
    public List<int> MixingHiddenLayers { get; init; } = new List<int> { 32 };

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
