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
    public int NumAgents { get; init; }
    public int StateSize { get; init; }  // Per-agent observation size
    public int ActionSize { get; init; }  // Per-agent action size
    public int GlobalStateSize { get; init; }  // Global state for mixing network

    // Network architectures
    public List<int> AgentHiddenLayers { get; init; } = new List<int> { 64 };
    public List<int> MixingHiddenLayers { get; init; } = new List<int> { 32 };

    /// <summary>
    /// Gets or sets a factory for the optimizer of the agent's joint training step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// QMIX trains every agent network and the mixing network on one TD loss in one step (Rashid et al. 2018),
    /// so the agent invokes the factory once and steps all of them with that optimizer. When this is
    /// <c>null</c>, the step uses plain SGD at the agent's learning rate.
    /// </para>
    /// <para>
    /// The factory must create a new optimizer instance on each call, so agents built from copies of these
    /// options never share moment buffers or a step counter.
    /// </para>
    /// </remarks>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }
}
