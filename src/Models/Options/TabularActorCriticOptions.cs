using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Tabular Actor-Critic agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Tabular Actor-Critic combines policy learning (actor) with value function learning (critic)
/// using lookup tables. The actor learns which actions to take, while the critic evaluates
/// how good those actions are.
/// </para>
/// <para><b>For Beginners:</b>
/// Actor-Critic is like having both a player (actor) and a coach (critic). The player tries
/// different strategies, and the coach provides feedback on how well they're working.
///
/// Best for:
/// - Small discrete state/action spaces
/// - Problems requiring both policy and value learning
/// - More stable learning than pure policy gradient
/// - Reducing variance in policy updates
///
/// Not suitable for:
/// - Continuous states (use linear/neural versions)
/// - Large state spaces (table becomes too big)
/// - High-dimensional observations
/// </para>
/// </remarks>
public class TabularActorCriticOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Size of the state space (number of state features).
    /// </summary>
    public int StateSize { get; init; }

    /// <summary>
    /// Size of the action space (number of possible actions).
    /// </summary>
    public int ActionSize { get; init; }

    /// <summary>
    /// Learning rate for the actor (policy) updates.
    /// </summary>
    public double ActorLearningRate { get; init; } = 0.01;

    /// <summary>
    /// Learning rate for the critic (value function) updates.
    /// </summary>
    public double CriticLearningRate { get; init; } = 0.1;
}
