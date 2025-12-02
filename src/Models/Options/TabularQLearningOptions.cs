using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Tabular Q-Learning agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Tabular Q-Learning maintains a lookup table of Q-values for discrete
/// state-action pairs. No neural networks or function approximation.
/// </para>
/// <para><b>For Beginners:</b>
/// This is the simplest form of Q-Learning where we literally maintain a table.
/// Each row is a state, each column is an action, and the cells contain Q-values.
///
/// Best for:
/// - Small discrete state spaces (e.g., 10x10 grid world)
/// - Discrete action spaces
/// - Learning exact optimal policies
/// - Understanding RL fundamentals
///
/// Not suitable for:
/// - Continuous states (infinitely many states)
/// - Large state spaces (millions of states)
/// - High-dimensional observations (images, etc.)
/// </para>
/// </remarks>
public class TabularQLearningOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Size of the state space (number of state features).
    /// </summary>
    public int StateSize { get; init; }

    /// <summary>
    /// Size of the action space (number of possible actions).
    /// </summary>
    public int ActionSize { get; init; }
}
