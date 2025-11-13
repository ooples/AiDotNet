using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for SARSA agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm.
/// Unlike Q-Learning, it updates based on the action actually taken.
/// </para>
/// <para><b>For Beginners:</b>
/// SARSA is more conservative than Q-Learning because it learns from actions
/// it actually takes (including exploratory ones). This makes it safer in
/// environments where bad actions can be catastrophic.
///
/// Classic example: **Cliff Walking**
/// - Q-Learning learns the shortest path (risky, close to cliff)
/// - SARSA learns a safer path (further from cliff)
///
/// Use SARSA when:
/// - Safety matters during training
/// - You want to learn a safe policy
/// - Environment has dangerous states
///
/// Use Q-Learning when:
/// - You want the optimal policy
/// - Safety during training doesn't matter
/// - You can afford exploratory mistakes
/// </para>
/// </remarks>
public class SARSAOptions<T> : ReinforcementLearningOptions<T>
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
    /// Initial exploration rate (epsilon for epsilon-greedy).
    /// </summary>
    public double EpsilonStart { get; init; } = 1.0;

    /// <summary>
    /// Final exploration rate after decay.
    /// </summary>
    public double EpsilonEnd { get; init; } = 0.01;

    /// <summary>
    /// Epsilon decay rate per episode.
    /// </summary>
    public double EpsilonDecay { get; init; } = 0.995;
}
