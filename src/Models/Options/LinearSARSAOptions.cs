using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Linear SARSA agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Linear SARSA uses linear function approximation for on-policy learning.
/// Unlike Linear Q-Learning (off-policy), SARSA updates based on the action
/// actually taken by the current policy, making it more conservative.
/// </para>
/// <para><b>For Beginners:</b>
/// Linear SARSA is the on-policy version of Linear Q-Learning. It learns about
/// the policy it's currently following, rather than the optimal policy. This makes
/// it safer in risky environments where exploration could be dangerous.
///
/// Best for:
/// - Medium-sized continuous state spaces
/// - Risky environments (cliff walking, robotics)
/// - More conservative, safe learning
/// - Feature-based state representations
///
/// Not suitable for:
/// - Very small discrete states (use tabular SARSA)
/// - When fastest convergence is needed (use Q-learning)
/// - Highly non-linear problems (use neural networks)
/// </para>
/// </remarks>
public class LinearSARSAOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Number of features in the state representation.
    /// </summary>
    public int FeatureSize { get; init; }

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
