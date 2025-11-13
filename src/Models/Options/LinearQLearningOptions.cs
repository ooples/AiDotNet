using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Linear Q-Learning agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Linear Q-Learning uses linear function approximation to estimate Q-values.
/// Instead of maintaining a table, it learns weight vectors for each action
/// and computes Q(s,a) = w_a^T * φ(s) where φ(s) are state features.
/// </para>
/// <para><b>For Beginners:</b>
/// Linear Q-Learning extends tabular Q-learning to handle larger state spaces
/// by using feature representations. Think of it as learning a formula instead
/// of memorizing every single state.
///
/// Best for:
/// - Medium-sized continuous state spaces
/// - Problems where states can be represented as feature vectors
/// - Faster learning than tabular methods
/// - Generalization across similar states
///
/// Not suitable for:
/// - Very small discrete states (use tabular instead)
/// - Highly non-linear relationships (use neural networks)
/// - Continuous action spaces (use actor-critic)
/// </para>
/// </remarks>
public class LinearQLearningOptions<T> : ReinforcementLearningOptions<T>
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
