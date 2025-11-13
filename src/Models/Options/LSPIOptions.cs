using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for LSPI (Least-Squares Policy Iteration) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LSPI combines least-squares methods with policy iteration. It alternates between
/// policy evaluation (using LSTDQ) and policy improvement, iteratively refining
/// the policy until convergence.
/// </para>
/// <para><b>For Beginners:</b>
/// LSPI is like repeatedly asking "what's the best policy?" and "how good is it?"
/// until the answers stop changing. Each iteration uses LSTD to evaluate the current
/// policy, then improves it based on those evaluations.
///
/// Best for:
/// - Batch reinforcement learning
/// - Offline learning from fixed datasets
/// - Sample-efficient policy learning
/// - When you need guaranteed convergence
///
/// Not suitable for:
/// - Online/streaming scenarios
/// - Very large feature spaces
/// - Continuous action spaces
/// - Real-time learning requirements
/// </para>
/// </remarks>
public class LSPIOptions<T> : ReinforcementLearningOptions<T>
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
    /// Regularization parameter to prevent overfitting and ensure numerical stability.
    /// </summary>
    public double RegularizationParam { get; init; } = 0.01;

    /// <summary>
    /// Maximum number of policy iteration steps before stopping.
    /// </summary>
    public int MaxIterations { get; init; } = 20;

    /// <summary>
    /// Weight change threshold for determining convergence.
    /// </summary>
    public double ConvergenceThreshold { get; init; } = 0.01;
}
