using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for LSTD (Least-Squares Temporal Difference) agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LSTD solves for the optimal linear weights directly using matrix operations
/// (A^-1 * b) rather than incremental updates. This provides more sample-efficient
/// learning but requires solving a linear system.
/// </para>
/// <para><b>For Beginners:</b>
/// LSTD is like solving a math equation directly instead of guessing and checking.
/// It collects experiences and then computes the best weights all at once using
/// linear algebra, rather than slowly adjusting them one step at a time.
///
/// Best for:
/// - Limited data scenarios (sample efficient)
/// - Batch learning from fixed datasets
/// - When you have computational power for matrix operations
/// - Problems where convergence speed matters
///
/// Not suitable for:
/// - Very large feature spaces (matrix becomes huge)
/// - Online learning (needs batches)
/// - When computational resources are limited
/// - Non-linear function approximation needs
/// </para>
/// </remarks>
public class LSTDOptions<T> : ReinforcementLearningOptions<T>
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
}
