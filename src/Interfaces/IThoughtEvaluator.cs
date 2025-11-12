using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for evaluating the quality and promise of thoughts or reasoning steps.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A thought evaluator judges how good or promising a thought is.
/// It's like a teacher grading your work or a chess player evaluating a potential move.
///
/// The evaluator considers:
/// - Is this thought relevant to solving the problem?
/// - Is it logically sound?
/// - How likely is it to lead to a good solution?
/// - Is it supported by evidence?
///
/// Returns a score (usually 0.0 to 1.0) indicating quality.
/// </para>
/// </remarks>
public interface IThoughtEvaluator<T>
{
    /// <summary>
    /// Evaluates the quality of a thought node.
    /// </summary>
    /// <param name="node">The thought node to evaluate.</param>
    /// <param name="originalQuery">The original problem being solved.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Evaluation score (typically 0.0 to 1.0).</returns>
    Task<T> EvaluateThoughtAsync(
        ThoughtNode<T> node,
        string originalQuery,
        ReasoningConfig config,
        CancellationToken cancellationToken = default);
}
