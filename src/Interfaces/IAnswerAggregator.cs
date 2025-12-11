using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for aggregating multiple answers into a final consensus answer.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When you solve a problem multiple different ways, you get multiple answers.
/// An answer aggregator combines these answers to determine the final "best" answer.
///
/// Common approaches:
/// - **Majority Voting**: The most common answer wins (like a democratic vote)
/// - **Weighted Average**: Answers with higher confidence scores count more
/// - **Consensus Building**: Combine similar answers and reject outliers
///
/// For example, if 10 reasoning attempts give answers [36, 36, 35, 36, 36, 36, 37, 36, 36, 36],
/// majority voting would pick "36" as the final answer.
/// </para>
/// </remarks>
internal interface IAnswerAggregator<T>
{
    /// <summary>
    /// Aggregates multiple candidate answers into a single final answer.
    /// </summary>
    /// <param name="answers">List of candidate answers.</param>
    /// <param name="confidenceScores">Confidence scores for each answer (as a Vector).</param>
    /// <returns>The final aggregated answer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Takes a list of answers and their confidence scores,
    /// then determines which answer to use as the final result.
    /// </para>
    /// </remarks>
    string Aggregate(List<string> answers, Vector<T> confidenceScores);

    /// <summary>
    /// Gets the name of this aggregation method.
    /// </summary>
    string MethodName { get; }
}
