
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Provides a base implementation for RAG evaluation metrics with common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements the IRAGMetric interface and provides common validation
/// and utility methods for metric implementations. It ensures consistent behavior across
/// different metrics while allowing derived classes to focus on specific evaluation logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for all RAG metrics.
/// 
/// It handles common tasks like:
/// - Validating inputs (checking for null values)
/// - Normalizing scores (ensuring they're between 0 and 1)
/// - Providing helper methods for common calculations
/// 
/// Specific metrics (Faithfulness, Similarity, etc.) just need to implement
/// their specific scoring logic.
/// </para>
/// </remarks>
public abstract class RAGMetricBase<T> : IRAGMetric<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the description of what this metric measures.
    /// </summary>
    public abstract string Description { get; }

    /// <summary>
    /// Evaluates a grounded answer and returns a score.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">The expected/correct answer (null for reference-free metrics).</param>
    /// <returns>A score between 0 and 1, where 1 is perfect.</returns>
    public T Evaluate(GroundedAnswer<T> answer, string? groundTruth = null)
    {
        ValidateAnswer(answer);

        if (RequiresGroundTruth && string.IsNullOrWhiteSpace(groundTruth))
            throw new ArgumentException("This metric requires ground truth for evaluation", nameof(groundTruth));

        var score = EvaluateCore(answer, groundTruth);
        return MathHelper.Clamp(score, NumOps.Zero, NumOps.One);
    }

    /// <summary>
    /// Gets a value indicating whether this metric requires ground truth for evaluation.
    /// </summary>
    protected abstract bool RequiresGroundTruth { get; }

    /// <summary>
    /// Core evaluation logic to be implemented by derived classes.
    /// </summary>
    /// <param name="answer">The validated grounded answer.</param>
    /// <param name="groundTruth">The ground truth (if required).</param>
    /// <returns>A score (will be clamped to 0-1 range).</returns>
    protected abstract T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth);

    /// <summary>
    /// Validates the grounded answer.
    /// </summary>
    /// <param name="answer">The answer to validate.</param>
    protected virtual void ValidateAnswer(GroundedAnswer<T> answer)
    {
        if (answer == null)
            throw new ArgumentNullException(nameof(answer));

        if (string.IsNullOrWhiteSpace(answer.Answer))
            throw new ArgumentException("Answer text cannot be null or empty", nameof(answer));
    }

    /// <summary>
    /// Extracts words from text.
    /// </summary>
    /// <param name="text">The text to process.</param>
    /// <returns>Set of lowercase words.</returns>
    protected HashSet<string> GetWords(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new HashSet<string>();

        return new HashSet<string>(
            text.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?' },
                       StringSplitOptions.RemoveEmptyEntries));
    }
}
