
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

    /// <summary>
    /// Computes the Jaccard (word-overlap) similarity between two texts. Used as the offline
    /// lexical fallback for metrics when no embedding model or text generator is supplied.
    /// </summary>
    /// <param name="a">The first text.</param>
    /// <param name="b">The second text.</param>
    /// <returns>A value in [0, 1]; 0 when both texts have no words.</returns>
    protected double JaccardSimilarity(string a, string b)
    {
        var wa = GetWords(a);
        var wb = GetWords(b);
        var union = wa.Union(wb).Count();
        if (union == 0)
            return 0.0;

        var intersection = wa.Intersect(wb).Count();
        return (double)intersection / union;
    }

    /// <summary>
    /// Splits text into individual sentences/claims on sentence-terminating punctuation and
    /// line breaks. Whitespace-only fragments are dropped.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>The list of non-empty, trimmed sentences.</returns>
    protected static List<string> SplitIntoSentences(string text)
    {
        var result = new List<string>();
        if (string.IsNullOrWhiteSpace(text))
            return result;

        var parts = text.Split(new[] { '.', '!', '?', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var part in parts)
        {
            var trimmed = part.Trim();
            if (trimmed.Length > 0)
                result.Add(trimmed);
        }

        return result;
    }

    /// <summary>
    /// Parses a multi-line LLM reply into individual lines, stripping common list markers
    /// (leading numbering like "1." / "1)" and bullets like "-", "*", "•").
    /// </summary>
    /// <param name="reply">The raw generator reply.</param>
    /// <returns>The cleaned, non-empty lines.</returns>
    protected static List<string> ParseListLines(string? reply)
    {
        var result = new List<string>();
        if (string.IsNullOrWhiteSpace(reply))
            return result;

        var lines = reply!.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var line in lines)
        {
            var cleaned = System.Text.RegularExpressions.Regex.Replace(
                line.Trim(), @"^\s*(\d+[\.\)]|[-*•])\s*", string.Empty).Trim();
            if (cleaned.Length > 0)
                result.Add(cleaned);
        }

        return result;
    }

    /// <summary>
    /// Interprets a yes/no style LLM verdict. Returns <c>true</c> when the reply affirms
    /// (starts with or clearly contains "yes"/"true"/"supported"), <c>false</c> otherwise.
    /// </summary>
    /// <param name="reply">The raw generator reply.</param>
    /// <returns><c>true</c> for an affirmative verdict; otherwise <c>false</c>.</returns>
    protected static bool ParseAffirmative(string? reply)
    {
        if (string.IsNullOrWhiteSpace(reply))
            return false;

        var s = reply!.Trim().ToLowerInvariant();
        if (s.StartsWith("yes") || s.StartsWith("true") || s.StartsWith("support"))
            return true;
        if (s.StartsWith("no") || s.StartsWith("false") || s.StartsWith("not"))
            return false;

        return s.Contains("yes") && !s.Contains("no");
    }

    /// <summary>
    /// Computes cosine similarity between the embeddings of two texts using the supplied
    /// embedding model. Reuses the shared <see cref="StatisticsHelper{T}.CosineSimilarity"/>.
    /// </summary>
    /// <param name="model">The embedding model.</param>
    /// <param name="a">The first text.</param>
    /// <param name="b">The second text.</param>
    /// <returns>The cosine similarity as a double in [-1, 1].</returns>
    protected static double EmbeddingCosine(IEmbeddingModel<T> model, string a, string b)
    {
        var va = model.Embed(a ?? string.Empty);
        var vb = model.Embed(b ?? string.Empty);
        return Convert.ToDouble(StatisticsHelper<T>.CosineSimilarity(va, vb));
    }
}
