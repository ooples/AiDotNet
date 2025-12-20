using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates how well a RAG system handles noisy or irrelevant documents in the retrieved context.
/// </summary>
/// <typeparam name="T">The numeric data type used for scoring (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This metric measures the robustness of a RAG system when the retrieved document set contains
/// irrelevant or noisy documents. A robust system should focus on high-quality documents and
/// maintain answer quality even when some retrieved documents are not relevant. The metric
/// analyzes the distribution of relevance scores to determine if the system can distinguish
/// signal from noise.
/// </para>
/// <para><b>For Beginners:</b> This checks if your RAG system can ignore "junk" documents.
/// 
/// Think of it like reading a research paper:
/// - You have 10 sources
/// - 5 are highly relevant to your topic
/// - 5 are somewhat related but not very useful
/// - A good researcher focuses on the 5 good sources
/// - A bad researcher treats all 10 equally
/// 
/// This metric checks if your RAG system acts like a good researcher!
/// 
/// Score interpretation:
/// - Score near 1.0: System focuses on relevant documents, ignores noise
/// - Score near 0.5: System treats all documents equally (not robust)
/// - Score near 0.0: System focuses on wrong documents
/// 
/// For example:
/// ```csharp
/// var metric = new NoiseRobustnessMetric<double>(NumOps.FromDouble(0.3));
/// var answer = new GroundedAnswer<double>(
///     query: "What is AI?",
///     answer: "AI is artificial intelligence...",
///     sourceDocuments: retrievedDocs,
///     citations: new List<string>(),
///     confidenceScore: 0.8
/// );
/// var score = metric.Evaluate(answer);
/// Console.WriteLine($"Robustness: {score} (higher means better at filtering noise)");
/// ```
/// 
/// Why this matters:
/// - Real-world retrieval often includes some irrelevant documents
/// - Robust systems maintain quality despite imperfect retrieval
/// - Helps evaluate production reliability
/// </para>
/// </remarks>
public class NoiseRobustnessMetric<T> : RAGMetricBase<T>
{
    private readonly T _noiseRatio;

    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    public override string Name => "Noise Robustness";

    /// <summary>
    /// Gets the description of what this metric measures.
    /// </summary>
    public override string Description => "Evaluates robustness to noisy or irrelevant documents in the context";

    /// <summary>
    /// Gets a value indicating whether this metric requires ground truth for evaluation.
    /// </summary>
    protected override bool RequiresGroundTruth => false;

    /// <summary>
    /// Initializes a new instance of the NoiseRobustnessMetric class.
    /// </summary>
    /// <param name="noiseRatio">The expected ratio of noise documents in the context (0-1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The noise ratio tells the metric what percentage of
    /// documents you expect to be irrelevant. For example, 0.3 means you expect about 30%
    /// of retrieved documents to be noise.
    /// </para>
    /// </remarks>
    public NoiseRobustnessMetric(T noiseRatio)
    {
        _noiseRatio = noiseRatio;
    }

    /// <summary>
    /// Evaluates robustness by measuring the quality distribution of source documents.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">Not used for this metric.</param>
    /// <returns>Robustness score between 0 and 1, where 1 indicates perfect noise filtering.</returns>
    /// <exception cref="ArgumentNullException">Thrown when answer is null.</exception>
    /// <remarks>
    /// <para>
    /// The metric analyzes the distribution of relevance scores across source documents.
    /// A robust system shows a clear separation between high-quality and low-quality documents,
    /// with top-ranked documents having significantly higher scores than bottom-ranked ones.
    /// The score is calculated as the ratio of average scores between top and bottom halves
    /// of the document set, normalized to 0-1 range.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if there's a big difference between
    /// the best and worst documents in your results.
    /// 
    /// What it does:
    /// 1. Sorts documents by relevance score (best to worst)
    /// 2. Splits them into top half and bottom half
    /// 3. Compares average scores of each half
    /// 4. Returns high score if top half is much better (good filtering)
    /// 5. Returns low score if both halves are similar (poor filtering)
    /// </para>
    /// </remarks>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (answer == null)
            throw new ArgumentNullException(nameof(answer));

        if (answer.SourceDocuments == null || answer.SourceDocuments.Count == 0)
            return NumOps.Zero;

        // Measure how well the answer focuses on relevant content despite noise
        // We check if documents with higher relevance scores contribute more to the answer
        var sortedDocs = answer.SourceDocuments
            .Where(d => d.HasRelevanceScore)
            .OrderByDescending(d => d.RelevanceScore)
            .ToList();

        if (sortedDocs.Count == 0)
            return NumOps.Zero;

        // Top documents should have significantly higher relevance scores
        var topHalf = sortedDocs.Take(sortedDocs.Count / 2).ToList();
        var bottomHalf = sortedDocs.Skip(sortedDocs.Count / 2).ToList();

        if (topHalf.Count == 0)
            return NumOps.Zero;

        var topAvg = topHalf.Average(d => Convert.ToDouble(d.RelevanceScore));
        var bottomAvg = bottomHalf.Count > 0 ? bottomHalf.Average(d => Convert.ToDouble(d.RelevanceScore)) : 0.0;

        // Robustness score: ratio of top to bottom scores (higher = more robust)
        var robustnessRatio = bottomAvg > 0 ? topAvg / bottomAvg : topAvg;

        // Normalize to 0-1 range using sigmoid-like function
        var normalizedScore = Math.Min(1.0, robustnessRatio / (1.0 + robustnessRatio));

        return NumOps.FromDouble(normalizedScore);
    }
}
