using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates how well the retrieved documents cover the information needed to answer the query.
/// </summary>
/// <remarks>
/// <para>
/// Context coverage measures whether the retrieved documents contain the information needed
/// to answer the query. When ground truth is provided, it checks if the sources contain
/// the key information from the correct answer. Without ground truth, it estimates coverage
/// by analyzing document relevance scores and diversity.
/// </para>
/// <para><b>For Beginners:</b> This checks if the retrieved documents have enough info to answer the question.
/// 
/// Think of it like checking if you brought the right textbooks to class:
/// - Question: "What is photosynthesis?"
/// - Retrieved docs about plants, sunlight, energy ✓ (good coverage)
/// - Retrieved docs about animals, water, soil ✗ (poor coverage)
/// 
/// How it works:
/// 
/// **With Ground Truth** (you know the correct answer):
/// - Checks if the retrieved documents contain words from the correct answer
/// - High score: Sources have all the key information
/// - Low score: Sources are missing important facts
/// 
/// **Without Ground Truth** (no reference answer):
/// - Uses relevance scores from retrieval
/// - Checks document diversity (not all the same topic)
/// - Estimates if sources are comprehensive enough
/// 
/// For example:
/// - Query: "What are the products of photosynthesis?"
/// - Ground truth: "Glucose and oxygen"
/// - Retrieved docs mention "glucose" and "oxygen" ✓ (score: 1.0)
/// - Retrieved docs only mention "glucose" ✗ (score: 0.5)
/// 
/// Why this matters:
/// - Bad retrieval = bad answers (garbage in, garbage out)
/// - Identifies when you need better retrieval or more documents
/// - Helps tune retrieval parameters (topK, similarity threshold)
/// </para>
/// </remarks>
public class ContextCoverageMetric<T> : RAGMetricBase<T>
{
    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    public override string Name => "Context Coverage";

    /// <summary>
    /// Gets the description of what this metric measures.
    /// </summary>
    public override string Description =>
        "Measures how well the retrieved documents cover the information needed to answer the query";

    /// <summary>
    /// Gets a value indicating whether this metric requires ground truth.
    /// </summary>
    /// <remarks>
    /// This metric can work both with and without ground truth, providing different
    /// types of coverage evaluation.
    /// </remarks>
    protected override bool RequiresGroundTruth => false;

    /// <summary>
    /// Evaluates context coverage.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">The reference answer (optional).</param>
    /// <returns>Coverage score (0-1).</returns>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (!answer.SourceDocuments.Any())
            return NumOps.Zero;

        if (!string.IsNullOrWhiteSpace(groundTruth))
        {
            return EvaluateWithGroundTruth(answer, groundTruth!);
        }

        return EvaluateWithoutGroundTruth(answer);
    }

    /// <summary>
    /// Evaluates coverage when ground truth is available.
    /// </summary>
    private T EvaluateWithGroundTruth(GroundedAnswer<T> answer, string groundTruth)
    {
        var groundTruthWords = GetWords(groundTruth);
        if (groundTruthWords.Count == 0)
            return NumOps.Zero;

        var sourceText = string.Join(" ", answer.SourceDocuments.Select(d => d.Content));
        var sourceWords = GetWords(sourceText);

        var coveredWords = groundTruthWords.Intersect(sourceWords).Count();

        return NumOps.Divide(NumOps.FromDouble(coveredWords), NumOps.FromDouble(groundTruthWords.Count));
    }

    /// <summary>
    /// Evaluates coverage without ground truth using heuristics.
    /// </summary>
    private T EvaluateWithoutGroundTruth(GroundedAnswer<T> answer)
    {
        var docs = answer.SourceDocuments.ToList();

        var sumRelevance = NumOps.Zero;
        var countWithScores = 0;

        foreach (var doc in docs)
        {
            if (doc.HasRelevanceScore)
            {
                sumRelevance = NumOps.Add(sumRelevance, doc.RelevanceScore);
                countWithScores++;
            }
        }

        var avgRelevance = countWithScores > 0
            ? NumOps.Divide(sumRelevance, NumOps.FromDouble(countWithScores))
            : NumOps.FromDouble(0.5);

        var allWords = new HashSet<string>();
        var totalWords = 0;

        foreach (var doc in docs)
        {
            var docWords = GetWords(doc.Content);
            allWords.UnionWith(docWords);
            totalWords += docWords.Count;
        }

        var diversityScore = totalWords == 0
            ? NumOps.Zero
            : NumOps.Divide(NumOps.FromDouble(allWords.Count), NumOps.FromDouble(totalWords));

        var weightedRelevance = NumOps.Multiply(avgRelevance, NumOps.FromDouble(0.7));
        var weightedDiversity = NumOps.Multiply(diversityScore, NumOps.FromDouble(0.3));

        return NumOps.Add(weightedRelevance, weightedDiversity);
    }
}
