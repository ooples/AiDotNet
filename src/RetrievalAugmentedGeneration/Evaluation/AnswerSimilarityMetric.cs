using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates the similarity between the generated answer and ground truth.
/// </summary>
/// <remarks>
/// <para>
/// This metric measures how similar the generated answer is to a known correct answer (ground truth).
/// It uses Jaccard similarity (word overlap) to compare the two texts. This is useful for benchmarking
/// and regression testing when you have reference answers.
/// </para>
/// <para><b>For Beginners:</b> This checks how close the AI's answer is to the correct answer.
/// 
/// Think of it like grading an exam:
/// - You have the answer key (ground truth)
/// - The student's answer (generated answer)
/// - This metric gives partial credit based on how much overlaps
/// 
/// For example:
/// - Ground truth: "Photosynthesis converts sunlight into energy"
/// - Generated: "Photosynthesis converts sunlight into chemical energy"
/// - Score: ~0.85 (most words match)
/// 
/// Scoring:
/// - 1.0: Perfect match
/// - 0.5-0.8: Partially correct
/// - 0.0-0.3: Mostly incorrect
/// 
/// Use cases:
/// - Benchmarking your RAG system against test datasets
/// - A/B testing different configurations
/// - Regression testing (ensure changes don't hurt quality)
/// 
/// Note: This uses simple word overlap. Production systems should use semantic
/// similarity with embeddings or BERTScore for better accuracy.
/// </para>
/// </remarks>
public class AnswerSimilarityMetric : RAGMetricBase
{
    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    public override string Name => "Answer Similarity";

    /// <summary>
    /// Gets the description of what this metric measures.
    /// </summary>
    public override string Description =>
        "Measures how similar the generated answer is to the ground truth answer";

    /// <summary>
    /// Gets a value indicating whether this metric requires ground truth.
    /// </summary>
    protected override bool RequiresGroundTruth => true;

    /// <summary>
    /// Evaluates similarity using Jaccard similarity.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">The reference/correct answer.</param>
    /// <returns>Similarity score (0-1).</returns>
    protected override double EvaluateCore(GroundedAnswer answer, string? groundTruth)
    {
        // Ground truth is guaranteed to be non-null by base class validation
        return JaccardSimilarity(answer.Answer, groundTruth!);
    }
}
