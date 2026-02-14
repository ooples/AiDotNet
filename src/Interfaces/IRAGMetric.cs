using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for RAG evaluation metrics.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// A RAG metric evaluates the quality of retrieval-augmented generation systems
/// by comparing generated answers against ground truth or analyzing specific aspects
/// of the generation process. Metrics help developers understand system performance
/// and guide improvements.
/// </para>
/// <para><b>For Beginners:</b> Metrics are like test scores for your RAG system.
/// 
/// Think of it like grading an exam:
/// - The metric looks at the AI's answer
/// - Compares it to what the answer should be (or checks quality)
/// - Gives a score (0-1, where 1 is perfect)
/// 
/// Different metrics measure different things:
/// - Faithfulness: Does the answer stick to the source documents?
/// - Similarity: How close is the answer to the ground truth?
/// - Coverage: Does the answer address all parts of the question?
/// 
/// Use metrics to:
/// - Compare different RAG configurations
/// - Track improvements over time
/// - Identify weak points in your system
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("RAGMetric")]
public interface IRAGMetric<T>
{
    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the description of what this metric measures.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Evaluates a grounded answer and returns a score.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">The expected/correct answer (null for reference-free metrics).</param>
    /// <returns>A score between 0 and 1, where 1 is perfect.</returns>
    T Evaluate(GroundedAnswer<T> answer, string? groundTruth = null);
}
