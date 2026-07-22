using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates context precision (RAGAS): whether the relevant retrieved contexts are ranked highly.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Context precision rewards a retriever for placing the useful contexts near the top of the ranked
/// list. Each retrieved context is first judged relevant or not (against the ground-truth answer when
/// present, otherwise the query). The score is the average of Precision@k evaluated at each rank that
/// holds a relevant context — the standard average-precision formulation used by RAGAS:
/// <c>sum_k (Precision@k * relevant_k) / total_relevant</c>.
/// </para>
/// <para>
/// Relevance of each context is decided by the text generator (a yes/no judge) when one is supplied,
/// otherwise by cosine similarity against an embedding model, otherwise by an offline lexical
/// Jaccard-overlap threshold.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class ContextPrecisionMetric<T> : RAGMetricBase<T>
{
    private readonly ITextGenerator? _generator;
    private readonly IEmbeddingModel<T>? _embeddingModel;
    private readonly double _relevanceThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="ContextPrecisionMetric{T}"/> class.
    /// </summary>
    /// <param name="generator">Optional text generator used to judge each context relevant / not relevant.</param>
    /// <param name="embeddingModel">Optional embedding model used to score context relevance by cosine similarity.</param>
    /// <param name="relevanceThreshold">
    /// The similarity threshold (0-1) at or above which a context counts as relevant for the
    /// embedding and lexical paths (default 0.5). Ignored when a generator is supplied.
    /// </param>
    public ContextPrecisionMetric(
        ITextGenerator? generator = null,
        IEmbeddingModel<T>? embeddingModel = null,
        double relevanceThreshold = 0.5)
    {
        _generator = generator;
        _embeddingModel = embeddingModel;
        _relevanceThreshold = relevanceThreshold;
    }

    /// <summary>Gets the name of this metric.</summary>
    public override string Name => "Context Precision";

    /// <summary>Gets the description of what this metric measures.</summary>
    public override string Description =>
        "Measures whether relevant retrieved contexts are ranked ahead of irrelevant ones (RAGAS context precision)";

    /// <summary>Gets a value indicating whether this metric requires ground truth.</summary>
    protected override bool RequiresGroundTruth => false;

    /// <summary>
    /// Evaluates context precision over the ranked retrieved documents.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">The reference answer; when absent the query is used as the relevance target.</param>
    /// <returns>Context precision score (0-1).</returns>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        var docs = answer.SourceDocuments.ToList();
        if (docs.Count == 0)
            return NumOps.Zero;

        var reference = !string.IsNullOrWhiteSpace(groundTruth) ? groundTruth! : answer.Query;
        if (string.IsNullOrWhiteSpace(reference))
            return NumOps.Zero;

        double weightedPrecisionSum = 0.0;
        int relevantSoFar = 0;
        int totalRelevant = 0;

        for (int i = 0; i < docs.Count; i++)
        {
            bool relevant = IsRelevant(reference, docs[i].Content);
            if (relevant)
            {
                relevantSoFar++;
                totalRelevant++;
                double precisionAtK = (double)relevantSoFar / (i + 1);
                weightedPrecisionSum += precisionAtK;
            }
        }

        if (totalRelevant == 0)
            return NumOps.Zero;

        return NumOps.FromDouble(weightedPrecisionSum / totalRelevant);
    }

    /// <summary>
    /// Decides whether a single context is relevant to the reference using the best available signal:
    /// LLM judge, then embedding cosine, then lexical overlap.
    /// </summary>
    private bool IsRelevant(string reference, string context)
    {
        if (_generator != null)
        {
            var prompt =
                "Is the following context useful for answering or verifying the reference? " +
                "Reply with only 'yes' or 'no'.\n\n" +
                $"Reference: {reference}\n\nContext: {context}\n\nUseful (yes/no):";
            return ParseAffirmative(_generator.Generate(prompt));
        }

        if (_embeddingModel != null)
            return EmbeddingCosine(_embeddingModel, reference, context) >= _relevanceThreshold;

        return JaccardSimilarity(reference, context) >= _relevanceThreshold;
    }
}
