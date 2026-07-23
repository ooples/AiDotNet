using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates context recall (RAGAS): what fraction of the ground-truth answer is supported by the
/// retrieved contexts.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Context recall measures retrieval completeness. The ground-truth answer is broken into individual
/// claims (sentences); each claim is checked for whether it can be attributed to the retrieved
/// contexts. The score is attributable-claims / total-claims. A low score means the retriever missed
/// information that the correct answer depends on.
/// </para>
/// <para>
/// Attribution of each claim is decided by the text generator (a yes/no judge) when one is supplied,
/// otherwise by cosine similarity against an embedding model, otherwise by an offline lexical
/// word-coverage threshold.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class ContextRecallMetric<T> : RAGMetricBase<T>
{
    private readonly ITextGenerator? _generator;
    private readonly IEmbeddingModel<T>? _embeddingModel;
    private readonly double _attributionThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="ContextRecallMetric{T}"/> class.
    /// </summary>
    /// <param name="generator">Optional text generator used to judge whether each claim is attributable to the context.</param>
    /// <param name="embeddingModel">Optional embedding model used to score claim attribution by cosine similarity.</param>
    /// <param name="attributionThreshold">
    /// The similarity/coverage threshold (0-1) at or above which a claim counts as attributable for the
    /// embedding and lexical paths (default 0.5). Ignored when a generator is supplied.
    /// </param>
    public ContextRecallMetric(
        ITextGenerator? generator = null,
        IEmbeddingModel<T>? embeddingModel = null,
        double attributionThreshold = 0.5)
    {
        _generator = generator;
        _embeddingModel = embeddingModel;
        _attributionThreshold = attributionThreshold;
    }

    /// <summary>Gets the name of this metric.</summary>
    public override string Name => "Context Recall";

    /// <summary>Gets the description of what this metric measures.</summary>
    public override string Description =>
        "Measures what fraction of the ground-truth answer is supported by the retrieved contexts (RAGAS context recall)";

    /// <summary>Gets a value indicating whether this metric requires ground truth.</summary>
    protected override bool RequiresGroundTruth => true;

    /// <summary>
    /// Evaluates context recall of the ground-truth claims against the retrieved documents.
    /// </summary>
    /// <param name="answer">The grounded answer whose source documents are the retrieved context.</param>
    /// <param name="groundTruth">The reference answer that is broken into claims.</param>
    /// <returns>Context recall score (0-1).</returns>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        var docs = answer.SourceDocuments.ToList();
        if (docs.Count == 0)
            return NumOps.Zero;

        var claims = SplitIntoSentences(groundTruth!);
        if (claims.Count == 0)
            return NumOps.Zero;

        var context = string.Join(" ", docs.Select(d => d.Content));

        int attributable = 0;
        foreach (var claim in claims)
        {
            if (IsAttributable(claim, context, docs))
                attributable++;
        }

        return NumOps.FromDouble((double)attributable / claims.Count);
    }

    /// <summary>
    /// Decides whether a ground-truth claim can be attributed to the retrieved context using the best
    /// available signal: LLM judge, then embedding cosine (best matching document), then lexical
    /// word coverage.
    /// </summary>
    private bool IsAttributable(string claim, string context, List<Document<T>> docs)
    {
        if (_generator != null)
        {
            var prompt =
                "Can the statement be attributed to (supported by) the context? " +
                "Reply with only 'yes' or 'no'.\n\n" +
                $"Context: {context}\n\nStatement: {claim}\n\nAttributable (yes/no):";
            return ParseAffirmative(_generator.Generate(prompt));
        }

        if (_embeddingModel != null)
        {
            double best = 0.0;
            foreach (var doc in docs)
            {
                double sim = EmbeddingCosine(_embeddingModel, claim, doc.Content);
                if (sim > best)
                    best = sim;
            }

            return best >= _attributionThreshold;
        }

        // Offline lexical fallback: fraction of the claim's words present in the combined context.
        var claimWords = GetWords(claim);
        if (claimWords.Count == 0)
            return false;

        var contextWords = GetWords(context);
        double coverage = (double)claimWords.Intersect(contextWords).Count() / claimWords.Count;
        return coverage >= _attributionThreshold;
    }
}
