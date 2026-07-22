using System.Globalization;
using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates the factual correctness of generated answers.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Assesses whether the generated answer contains factually correct information by comparing
/// against ground truth. When a text generator is supplied it is used as an LLM judge that rates
/// correctness; when an embedding model is supplied semantic (cosine) similarity is used. When both
/// are supplied the two signals are averaged (RAGAS-style). When neither is supplied the metric
/// falls back to the offline lexical Jaccard word-overlap heuristic.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class AnswerCorrectnessMetric<T> : RAGMetricBase<T>
{
    private readonly string _llmEndpoint;
    private readonly string _llmApiKey;
    private readonly ITextGenerator? _generator;
    private readonly IEmbeddingModel<T>? _embeddingModel;

    /// <summary>
    /// Gets the name of this metric.
    /// </summary>
    public override string Name => "Answer Correctness";

    /// <summary>
    /// Gets the description of what this metric measures.
    /// </summary>
    public override string Description => "Evaluates the factual correctness of generated answers by comparing against ground truth.";

    /// <summary>
    /// Gets a value indicating whether this metric requires ground truth.
    /// </summary>
    protected override bool RequiresGroundTruth => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="AnswerCorrectnessMetric{T}"/> class backed by a
    /// real LLM judge and/or embedding model.
    /// </summary>
    /// <param name="generator">
    /// Optional text generator used to rate factual correctness of the answer against the ground truth.
    /// </param>
    /// <param name="embeddingModel">
    /// Optional embedding model used to compute semantic similarity between the answer and ground truth.
    /// </param>
    /// <remarks>When both arguments are <c>null</c>, the metric uses the offline lexical fallback.</remarks>
    public AnswerCorrectnessMetric(ITextGenerator? generator = null, IEmbeddingModel<T>? embeddingModel = null)
    {
        _llmEndpoint = string.Empty;
        _llmApiKey = string.Empty;
        _generator = generator;
        _embeddingModel = embeddingModel;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="AnswerCorrectnessMetric{T}"/> class.
    /// </summary>
    /// <param name="llmEndpoint">The LLM API endpoint for fact checking (informational).</param>
    /// <param name="llmApiKey">The API key for the LLM service (informational).</param>
    /// <remarks>
    /// Back-compatible constructor. Because it is not wired to a real generator, this overload uses
    /// the offline lexical Jaccard fallback. Prefer the constructor that accepts a text generator
    /// and/or embedding model for semantic evaluation.
    /// </remarks>
    public AnswerCorrectnessMetric(string llmEndpoint, string llmApiKey)
    {
        Guard.NotNull(llmEndpoint);
        _llmEndpoint = llmEndpoint;
        Guard.NotNull(llmApiKey);
        _llmApiKey = llmApiKey;
        _generator = null;
        _embeddingModel = null;
    }

    /// <summary>
    /// Evaluates answer correctness against the ground truth.
    /// </summary>
    /// <param name="answer">The grounded answer to evaluate.</param>
    /// <param name="groundTruth">The reference/correct answer.</param>
    /// <returns>Correctness score (0-1).</returns>
    protected override T EvaluateCore(GroundedAnswer<T> answer, string? groundTruth)
    {
        if (string.IsNullOrWhiteSpace(answer.Answer) || string.IsNullOrWhiteSpace(groundTruth))
            return NumOps.Zero;

        double? judgeScore = _generator != null ? RateWithJudge(answer.Answer, groundTruth!) : (double?)null;
        double? semanticScore = _embeddingModel != null
            ? Math.Max(0.0, EmbeddingCosine(_embeddingModel, answer.Answer, groundTruth!))
            : (double?)null;

        if (judgeScore.HasValue && semanticScore.HasValue)
            return NumOps.FromDouble((judgeScore.Value + semanticScore.Value) / 2.0);
        if (judgeScore.HasValue)
            return NumOps.FromDouble(judgeScore.Value);
        if (semanticScore.HasValue)
            return NumOps.FromDouble(semanticScore.Value);

        // Offline lexical fallback: Jaccard word overlap.
        return NumOps.FromDouble(JaccardSimilarity(answer.Answer, groundTruth!));
    }

    /// <summary>
    /// Asks the generator to rate how factually correct the answer is against the ground truth on a
    /// 0-10 scale, and normalizes the parsed number to [0, 1].
    /// </summary>
    private double RateWithJudge(string answerText, string groundTruth)
    {
        var prompt =
            "On a scale of 0 to 10, how factually correct and complete is the answer compared to " +
            "the reference answer? Reply with only a single number.\n\n" +
            $"Reference: {groundTruth}\n\nAnswer: {answerText}\n\nCorrectness (0-10):";
        var reply = _generator!.Generate(prompt);
        var match = Regex.Match(reply ?? string.Empty, @"\d+(\.\d+)?");
        if (match.Success && double.TryParse(match.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double raw))
            return Math.Min(1.0, Math.Max(0.0, raw / 10.0));

        // Unparseable reply: fall back to lexical overlap so the score is still meaningful.
        return JaccardSimilarity(answerText, groundTruth);
    }
}
