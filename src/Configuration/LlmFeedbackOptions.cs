namespace AiDotNet.Configuration;

/// <summary>Configures how a language model's opinion of a program is blended into that program's fitness.</summary>
/// <remarks>
/// <para>
/// A judge scores qualities an automated test cannot: whether an approach is readable, whether it generalises,
/// whether it is doing something clever or merely memorising the examples. Those scores are never allowed to become
/// the fitness on their own. <see cref="CombinedBlend"/> is the share of the final quality that stays with the
/// measured evaluator, so the default of 0.7 means a judge can move a candidate by at most three tenths of the
/// scale no matter how enthusiastic it is.
/// </para>
/// <para>
/// Three defaults differ deliberately from the reference implementation. Upstream hard-codes the 0.7/0.3 blend in
/// the evaluator body, applies its weights when averaging the individual criteria but not when blending, and runs
/// the judge even on candidates whose own evaluation failed, so a program that crashed can still be praised into a
/// respectable score. Here the blend is a configured number, <see cref="Weight"/> is applied consistently, and
/// <see cref="RunOnFailedEvaluations"/> defaults to <c>false</c>.
/// </para>
/// <para>
/// Nothing here contacts a model by itself. The options take effect only once a chat client is supplied to the
/// judge evaluator, and an answer that cannot be parsed leaves the measured quality untouched rather than failing
/// the candidate, so an unreliable judge degrades the search's extra signal rather than its correctness.
/// </para>
/// <para><b>For Beginners:</b> This controls a second opinion on each candidate program. Your tests say whether it
/// works; the model says whether it is any good as code. The two are mixed, with the tests carrying most of the
/// weight, so a program that fails its tests cannot be talked into a high score. List the qualities you care about
/// in <see cref="Criteria"/> — the defaults are correctness, efficiency, and readability — and leave the rest
/// alone until you have a reason to change it.</para>
/// </remarks>
public sealed class LlmFeedbackOptions
{
    /// <summary>The prefix given to every judge-produced metric name.</summary>
    public const string DefaultMetricPrefix = "llm_";

    /// <summary>The metric name holding the mean of the individual judge scores.</summary>
    public const string AverageMetricSuffix = "average";

    private IList<string>? _criteria;

    /// <summary>Gets or sets whether the judge runs at all.</summary>
    /// <remarks>When <c>false</c> the evaluator returns the measured result untouched and contacts no model.</remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>Gets or sets the qualities the judge is asked to score, one per prompt line.</summary>
    /// <remarks>Defaults to correctness, efficiency, and readability; an empty list is rejected by validation.</remarks>
    public IList<string> Criteria
    {
        get => _criteria ??= new List<string> { "correctness", "efficiency", "readability" };
        set => _criteria = value;
    }

    /// <summary>Gets or sets the multiplier applied to every judge score before they are averaged.</summary>
    /// <remarks>
    /// Use it to damp a judge you do not fully trust without changing the blend. Upstream applies its weights when
    /// averaging but drops them again when blending, so its configured weight silently has no effect on the final
    /// score; here the weighted average is what the blend consumes.
    /// </remarks>
    public double Weight { get; set; } = 1.0;

    /// <summary>Gets or sets the share of the final quality that stays with the measured evaluator.</summary>
    /// <remarks>The judge receives the remainder. Set it to 1 to record judge metrics without letting them move fitness.</remarks>
    public double CombinedBlend { get; set; } = 0.7;

    /// <summary>Gets or sets the prefix given to each judge metric recorded on the result.</summary>
    public string MetricPrefix { get; set; } = DefaultMetricPrefix;

    /// <summary>Gets or sets whether the judge also runs on candidates whose own evaluation did not complete.</summary>
    /// <remarks>
    /// Left <c>false</c>, a failed, rejected, or timed-out candidate is returned unchanged and costs no model call.
    /// </remarks>
    public bool RunOnFailedEvaluations { get; set; }

    /// <summary>Gets or sets whether judge scores are appended to the result's objectives as well as its descriptors.</summary>
    public bool RecordObjectives { get; set; } = true;

    /// <summary>Gets or sets the JSON shape the judge's answer must take, or <c>null</c> to derive one from the criteria.</summary>
    public string? ResponseSchema { get; set; }

    /// <summary>Gets or sets whether the request asks the provider to constrain the answer to JSON.</summary>
    /// <remarks>
    /// Providers that support constrained decoding then cannot return unparseable text at all, which is the
    /// difference between a judge that occasionally wastes a call and one that never does. Providers that ignore
    /// the request are unaffected, because the answer is still parsed leniently.
    /// </remarks>
    public bool RequestJsonResponseFormat { get; set; } = true;

    /// <summary>Gets or sets how many times an unparseable judge answer is requested again.</summary>
    public int MaxJudgeRetries { get; set; } = 1;

    /// <summary>Gets or sets the sampling temperature for judge requests, or <c>null</c> for the client's default.</summary>
    public double? Temperature { get; set; }

    /// <summary>Gets or sets the output token cap for judge requests, or <c>null</c> for the client's default.</summary>
    public int? MaxOutputTokens { get; set; }

    /// <summary>Creates an independent copy so a running evaluator is unaffected by later mutation.</summary>
    /// <returns>A new options instance carrying the same values and a copied criteria list.</returns>
    public LlmFeedbackOptions Clone() => new()
    {
        Enabled = Enabled,
        _criteria = _criteria is null ? null : new List<string>(_criteria),
        Weight = Weight,
        CombinedBlend = CombinedBlend,
        MetricPrefix = MetricPrefix,
        RunOnFailedEvaluations = RunOnFailedEvaluations,
        RecordObjectives = RecordObjectives,
        ResponseSchema = ResponseSchema,
        RequestJsonResponseFormat = RequestJsonResponseFormat,
        MaxJudgeRetries = MaxJudgeRetries,
        Temperature = Temperature,
        MaxOutputTokens = MaxOutputTokens
    };

    /// <summary>Validates the criteria, the blend, and the request bounds.</summary>
    /// <exception cref="ArgumentException">
    /// <see cref="Criteria"/> is empty or holds a blank entry, or <see cref="MetricPrefix"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <see cref="Weight"/> or <see cref="CombinedBlend"/> is outside its range, or a retry or token bound is invalid.
    /// </exception>
    public void Validate()
    {
        if (MetricPrefix is null) throw new ArgumentException("MetricPrefix cannot be null.", nameof(MetricPrefix));
        if (Criteria.Count == 0)
            throw new ArgumentException("At least one judging criterion is required.", nameof(Criteria));

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string criterion in Criteria)
        {
            if (criterion is not { } text || text.Trim().Length == 0)
                throw new ArgumentException("A judging criterion cannot be empty or white space.", nameof(Criteria));
            if (!seen.Add(text.Trim()))
                throw new ArgumentException("Judging criteria must be distinct.", nameof(Criteria));
        }

        if (double.IsNaN(Weight) || double.IsInfinity(Weight) || Weight < 0 || Weight > 1)
            throw new ArgumentOutOfRangeException(nameof(Weight), Weight, "Value must be between 0 and 1.");
        if (double.IsNaN(CombinedBlend) || double.IsInfinity(CombinedBlend) || CombinedBlend < 0 || CombinedBlend > 1)
            throw new ArgumentOutOfRangeException(nameof(CombinedBlend), CombinedBlend, "Value must be between 0 and 1.");
        if (MaxJudgeRetries < 0 || MaxJudgeRetries > 8)
            throw new ArgumentOutOfRangeException(nameof(MaxJudgeRetries), MaxJudgeRetries, "Value must be between 0 and 8.");
        if (MaxOutputTokens.HasValue && MaxOutputTokens.Value <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxOutputTokens), MaxOutputTokens.Value, "Value must be positive.");
        if (Temperature.HasValue
            && (double.IsNaN(Temperature.Value) || double.IsInfinity(Temperature.Value)
                || Temperature.Value < 0 || Temperature.Value > 2))
        {
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature.Value,
                "Value must be a finite number between 0 and 2.");
        }
    }
}
