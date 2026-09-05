using System.Globalization;
using AiDotNet.Evolution;

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

    /// <summary>The JSON field the judge's written criticism is read from unless another is configured.</summary>
    /// <remarks>
    /// This is the field the derived response schema has always asked for, so a judge following it already returns
    /// the text; before it was carried forward the answer was parsed for the scores and the prose was discarded.
    /// </remarks>
    public const string DefaultCritiqueField = "reasoning";

    /// <summary>The artifact key the carried-forward critique is attached under.</summary>
    public const string CritiqueArtifactKey = "llm_judge_critique";

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

    /// <summary>Gets or sets whether every ensemble member judges and their scores are averaged by weight.</summary>
    /// <remarks>
    /// <para>
    /// A weighted ensemble normally picks one member per call, which is right for generating a candidate: you want
    /// one answer, and the weights decide whose. Judging is the opposite case. A score is an opinion, and one model's
    /// opinion of a program is noisier than several combined, which is why the reference implementation asks every
    /// evaluator model and weights each one's metrics by its ensemble weight. Setting this does the same: each member
    /// scores the candidate, and each criterion becomes the weighted mean of the members that answered.
    /// </para>
    /// <para>
    /// A member that fails or answers unusably is left out of the mean rather than counted as zero, so one
    /// unavailable provider costs precision instead of corrupting the score. It applies only when the judge is given
    /// a weighted ensemble; with a single client there is nothing to average and the setting has no effect.
    /// </para>
    /// <para><b>For Beginners:</b> Turn this on when you have several models available and want a panel of judges
    /// rather than one. It costs one call per model for every candidate judged.</para>
    /// </remarks>
    public bool JudgeWithEveryEnsembleMember { get; set; }

    /// <summary>Gets or sets whether the judge's written criticism is carried forward to the next proposal.</summary>
    /// <remarks>
    /// <para>
    /// A judge that scores 0.4 for readability has a reason, and that reason is worth more to the next proposal than
    /// the number is. When this is on, the evaluator reads <see cref="CritiqueField"/> out of the judge's answer and
    /// attaches it to the candidate as an artifact, which the engine then shows to the model that proposes the
    /// candidate's successor. The search stops rediscovering the same criticism and starts answering it.
    /// </para>
    /// <para>
    /// The text is written by a language model about a program that was itself generated: it is untrusted, it is
    /// bounded by <see cref="MaxCritiqueChars"/>, and it reaches the next prompt through the same redacted artifact
    /// path as any other evaluator output. It never affects the score — only the scores in
    /// <see cref="Criteria"/> do that — so a judge that writes nothing useful costs a little prompt space and
    /// nothing else.
    /// </para>
    /// <para><b>For Beginners:</b> Leave this on to let the next attempt see what the judge disliked about the last
    /// one. Turn it off to keep prompts shorter, or if you would rather the search not read the judge's prose.</para>
    /// </remarks>
    public bool CarryCritiqueForward { get; set; } = true;

    /// <summary>Gets or sets the JSON field the judge's written criticism is read from.</summary>
    /// <remarks>
    /// The derived response schema asks for this field alongside the criteria, so a judge following the schema fills
    /// it in. A missing or blank field is not an error: the scores are still used and nothing is carried forward.
    /// </remarks>
    public string CritiqueField { get; set; } = DefaultCritiqueField;

    /// <summary>Gets or sets the largest critique carried forward, in characters.</summary>
    /// <remarks>Longer text is cut and marked as truncated rather than dropped.</remarks>
    public int MaxCritiqueChars { get; set; } = 1_200;

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
        JudgeWithEveryEnsembleMember = JudgeWithEveryEnsembleMember,
        CarryCritiqueForward = CarryCritiqueForward,
        CritiqueField = CritiqueField,
        MaxCritiqueChars = MaxCritiqueChars,
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
        if (CritiqueField is not { } critiqueField || critiqueField.Trim().Length == 0)
            throw new ArgumentException("CritiqueField cannot be empty or white space.", nameof(CritiqueField));

        // A critique field colliding with a criterion would make the same key mean both a score and prose, and the
        // judge would have to pick one. Rejecting it here beats an unparseable answer per candidate at run time.
        if (seen.Contains(CritiqueField.Trim()))
            throw new ArgumentException("CritiqueField cannot name one of the judging criteria.", nameof(CritiqueField));
        if (MaxCritiqueChars < 1 || MaxCritiqueChars > EvolutionArtifact.MaximumTextLength)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxCritiqueChars), MaxCritiqueChars,
                "Value must be between 1 and " + EvolutionArtifact.MaximumTextLength.ToString(CultureInfo.InvariantCulture) + ".");
        }
        if (Temperature.HasValue
            && (double.IsNaN(Temperature.Value) || double.IsInfinity(Temperature.Value)
                || Temperature.Value < 0 || Temperature.Value > 2))
        {
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature.Value,
                "Value must be a finite number between 0 and 2.");
        }
    }
}
