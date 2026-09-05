using System.Globalization;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Prompts;
using AiDotNet.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json.Linq;

// AiDotNet.PromptEngineering.Templates is imported project-wide and also declares a ChatMessage type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Evolution.Programs;

/// <summary>Blends a language model's judgement of a program into the score a measured evaluator produced.</summary>
/// <typeparam name="T">
/// The numeric type the AiDotNet chat abstraction is parameterized on, matching the chat client supplied to the
/// constructor. It is a marker for ecosystem consistency and does not affect judging.
/// </typeparam>
/// <remarks>
/// <para>
/// The evaluator runs the inner <see cref="IProgramFitnessEvaluator"/> first, then asks the model to score the same
/// program against <see cref="LlmFeedbackOptions.Criteria"/> and mixes the two. The measured result keeps
/// <see cref="LlmFeedbackOptions.CombinedBlend"/> of the final quality, so a judge cannot overturn a failing test
/// suite. Every individual judge score is recorded as its own descriptor under
/// <see cref="LlmFeedbackOptions.MetricPrefix"/>, together with their mean, so a run can be analysed afterwards to
/// see whether the judge and the tests agreed.
/// </para>
/// <para>
/// Four things make this strictly better than the reference OpenEvolve feedback path. The blend is a configured
/// number rather than a hard-coded 0.7/0.3 in the evaluator body. The configured weight is applied consistently,
/// where upstream weights the criteria average and then blends the unweighted one, so its weight silently does
/// nothing. A candidate whose own evaluation failed is not judged at all by default, where upstream judges error
/// results and lets a crashed program earn a respectable combined score. And the request asks the provider to
/// constrain its answer to JSON, so on a provider that supports constrained decoding an unparseable judge answer
/// is impossible rather than merely unlikely; where it is still unparseable, the measured quality passes through
/// untouched and a bounded diagnostic records why.
/// </para>
/// <para>
/// Determinism is preserved: the judge's sampling seed is derived from the evaluation's own random stream, so a
/// replayed run asks the judge identically. Security follows the untrusted-content rule — the program reaches the
/// prompt only through the bounding and redaction the prompt builder applies, and a provider exception contributes
/// only its type name to a diagnostic, never its message.
/// </para>
/// <para><b>For Beginners:</b> Tests tell you whether a program works, but not whether it is well written or
/// likely to keep working on inputs you did not think of. This asks a language model those questions and folds its
/// answer into the score, with the tests keeping most of the weight so a broken program cannot be flattered into
/// winning. If the model answers with something unreadable, the score simply stays as your tests measured it.</para>
/// </remarks>
public sealed class LlmJudgeProgramFitnessEvaluator<T> : IProgramFitnessEvaluator
{
    private const int MaxDiagnosticLength = 240;

    private readonly IChatClient<T> _chatClient;
    private readonly IProgramFitnessEvaluator _inner;
    private readonly ProgramPromptBuilder _promptBuilder;
    private readonly LlmFeedbackOptions _options;
    private readonly string[] _criteria;
    private readonly string[] _fieldNames;
    private long _judgeCalls;
    private long _judgeFailures;

    /// <summary>Initializes a judging evaluator around a measured one.</summary>
    /// <param name="chatClient">The chat client that answers judging requests; supplied by the caller.</param>
    /// <param name="inner">The measured evaluator whose score the judge adjusts.</param>
    /// <param name="promptBuilder">The builder that renders judging prompts; <c>null</c> uses the defaults.</param>
    /// <param name="options">Criteria, weighting, and blending settings; <c>null</c> uses the defaults.</param>
    /// <param name="id">A stable evaluator identifier.</param>
    /// <exception cref="ArgumentNullException"><paramref name="chatClient"/> or <paramref name="inner"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> is empty or white space, or an option is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public LlmJudgeProgramFitnessEvaluator(
        IChatClient<T> chatClient,
        IProgramFitnessEvaluator inner,
        ProgramPromptBuilder? promptBuilder = null,
        LlmFeedbackOptions? options = null,
        string id = "llm-judge-program-evaluator")
    {
        Guard.NotNull(chatClient);
        Guard.NotNull(inner);
        Guard.NotNullOrWhiteSpace(id);

        LlmFeedbackOptions copy = (options ?? new LlmFeedbackOptions()).Clone();
        copy.Validate();

        _chatClient = chatClient;
        _inner = inner;
        _promptBuilder = promptBuilder ?? new ProgramPromptBuilder();
        _options = copy;
        _criteria = copy.Criteria.Select(criterion => criterion.Trim()).ToArray();
        _fieldNames = _criteria.Select(ProgramPromptBuilder.ToCriterionFieldName).ToArray();
        Id = id.Trim();
        VersionHash = BuildVersionHash(copy, _criteria, inner, _promptBuilder);
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <summary>Gets the measured evaluator whose score this judge adjusts.</summary>
    public IProgramFitnessEvaluator Inner => _inner;

    /// <summary>Gets a copy of the judging settings this evaluator uses.</summary>
    /// <returns>An independent copy; mutating it does not affect the evaluator.</returns>
    public LlmFeedbackOptions GetOptions() => _options.Clone();

    /// <summary>Gets how many judging requests were sent since this evaluator was constructed.</summary>
    public long JudgeCalls => Interlocked.Read(ref _judgeCalls);

    /// <summary>Gets how many candidates ended with no usable judge answer and kept their measured score.</summary>
    public long JudgeFailures => Interlocked.Read(ref _judgeFailures);

    /// <inheritdoc/>
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(context);

        EvolutionTaskResult measured = await _inner
            .EvaluateAsync(candidate, context, cancellationToken)
            .ConfigureAwait(false);

        if (measured is null)
        {
            return EvolutionTaskResult.Failed(
                "llm_judge_inner_returned_null", "The measured evaluator returned no result.");
        }

        if (!_options.Enabled) return measured;
        if (measured.Status != EvolutionEvaluationStatus.Completed && !_options.RunOnFailedEvaluations) return measured;
        if (!measured.Quality.HasValue) return measured;

        JudgeOutcome outcome = await JudgeAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        if (!outcome.HasScores)
        {
            Interlocked.Increment(ref _judgeFailures);
            return Append(measured, measured.Quality.Value, null, outcome.Diagnostic);
        }

        double blended = (_options.CombinedBlend * measured.Quality.Value)
            + ((1.0 - _options.CombinedBlend) * outcome.Average);
        return Append(measured, blended, outcome, outcome.Diagnostic);
    }

    private async Task<JudgeOutcome> JudgeAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<ChatMessage> messages = _promptBuilder.BuildEvaluationMessages(
            candidate, _criteria, _options.ResponseSchema);
        StableRandom random = context.CreateRandom();

        // The ensemble is usually wrapped: the standard way to build a production client adds retry and telemetry
        // middleware around it, and a plain type test on the outermost object would then quietly fall back to
        // single-member judging with nothing to say why.
        if (_options.JudgeWithEveryEnsembleMember && FindPanel(_chatClient) is { } panel)
        {
            return await JudgeWithPanelAsync(panel, messages, random, cancellationToken).ConfigureAwait(false);
        }

        string? lastProblem = null;
        int attempts = _options.MaxJudgeRetries + 1;
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref _judgeCalls);

            string answer;
            try
            {
                ChatResponse response = await _chatClient
                    .GetResponseAsync(messages, BuildChatOptions(random), cancellationToken)
                    .ConfigureAwait(false);
                answer = response is null ? string.Empty : response.Text;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
#pragma warning disable CA1031
            catch (Exception exception)
#pragma warning restore CA1031
            {
                // Only the exception type reaches a diagnostic; a provider message can carry a key or an endpoint.
                lastProblem = "the request failed with " + exception.GetType().Name;
                continue;
            }

            if (TryReadScores(answer, out double[] scores, out string? critique, out string problem))
            {
                return JudgeOutcome.FromScores(_fieldNames, scores, _options.Weight, critique);
            }

            lastProblem = problem;
        }

        return JudgeOutcome.Unusable(new EvolutionDiagnostic(
            "llm_judge_unusable",
            ProgramText.Bound(
                "The judge produced no usable scores after " +
                attempts.ToString(CultureInfo.InvariantCulture) + " attempts: " +
                (lastProblem ?? "no reason was recorded") + ".",
                MaxDiagnosticLength),
            isRedacted: true));
    }

    /// <summary>Finds the weighted ensemble a client is, or wraps, so panel judging survives the usual middleware.</summary>
    /// <param name="client">The configured chat client.</param>
    /// <returns>The ensemble, or <c>null</c> when there is none to find.</returns>
    /// <remarks>
    /// Unwrapping is bounded rather than recursive without limit, so a pipeline that somehow refers to itself cannot
    /// spin here.
    /// </remarks>
    private static WeightedEnsembleChatClient<T>? FindPanel(IChatClient<T> client)
    {
        IChatClient<T>? current = client;
        for (int depth = 0; current is not null && depth < 8; depth++)
        {
            if (current is WeightedEnsembleChatClient<T> panel) return panel;
            current = (current as IChatClientDecorator<T>)?.Inner;
        }
        return null;
    }

    /// <summary>Scores a candidate with every ensemble member and averages each criterion by member weight.</summary>
    /// <remarks>
    /// One model's opinion of a program is noisier than several combined, and a panel is what the reference
    /// implementation uses for exactly that reason. A member that fails or answers unusably is left out of the mean
    /// rather than counted as zero, so an unavailable provider costs precision instead of corrupting the score, and
    /// a panel where nobody answered is reported as unusable rather than as a score of zero.
    /// </remarks>
    private async Task<JudgeOutcome> JudgeWithPanelAsync(
        WeightedEnsembleChatClient<T> panel,
        IReadOnlyList<ChatMessage> messages,
        StableRandom random,
        CancellationToken cancellationToken)
    {
        Interlocked.Add(ref _judgeCalls, panel.Members.Count);
        IReadOnlyList<ChatResponse?> responses;
        try
        {
            responses = await panel
                .GetAllResponsesAsync(messages, BuildChatOptions(random), cancellationToken)
                .ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
#pragma warning disable CA1031
        catch (Exception exception)
#pragma warning restore CA1031
        {
            // Only the exception type reaches a diagnostic; a provider message can carry a key or an endpoint.
            return JudgeOutcome.Unusable(new EvolutionDiagnostic(
                "llm_judge_unusable",
                "The judge panel failed with " + exception.GetType().Name + ".",
                isRedacted: true));
        }

        var totals = new double[_fieldNames.Length];
        var weights = new double[_fieldNames.Length];
        int answered = 0;
        string? lastProblem = null;
        var critiques = new List<string>();

        for (int index = 0; index < responses.Count && index < panel.Members.Count; index++)
        {
            ChatResponse? response = responses[index];
            if (response is null)
            {
                lastProblem = "a member returned no answer";
                continue;
            }

            if (!TryReadScores(response.Text, out double[] scores, out string? memberCritique, out string problem))
            {
                lastProblem = problem;
                continue;
            }

            double weight = panel.Members[index].Weight;
            if (weight <= 0 || double.IsNaN(weight) || double.IsInfinity(weight)) continue;

            // Numbered rather than named: a member's model identifier is configuration, and the next prompt has no
            // use for it beyond telling two opinions apart.
            if (memberCritique is not null)
            {
                critiques.Add("Judge " + (critiques.Count + 1).ToString(CultureInfo.InvariantCulture) + ": " + memberCritique);
            }

            answered++;
            for (int field = 0; field < totals.Length && field < scores.Length; field++)
            {
                totals[field] += scores[field] * weight;
                weights[field] += weight;
            }
        }

        if (answered == 0)
        {
            return JudgeOutcome.Unusable(new EvolutionDiagnostic(
                "llm_judge_unusable",
                ProgramText.Bound(
                    "No member of the judge panel produced usable scores: " +
                    (lastProblem ?? "no reason was recorded") + ".",
                    MaxDiagnosticLength),
                isRedacted: true));
        }

        var averaged = new double[totals.Length];
        for (int field = 0; field < totals.Length; field++)
            averaged[field] = weights[field] > 0 ? totals[field] / weights[field] : 0;

        // One combined critique rather than one artifact per member: the bound is on the whole text either way, and
        // a single ordered block is what a proposing model can actually read.
        string? panelCritique = critiques.Count == 0
            ? null
            : ProgramText.Bound(string.Join("\n\n", critiques), _options.MaxCritiqueChars);

        return JudgeOutcome.FromScores(_fieldNames, averaged, _options.Weight, panelCritique);
    }

    private bool TryReadScores(string answer, out double[] scores, out string? critique, out string problem)
    {
        scores = Array.Empty<double>();
        critique = null;
        if (string.IsNullOrWhiteSpace(answer))
        {
            problem = "the answer was empty";
            return false;
        }

        if (!LlmJsonExtractor.TryExtract(answer, out JObject json))
        {
            problem = "the answer held no JSON object";
            return false;
        }

        var values = new double[_fieldNames.Length];
        for (int index = 0; index < _fieldNames.Length; index++)
        {
            if (!LlmJsonExtractor.TryReadNumber(json, _fieldNames[index], out double value))
            {
                problem = "the answer had no finite number for one criterion";
                return false;
            }

            // A judge that answers 1.4 is clamped rather than rejected: the intent is unambiguous and refusing it
            // would spend another call to learn nothing.
            values[index] = value < 0 ? 0 : value > 1 ? 1 : value;
        }

        scores = values;
        critique = ReadCritique(json);
        problem = string.Empty;
        return true;
    }

    /// <summary>Reads the judge's written criticism out of an answer that already parsed.</summary>
    /// <param name="json">The judge's answer.</param>
    /// <returns>The bounded, sanitized criticism, or <c>null</c> when there is none worth carrying.</returns>
    /// <remarks>
    /// A missing or blank field is not a failure. The scores are the contract; the prose is a bonus, and refusing an
    /// otherwise-good answer because the judge stayed quiet would spend a call to learn nothing.
    /// </remarks>
    private string? ReadCritique(JObject json)
    {
        if (!_options.CarryCritiqueForward) return null;

        JToken? token = json[_options.CritiqueField.Trim()];
        if (token is null || token.Type == JTokenType.Null) return null;

        // Read as a scalar string: a judge that answers with an object or an array here has not followed the schema,
        // and serializing whatever it did send would put unbounded JSON into the next prompt.
        // Written as an explicit null test rather than IsNullOrWhiteSpace, which net471 does not annotate for flow
        // analysis, so the shorter form fails the nullable build on that target alone.
        if (token.Type != JTokenType.String || token.Value<string>() is not { } text || text.Trim().Length == 0)
            return null;

        string sanitized = ProgramText.Sanitize(text).Trim();
        return sanitized.Length == 0 ? null : ProgramText.Bound(sanitized, _options.MaxCritiqueChars);
    }

    private ChatOptions BuildChatOptions(StableRandom random)
    {
        var options = new ChatOptions
        {
            Temperature = _options.Temperature,
            MaxOutputTokens = _options.MaxOutputTokens,
            Seed = unchecked((int)(random.NextUInt32() & 0x7FFFFFFF))
        };

        if (_options.RequestJsonResponseFormat) options.ResponseFormat = ChatResponseFormatKind.Json;
        return options;
    }

    private EvolutionTaskResult Append(
        EvolutionTaskResult measured,
        double quality,
        JudgeOutcome? outcome,
        EvolutionDiagnostic? diagnostic)
    {
        var descriptors = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, double> pair in measured.Descriptors) descriptors[pair.Key] = pair.Value;

        var objectives = new List<double>(measured.Objectives);
        if (outcome is not null && outcome.HasScores)
        {
            for (int index = 0; index < outcome.Names.Count; index++)
            {
                descriptors[_options.MetricPrefix + outcome.Names[index]] = outcome.Scores[index];
                if (_options.RecordObjectives) objectives.Add(outcome.Scores[index]);
            }

            descriptors[_options.MetricPrefix + LlmFeedbackOptions.AverageMetricSuffix] = outcome.Average;
        }

        var diagnostics = new List<EvolutionDiagnostic>(measured.Diagnostics);
        if (diagnostic is not null && diagnostics.Count < 64) diagnostics.Add(diagnostic);

        // The judge's written criticism travels as an artifact, which is the channel the engine already shows to
        // whoever proposes this candidate's successor. Attaching it here is what turns one judge's opinion into
        // something the search can answer rather than rediscover.
        var artifacts = new List<EvolutionArtifact>(measured.Artifacts);
        if (outcome?.Critique is { } critique && artifacts.Count < EvolutionTaskResult.MaximumArtifacts)
        {
            artifacts.Add(new EvolutionArtifact(
                LlmFeedbackOptions.CritiqueArtifactKey,
                critique,
                isTruncated: critique.Length >= _options.MaxCritiqueChars,
                isRedacted: true));
        }

        // Metrics and artifacts the measured evaluator reported are carried rather than replaced: the judge is a
        // wrapper, and a wrapper that silently drops what it wraps makes every metric query wrong for judged runs.
        return new EvolutionTaskResult(
            measured.Status,
            quality,
            measured.Direction,
            descriptors,
            objectives,
            measured.ConstraintViolations,
            measured.CostUnits,
            diagnostics,
            measured.Metrics,
            artifacts);
    }

    private static string BuildVersionHash(
        LlmFeedbackOptions options,
        IReadOnlyList<string> criteria,
        IProgramFitnessEvaluator inner,
        ProgramPromptBuilder promptBuilder)
    {
        var components = new List<string>
        {
            "llm-judge-program-evaluator-v1",
            inner.Id,
            inner.VersionHash,
            promptBuilder.VersionHash,
            options.Enabled ? "on" : "off",
            string.Join("|", criteria),
            options.Weight.ToString("R", CultureInfo.InvariantCulture),
            options.CombinedBlend.ToString("R", CultureInfo.InvariantCulture),
            options.MetricPrefix,
            options.RunOnFailedEvaluations ? "judge-failed" : "skip-failed",
            options.RecordObjectives ? "objectives" : "descriptors-only",
            options.MaxJudgeRetries.ToString(CultureInfo.InvariantCulture),
            options.ResponseSchema ?? string.Empty,
            options.RequestJsonResponseFormat ? "json" : "free",
            // The critique changes the next prompt, so two runs that carry it differently are not the same run.
            options.CarryCritiqueForward ? "critique:" + options.CritiqueField.Trim() : "no-critique",
            options.MaxCritiqueChars.ToString(CultureInfo.InvariantCulture)
        };

        return "llm-judge-" + EvolutionHash.Combine(components);
    }

    private sealed class JudgeOutcome
    {
        private JudgeOutcome(
            IReadOnlyList<string> names,
            IReadOnlyList<double> scores,
            double average,
            EvolutionDiagnostic? diagnostic,
            string? critique = null)
        {
            Names = names;
            Scores = scores;
            Average = average;
            Diagnostic = diagnostic;
            Critique = critique;
        }

        public IReadOnlyList<string> Names { get; }

        public IReadOnlyList<double> Scores { get; }

        public double Average { get; }

        public EvolutionDiagnostic? Diagnostic { get; }

        /// <summary>Gets the judge's written criticism, or <c>null</c> when it wrote none or it was not asked for.</summary>
        public string? Critique { get; }

        public bool HasScores => Scores.Count > 0;

        public static JudgeOutcome FromScores(
            IReadOnlyList<string> fieldNames,
            IReadOnlyList<double> scores,
            double weight,
            string? critique = null)
        {
            double total = 0;
            var weighted = new double[scores.Count];
            for (int index = 0; index < scores.Count; index++)
            {
                weighted[index] = scores[index] * weight;
                total += weighted[index];
            }

            return new JudgeOutcome(fieldNames, weighted, scores.Count == 0 ? 0 : total / scores.Count, null, critique);
        }

        public static JudgeOutcome Unusable(EvolutionDiagnostic diagnostic) =>
            new(Array.Empty<string>(), Array.Empty<double>(), 0, diagnostic);
    }
}
