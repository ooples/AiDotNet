using System.Globalization;
using System.Text;
using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Prompts;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

// AiDotNet.PromptEngineering.Templates is imported project-wide and also declares a ChatMessage type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Evolution.Programs;

/// <summary>Proposes the next candidate program by asking a chat model to edit or rewrite the parent.</summary>
/// <typeparam name="T">
/// The numeric type the AiDotNet chat abstraction is parameterized on, matching the chat client supplied to the
/// constructor. It is a marker for ecosystem consistency and does not affect prompting.
/// </typeparam>
/// <remarks>
/// <para>
/// The operator turns one archive parent into one new <see cref="ProgramGenome"/>. It renders the prompt through a
/// <see cref="ProgramPromptBuilder"/>, so the parent, its score, its archive cell, the sibling programs the
/// selection policy offered as inspiration, and the evaluator's captured output all reach the model through the
/// templated, bounded, redacted path rather than through ad-hoc string concatenation. The answer is parsed with
/// <see cref="ProgramDiff"/> or <see cref="FencedCodeExtractor"/> according to the mode the builder resolved.
/// Nothing is executed here; the resulting genome goes back to the engine, which canonicalizes it and passes it to
/// whatever sandboxed evaluator the caller configured.
/// </para>
/// <para>
/// Two behaviours make this strictly better than the reference OpenEvolve worker loop. First, an answer whose edits
/// do not apply is not thrown away in silence: the typed failures become a short feedback message and the model is
/// asked again, up to <see cref="LlmProgramVariationOptions.MaxProposalRetries"/> times, where upstream logs
/// "No valid diffs found" and loses the iteration. Second, when every attempt fails the operator returns the parent
/// genome unchanged, so the engine recognizes a duplicate and the proposal costs no evaluator budget at all,
/// whereas upstream a child identical to its parent is evaluated like any other candidate. Every request is
/// recorded as a <see cref="ProgramProposalAttempt"/> and summed into <see cref="GetUsage"/>, so the cost of
/// unusable answers is measurable rather than invisible.
/// </para>
/// <para>
/// Determinism follows the engine contract: the prompt is rendered from the proposal's <see cref="StableRandom"/>
/// stream and the sampling seed is drawn from the same stream unless
/// <see cref="LlmProgramVariationOptions.Seed"/> pins one, so replaying a run sends the model identical requests.
/// The prompt builder's own version hash is folded into <see cref="VersionHash"/>, so editing a template is visible
/// to checkpoint resume instead of silently changing what the model is shown mid-run. Security follows the
/// untrusted-content rule: program text, artifacts, and diagnostics are truncated and redacted by the builder,
/// feedback messages carry bounded control-character-sanitized excerpts, and a provider exception contributes only
/// its type name, never its message, so credentials and endpoints cannot leak into a prompt or a log.
/// </para>
/// <para><b>For Beginners:</b> This is the piece that actually asks an AI to improve your program. It shows the
/// model the current program along with its score, what the tests printed, and a couple of other programs that did
/// well, then reads the answer and hands the improved program back to the search. If the answer cannot be used — a
/// common case, usually because the model asked to replace text that is not in the file — it explains the problem
/// and asks again instead of wasting the round. You supply the chat client, so no model is contacted unless you
/// configure one.</para>
/// </remarks>
public sealed class LlmProgramVariationOperator<T> : IVariationOperator<ProgramGenome>
{
    private const int MaxFeedbackFailures = 5;
    private const int MaxFeedbackChars = 1_200;

    private readonly IChatClient<T> _chatClient;
    private readonly ProgramEvolutionOptions _programOptions;
    private readonly LlmProgramVariationOptions _variationOptions;
    private readonly ProgramPromptBuilder _promptBuilder;
    private readonly object _attemptLock = new();
    private readonly Queue<ProgramProposalAttempt> _attempts = new();
    private long _proposals;
    private long _chatCalls;
    private long _retries;
    private long _abandoned;
    private long _providerErrors;
    private long _inputTokens;
    private long _outputTokens;

    /// <summary>Initializes a language-model variation operator.</summary>
    /// <param name="chatClient">The chat client used to request proposals; supplied by the caller.</param>
    /// <param name="programOptions">Program bounds, language, and diff behaviour; <c>null</c> uses the defaults.</param>
    /// <param name="variationOptions">Prompting, retry, and sampling settings; <c>null</c> uses the defaults.</param>
    /// <param name="id">A stable operator identifier recorded in candidate lineage.</param>
    /// <param name="promptBuilder">
    /// An explicit prompt builder, or <c>null</c> to derive one from <paramref name="programOptions"/>. When one is
    /// supplied it owns the prompt entirely, so <see cref="LlmProgramVariationOptions.Mode"/>,
    /// <see cref="LlmProgramVariationOptions.SystemMessage"/>, and
    /// <see cref="LlmProgramVariationOptions.MaxPromptProgramChars"/> are then read from the builder instead. When
    /// one is derived, those three settings override the matching fields of
    /// <see cref="ProgramEvolutionOptions.Prompt"/>, except that an explicit
    /// <see cref="ProgramPromptEvolutionMode.AutoBySize"/> is preserved because the variation options have no
    /// equivalent for it.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="chatClient"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> is empty or white space, or an option is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public LlmProgramVariationOperator(
        IChatClient<T> chatClient,
        ProgramEvolutionOptions? programOptions = null,
        LlmProgramVariationOptions? variationOptions = null,
        string id = "llm-program-variation",
        ProgramPromptBuilder? promptBuilder = null)
    {
        Guard.NotNull(chatClient);
        Guard.NotNullOrWhiteSpace(id);

        ProgramEvolutionOptions programCopy = (programOptions ?? new ProgramEvolutionOptions()).Clone();
        programCopy.Validate();
        LlmProgramVariationOptions variationCopy = (variationOptions ?? new LlmProgramVariationOptions()).Clone();
        variationCopy.Validate();

        _chatClient = chatClient;
        _programOptions = programCopy;
        _variationOptions = variationCopy;
        _promptBuilder = promptBuilder ?? new ProgramPromptBuilder(
            DerivePromptOptions(programCopy, variationCopy), programCopy);
        Id = id.Trim();
        VersionHash = BuildVersionHash(programCopy, variationCopy, _promptBuilder);
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <summary>Gets the prompt builder that renders every request this operator sends.</summary>
    public ProgramPromptBuilder PromptBuilder => _promptBuilder;

    /// <summary>Gets a copy of the program bounds this operator enforces on every proposal.</summary>
    /// <returns>An independent copy; mutating it does not affect the operator.</returns>
    public ProgramEvolutionOptions GetProgramOptions() => _programOptions.Clone();

    /// <summary>Gets a copy of the prompting and retry settings this operator uses.</summary>
    /// <returns>An independent copy; mutating it does not affect the operator.</returns>
    public LlmProgramVariationOptions GetVariationOptions() => _variationOptions.Clone();

    /// <summary>Gets the language-model totals accumulated since this operator was constructed.</summary>
    /// <returns>An immutable snapshot of the counters.</returns>
    /// <remarks>
    /// Token counts are provider-reported and stay at zero when the chat client returns no usage information.
    /// Reading the totals never resets them, so a caller may sample them mid-run.
    /// </remarks>
    public ProgramEvolutionLlmUsage GetUsage() => new(
        Interlocked.Read(ref _proposals),
        Interlocked.Read(ref _chatCalls),
        Interlocked.Read(ref _retries),
        Interlocked.Read(ref _abandoned),
        Interlocked.Read(ref _providerErrors),
        Interlocked.Read(ref _inputTokens),
        Interlocked.Read(ref _outputTokens));

    /// <summary>Gets the most recent recorded proposal attempts, oldest first.</summary>
    /// <returns>
    /// Up to <see cref="LlmProgramVariationOptions.MaxRecordedAttempts"/> attempts; empty when recording is off.
    /// </returns>
    public IReadOnlyList<ProgramProposalAttempt> GetRecentAttempts()
    {
        lock (_attemptLock)
        {
            return _attempts.ToArray();
        }
    }

    /// <inheritdoc/>
    public async ValueTask<ProgramGenome> ProposeAsync(
        EvolutionVariationContext<ProgramGenome> context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(context);
        cancellationToken.ThrowIfCancellationRequested();

        ProgramGenome parent = context.Parent.Candidate.CanonicalGenome.Genome;
        Interlocked.Increment(ref _proposals);

        ProgramPromptResult prompt = _promptBuilder.Build(BuildPromptContext(context, parent), context.Random);
        var messages = new List<ChatMessage>(prompt.Messages);

        int attempts = _variationOptions.MaxProposalRetries + 1;
        int attemptNumber = 0;
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            attemptNumber = attempt + 1;
            if (attempt > 0) Interlocked.Increment(ref _retries);
            ChatOptions chatOptions = BuildChatOptions(context.Random);

            string responseText = string.Empty;
            int inputTokens = 0;
            int outputTokens = 0;
            try
            {
                Interlocked.Increment(ref _chatCalls);
                ChatResponse response = await _chatClient
                    .GetResponseAsync(messages, chatOptions, cancellationToken)
                    .ConfigureAwait(false);
                responseText = response is null ? string.Empty : response.Text;
                if (response?.Usage is { } usage)
                {
                    inputTokens = usage.InputTokens;
                    outputTokens = usage.OutputTokens;
                    Interlocked.Add(ref _inputTokens, usage.InputTokens);
                    Interlocked.Add(ref _outputTokens, usage.OutputTokens);
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
#pragma warning disable CA1031
            catch (Exception exception)
#pragma warning restore CA1031
            {
                Interlocked.Increment(ref _providerErrors);
                string typeName = exception.GetType().Name;
                Record(parent.Id, attemptNumber, ProgramProposalOutcome.ProviderError, typeName);
                AppendFeedback(messages, responseText: null, "The previous request failed with " + typeName + ".");
                continue;
            }

            ProgramProposalOutcome outcome = TryBuildChild(
                parent, responseText, prompt.Mode, out ProgramGenome child, out string feedback);
            Record(parent.Id, attemptNumber, outcome, feedback, inputTokens, outputTokens);
            if (outcome == ProgramProposalOutcome.Accepted) return child;
            AppendFeedback(messages, responseText, feedback);
        }

        Interlocked.Increment(ref _abandoned);
        Record(parent.Id, Math.Max(attemptNumber, 1), ProgramProposalOutcome.Exhausted,
            "Every permitted attempt failed; the parent was returned unchanged.");
        return parent;
    }

    private ProgramPromptContext BuildPromptContext(
        EvolutionVariationContext<ProgramGenome> context,
        ProgramGenome parent)
    {
        EvolutionEvaluation evaluation = context.Parent.Evaluation;
        var promptContext = new ProgramPromptContext(parent)
        {
            Direction = evaluation.Direction,
            Inspirations = BuildInspirations(context, parent)
        };

        if (_variationOptions.IncludeParentMetrics)
        {
            promptContext.ParentQuality = evaluation.Quality;
            promptContext.ParentMetrics = evaluation.Descriptors;
            promptContext.ParentDescriptors = evaluation.Descriptors;
            ApplyFeatureCoordinates(promptContext, context.Parent.Cell.Bins);
        }

        SplitDiagnostics(evaluation.Diagnostics, promptContext);
        return promptContext;
    }

    private void ApplyFeatureCoordinates(ProgramPromptContext promptContext, IReadOnlyList<int> bins)
    {
        IList<string> names = _variationOptions.FeatureDimensions;
        if (names.Count == 0) return;

        promptContext.FeatureDimensions = new List<string>(names);
        // The engine hands over bare bin indices, so they are only shown when the configured names line up with
        // them; a stale configuration then loses the indices rather than mislabelling them.
        if (bins.Count == names.Count) promptContext.FeatureBins = new List<int>(bins);
        if (_variationOptions.FeatureBinCounts.Count == names.Count)
        {
            promptContext.FeatureBinCounts = new List<int>(_variationOptions.FeatureBinCounts);
        }
    }

    private void SplitDiagnostics(
        IReadOnlyList<EvolutionDiagnostic> diagnostics,
        ProgramPromptContext promptContext)
    {
        if (diagnostics.Count == 0) return;

        string prefix = _variationOptions.ArtifactDiagnosticPrefix;
        var artifacts = new List<ProgramPromptArtifact>();
        var remaining = new List<EvolutionDiagnostic>();
        foreach (EvolutionDiagnostic diagnostic in diagnostics)
        {
            if (prefix.Length > 0 && diagnostic.Code.StartsWith(prefix, StringComparison.Ordinal))
            {
                artifacts.Add(new ProgramPromptArtifact(diagnostic.Code, diagnostic.Message));
            }
            else
            {
                remaining.Add(diagnostic);
            }
        }

        if (artifacts.Count > 0) promptContext.Artifacts = artifacts;
        if (remaining.Count > 0) promptContext.Diagnostics = remaining;
    }

    private IReadOnlyList<ProgramPromptExample> BuildInspirations(
        EvolutionVariationContext<ProgramGenome> context,
        ProgramGenome parent)
    {
        var examples = new List<ProgramPromptExample>();
        if (_variationOptions.MaxInspirations == 0) return examples;

        // The parent is pre-seeded so it can never be quoted back to the model as its own inspiration.
        var seen = new HashSet<string>(StringComparer.Ordinal) { parent.Id };
        foreach (EvolutionArchiveEntry<ProgramGenome> inspiration in context.Inspirations)
        {
            if (examples.Count >= _variationOptions.MaxInspirations) break;
            ProgramGenome genome = inspiration.Candidate.CanonicalGenome.Genome;
            if (!seen.Add(genome.Id)) continue;

            examples.Add(new ProgramPromptExample(
                genome,
                ProgramPromptExampleKind.Inspiration,
                inspiration.Evaluation.Quality,
                inspiration.Evaluation.Descriptors));
        }

        return examples;
    }

    private ProgramProposalOutcome TryBuildChild(
        ProgramGenome parent,
        string responseText,
        ProgramPromptEvolutionMode mode,
        out ProgramGenome child,
        out string feedback)
    {
        child = parent;
        if (string.IsNullOrWhiteSpace(responseText))
        {
            feedback = "The previous answer was empty.";
            return ProgramProposalOutcome.EmptyResponse;
        }

        string candidateSource;
        if (mode == ProgramPromptEvolutionMode.FullRewrite)
        {
            FencedCodeExtractionResult extraction = FencedCodeExtractor.Extract(
                responseText, _programOptions.Language, allowRawFallback: false);
            if (!extraction.HasCode)
            {
                feedback = "No fenced code block was found. Return the complete program inside one fenced block.";
                return ProgramProposalOutcome.ParseFailed;
            }

            candidateSource = extraction.Code;
        }
        else
        {
            ProgramDiffApplyResult applied = ProgramDiff.ApplyResponse(parent.Source, responseText, _programOptions);
            if (!applied.IsSuccess)
            {
                feedback = DescribeFailures(applied.Failures);
                return ProgramProposalOutcome.ParseFailed;
            }

            candidateSource = applied.ModifiedSource;
        }

        string normalized = ProgramGenome.Normalize(candidateSource);
        if (normalized.Length == 0)
        {
            feedback = "The proposed program was empty.";
            return ProgramProposalOutcome.EmptyResponse;
        }

        if (normalized.Length > _programOptions.MaxProgramChars || candidateSource.Length > ProgramGenome.MaxSourceLength)
        {
            feedback = "The proposed program is " + normalized.Length.ToString(CultureInfo.InvariantCulture) +
                " characters, above the limit of " + _programOptions.MaxProgramChars.ToString(CultureInfo.InvariantCulture) +
                ". Return a shorter program.";
            return ProgramProposalOutcome.TooLong;
        }

        if (string.Equals(normalized, parent.NormalizedSource, StringComparison.Ordinal))
        {
            feedback = "The proposed program is identical to the current one. Make a substantive change.";
            return ProgramProposalOutcome.Unchanged;
        }

        child = new ProgramGenome(candidateSource, parent.Language);
        feedback = string.Empty;
        return ProgramProposalOutcome.Accepted;
    }

    private void Record(
        string parentId,
        int attemptNumber,
        ProgramProposalOutcome outcome,
        string detail,
        int inputTokens = 0,
        int outputTokens = 0)
    {
        int capacity = _variationOptions.MaxRecordedAttempts;
        if (capacity == 0) return;

        var attempt = new ProgramProposalAttempt(parentId, attemptNumber, outcome, detail, inputTokens, outputTokens);
        lock (_attemptLock)
        {
            while (_attempts.Count >= capacity) _attempts.Dequeue();
            _attempts.Enqueue(attempt);
        }
    }

    private ChatOptions BuildChatOptions(StableRandom random)
    {
        int seed = _variationOptions.Seed ?? unchecked((int)(random.NextUInt32() & 0x7FFFFFFF));
        return new ChatOptions
        {
            Temperature = _variationOptions.Temperature,
            MaxOutputTokens = _variationOptions.MaxOutputTokens,
            Seed = seed
        };
    }

    private static ProgramEvolutionPromptOptions DerivePromptOptions(
        ProgramEvolutionOptions programOptions,
        LlmProgramVariationOptions variationOptions)
    {
        ProgramEvolutionPromptOptions prompt = programOptions.Prompt.Clone();
        prompt.MaxProgramSnippetChars = variationOptions.MaxPromptProgramChars;

        // AutoBySize is a deliberate choice with no equivalent on the variation options, so it survives; every
        // other value is driven by Mode, because the parser and the requested answer format must agree.
        if (prompt.EvolutionMode != ProgramPromptEvolutionMode.AutoBySize)
        {
            prompt.EvolutionMode = variationOptions.Mode == ProgramEvolutionMode.FullRewrite
                ? ProgramPromptEvolutionMode.FullRewrite
                : ProgramPromptEvolutionMode.Diff;
        }

        if (variationOptions.SystemMessage is { } system && system.Trim().Length > 0)
        {
            prompt.SystemMessage = system;
            prompt.SystemMessageMode = ProgramPromptSystemMessageMode.Literal;
        }

        if (prompt.TaskDescription is null && programOptions.TaskDescription is { } task && task.Trim().Length > 0)
        {
            prompt.TaskDescription = task;
        }

        return prompt;
    }

    private static void AppendFeedback(List<ChatMessage> messages, string? responseText, string feedback)
    {
        messages.Add(ChatMessage.Assistant(
            responseText is null ? "(no response)" : ProgramText.Bound(responseText, MaxFeedbackChars)));
        messages.Add(ChatMessage.User(feedback + " Answer again in the required format."));
    }

    private static string DescribeFailures(IReadOnlyList<ProgramDiffFailure> failures)
    {
        if (failures.Count == 0) return "The previous answer produced no usable edit.";

        var builder = new StringBuilder("The previous answer could not be applied:");
        int shown = Math.Min(failures.Count, MaxFeedbackFailures);
        for (int index = 0; index < shown; index++)
        {
            builder.Append("\n- ").Append(ProgramText.Bound(failures[index].Message, 300));
            if (failures[index].SearchExcerpt.Length > 0)
            {
                builder.Append(" Search text was: ").Append(ProgramText.Bound(failures[index].SearchExcerpt, 200));
            }
        }

        if (failures.Count > shown)
        {
            builder.Append("\n- and ")
                .Append((failures.Count - shown).ToString(CultureInfo.InvariantCulture))
                .Append(" more problems.");
        }

        return ProgramText.Bound(builder.ToString(), MaxFeedbackChars);
    }

    private static string BuildVersionHash(
        ProgramEvolutionOptions programOptions,
        LlmProgramVariationOptions variationOptions,
        ProgramPromptBuilder promptBuilder)
    {
        var components = new List<string>
        {
            "llm-program-variation-v2",
            programOptions.Language.ToString(),
            programOptions.ResolveEvolveBlockMarkers().ToString(),
            programOptions.EnforceEvolveBlocks ? "enforce" : "free",
            programOptions.MaxProgramChars.ToString(CultureInfo.InvariantCulture),
            programOptions.Diff.SearchMarker,
            programOptions.Diff.DividerMarker,
            programOptions.Diff.ReplaceMarker,
            programOptions.Diff.FuzzyWhitespace ? "fuzzy" : "exact",
            ((int)variationOptions.Mode).ToString(CultureInfo.InvariantCulture),
            variationOptions.MaxProposalRetries.ToString(CultureInfo.InvariantCulture),
            variationOptions.MaxInspirations.ToString(CultureInfo.InvariantCulture),
            variationOptions.MaxPromptProgramChars.ToString(CultureInfo.InvariantCulture),
            variationOptions.SystemMessage ?? string.Empty,
            variationOptions.ArtifactDiagnosticPrefix,
            string.Join(",", variationOptions.FeatureDimensions),
            variationOptions.Seed.HasValue
                ? variationOptions.Seed.Value.ToString(CultureInfo.InvariantCulture)
                : "stream",
            promptBuilder.VersionHash
        };

        return "llm-program-variation-" + EvolutionHash.Combine(components);
    }
}
