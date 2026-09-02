using System.Diagnostics;
using System.Globalization;
using System.Text;
using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Programs.Provenance;
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
/// <para>
/// Supply an <see cref="IProposalProvenanceSink"/> and every request is additionally written out as a
/// <see cref="ProposalProvenanceRecord"/>: the parent it started from, the model that answered, a hash of the exact
/// conversation, a bounded and redacted copy of the prompt and the answer, the reported token cost, the measured
/// latency, and what the answer parsed into. Replaying that stream through
/// <see cref="ProposalProvenanceReader.BuildLineages"/> rebuilds the ancestry of any program the run produced,
/// which is the post-hoc audit and training-data story upstream gets from its per-program files. Recording is
/// entirely opt-in, never blocks a proposal, and a sink failure is counted in
/// <see cref="ProvenanceFailureCount"/> rather than ending the run. Only requests are recorded: the terminal
/// <see cref="ProgramProposalOutcome.Exhausted"/> bookkeeping entry is not a request and does not appear in the
/// stream, so every record corresponds to exactly one call.
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
    private readonly IProposalProvenanceSink? _provenanceSink;
    private readonly ProposalProvenanceOptions _provenanceOptions;
    private readonly object _attemptLock = new();
    private readonly Queue<ProgramProposalAttempt> _attempts = new();
    private long _provenanceFailures;
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
    /// <param name="provenanceSink">
    /// Where a record of every request is written, or <c>null</c> to record nothing. Recording is opt-in and adds
    /// no dependency: the operator behaves identically with no sink attached.
    /// </param>
    /// <param name="provenanceOptions">
    /// How much of each request and answer is kept; <c>null</c> uses the defaults. Ignored when
    /// <paramref name="provenanceSink"/> is <c>null</c>.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="chatClient"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> is empty or white space, or an option is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public LlmProgramVariationOperator(
        IChatClient<T> chatClient,
        ProgramEvolutionOptions? programOptions = null,
        LlmProgramVariationOptions? variationOptions = null,
        string id = "llm-program-variation",
        ProgramPromptBuilder? promptBuilder = null,
        IProposalProvenanceSink? provenanceSink = null,
        ProposalProvenanceOptions? provenanceOptions = null)
    {
        Guard.NotNull(chatClient);
        Guard.NotNullOrWhiteSpace(id);

        ProgramEvolutionOptions programCopy = (programOptions ?? new ProgramEvolutionOptions()).Clone();
        programCopy.Validate();
        LlmProgramVariationOptions variationCopy = (variationOptions ?? new LlmProgramVariationOptions()).Clone();
        variationCopy.Validate();
        ProposalProvenanceOptions provenanceCopy = (provenanceOptions ?? new ProposalProvenanceOptions()).Clone();
        provenanceCopy.Validate();

        _chatClient = chatClient;
        _programOptions = programCopy;
        _variationOptions = variationCopy;
        _provenanceSink = provenanceSink;
        _provenanceOptions = provenanceCopy;
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

    /// <summary>Gets how many provenance writes failed since this operator was constructed.</summary>
    /// <remarks>
    /// A search must not stop because a note could not be filed, so a sink failure is counted here instead of
    /// thrown. A non-zero value means the provenance stream has gaps and any lineage rebuilt from it is partial.
    /// Stays at zero when no sink is attached.
    /// </remarks>
    public long ProvenanceFailureCount => Interlocked.Read(ref _provenanceFailures);

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

        bool recordProvenance = _provenanceSink is not null && _provenanceOptions.Enabled;
        string proposalId = recordProvenance ? BuildProposalId(context) : string.Empty;
        string responseModelId = _chatClient.ModelId;

        int attempts = _variationOptions.MaxProposalRetries + 1;
        int attemptNumber = 0;
        for (int attempt = 0; attempt < attempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            attemptNumber = attempt + 1;
            if (attempt > 0) Interlocked.Increment(ref _retries);
            ChatOptions chatOptions = BuildChatOptions(context.Random);

            // Captured before the call: the conversation grows with feedback, so a record written afterwards must
            // describe the messages this particular attempt actually sent.
            string promptText = recordProvenance ? RenderConversation(messages) : string.Empty;
            string promptHash = recordProvenance ? HashConversation(messages) : string.Empty;
            DateTimeOffset requestedAt = DateTimeOffset.UtcNow;
            Stopwatch? timer = recordProvenance ? Stopwatch.StartNew() : null;

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
                if (response?.ModelId is { } reported && reported.Length > 0) responseModelId = reported;
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
                if (recordProvenance)
                {
                    await RecordProvenanceAsync(
                        BuildProvenanceRecord(
                            context, proposalId, attemptNumber, ProgramProposalOutcome.ProviderError, typeName,
                            responseModelId, prompt, promptText, promptHash, responseText: string.Empty,
                            childGenomeId: string.Empty, inputTokens: 0, outputTokens: 0, requestedAt: requestedAt,
                            latencyMilliseconds: Elapsed(timer)),
                        cancellationToken).ConfigureAwait(false);
                }

                AppendFeedback(messages, responseText: null, "The previous request failed with " + typeName + ".");
                continue;
            }

            ProgramProposalOutcome outcome = TryBuildChild(
                parent, responseText, prompt.Mode, out ProgramGenome child, out string feedback);
            Record(parent.Id, attemptNumber, outcome, feedback, inputTokens, outputTokens);
            if (recordProvenance)
            {
                await RecordProvenanceAsync(
                    BuildProvenanceRecord(
                        context, proposalId, attemptNumber, outcome, feedback, responseModelId, prompt, promptText,
                        promptHash, responseText,
                        childGenomeId: outcome == ProgramProposalOutcome.Accepted ? child.Id : string.Empty,
                        inputTokens: inputTokens, outputTokens: outputTokens, requestedAt: requestedAt,
                        latencyMilliseconds: Elapsed(timer)),
                    cancellationToken).ConfigureAwait(false);
            }

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

        // The parent's leftover evaluator output, delivered exactly once by the engine. It is what tells the model
        // why the previous attempt scored as it did, and it is untrusted text, so the builder bounds and delimits
        // it rather than splicing it in raw.
        if (context.ParentArtifacts.Count > 0)
        {
            var artifacts = new List<ProgramPromptArtifact>(context.ParentArtifacts.Count);
            foreach (EvolutionArtifact artifact in context.ParentArtifacts)
            {
                artifacts.Add(new ProgramPromptArtifact(artifact.Key, artifact.Text));
            }

            promptContext.Artifacts = artifacts;
        }

        ApplyPreviousAttempts(promptContext, context.Parent.Evaluation.GenomeId);
        ApplyArchiveContext(promptContext, context);
        SplitDiagnostics(evaluation.Diagnostics, promptContext);
        return promptContext;
    }

    /// <summary>Tells the model what has already been tried on this same parent, and how it went.</summary>
    /// <param name="promptContext">The context being built.</param>
    /// <param name="parentGenomeId">The parent whose attempt history is relevant.</param>
    /// <remarks>
    /// Within a single proposal a rejected answer is already fed back into the conversation, so the model can see
    /// its own mistake. Across proposals it cannot: the next call starts a fresh conversation from the same parent
    /// and is free to repeat an edit that failed to parse or applied to nothing. This surfaces the recorded
    /// attempts for that parent so the same dead end is not paid for twice.
    /// </remarks>
    private void ApplyPreviousAttempts(ProgramPromptContext promptContext, string parentGenomeId)
    {
        int limit = _variationOptions.MaxPreviousAttempts;
        if (limit <= 0) return;

        ProgramProposalAttempt[] recorded;
        lock (_attemptLock)
        {
            recorded = _attempts.ToArray();
        }

        var recent = new List<ProgramPromptAttempt>();
        for (int i = recorded.Length - 1; i >= 0 && recent.Count < limit; i--)
        {
            ProgramProposalAttempt attempt = recorded[i];
            if (!string.Equals(attempt.ParentGenomeId, parentGenomeId, StringComparison.Ordinal)) continue;

            recent.Add(new ProgramPromptAttempt(
                attempt.AttemptNumber,
                attempt.Outcome + ": " + ProgramText.Bound(attempt.Detail, 200)));
        }

        // Reversed so the oldest attempt reads first, which is the order a person would recount them in.
        if (recent.Count == 0) return;
        recent.Reverse();
        promptContext.PreviousAttempts = recent;
    }

    /// <summary>Adds the two prompt sections that describe the frontier rather than the parent.</summary>
    /// <param name="promptContext">The context being built.</param>
    /// <param name="context">The variation context, whose archive view may be absent.</param>
    /// <remarks>
    /// Both sections are skipped when no archive view was supplied, which is the case for any caller that builds a
    /// variation context by hand, so an operator never depends on the engine having handed one over.
    /// </remarks>
    private void ApplyArchiveContext(
        ProgramPromptContext promptContext,
        EvolutionVariationContext<ProgramGenome> context)
    {
        if (context.Archive is not { } archive) return;

        int topCount = _variationOptions.MaxTopPrograms;
        if (topCount > 0)
        {
            bool maximize = archive.Direction == EvolutionOptimizationDirection.Maximize;
            var ranked = new List<EvolutionArchiveEntry<ProgramGenome>>();
            foreach (EvolutionArchiveEntry<ProgramGenome> entry in archive.Entries)
            {
                if (entry.Evaluation.Quality.HasValue) ranked.Add(entry);
            }

            ranked.Sort((left, right) =>
            {
                double a = left.Evaluation.Quality ?? 0d;
                double b = right.Evaluation.Quality ?? 0d;
                int byQuality = maximize ? b.CompareTo(a) : a.CompareTo(b);
                return byQuality != 0
                    ? byQuality
                    : string.CompareOrdinal(left.Evaluation.GenomeId, right.Evaluation.GenomeId);
            });

            if (ranked.Count > 0)
            {
                var top = new List<ProgramPromptExample>();
                for (int i = 0; i < ranked.Count && i < topCount; i++)
                {
                    EvolutionArchiveEntry<ProgramGenome> entry = ranked[i];
                    top.Add(new ProgramPromptExample(
                        entry.Candidate.CanonicalGenome.Genome,
                        ProgramPromptExampleKind.TopProgram,
                        entry.Evaluation.Quality,
                        entry.Evaluation.Descriptors));
                }

                promptContext.TopPrograms = top;
            }
        }

        int neighbourCount = _variationOptions.MaxEmptyNeighborCells;
        if (neighbourCount <= 0) return;

        // Cells one bin away from the parent along a single axis that nothing has reached yet. Naming them is what
        // lets a model aim at a gap instead of drifting back toward the crowded middle of the archive.
        IReadOnlyList<int> parentBins = context.Parent.Cell.Bins;
        IReadOnlyList<EvolutionDescriptorDefinition> descriptors = archive.Descriptors;
        if (parentBins.Count != descriptors.Count) return;

        var empty = new List<string>();
        for (int axis = 0; axis < parentBins.Count && empty.Count < neighbourCount; axis++)
        {
            foreach (int step in NeighbourSteps)
            {
                int candidate = parentBins[axis] + step;
                if (candidate < 0 || candidate >= descriptors[axis].BinCount) continue;

                var bins = new int[parentBins.Count];
                for (int i = 0; i < bins.Length; i++) bins[i] = parentBins[i];
                bins[axis] = candidate;

                if (archive.Get(new EvolutionCellKey(bins)) is not null) continue;

                empty.Add(descriptors[axis].Name + "=" + candidate.ToString(CultureInfo.InvariantCulture));
                if (empty.Count >= neighbourCount) break;
            }
        }

        if (empty.Count > 0) promptContext.EmptyNeighborCells = empty;
    }

    private static readonly int[] NeighbourSteps = { -1, 1 };

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

    private string BuildProposalId(EvolutionVariationContext<ProgramGenome> context)
    {
        // Derived from the proposal-local random stream's untouched starting state, so replaying a run produces
        // the same identifiers and two concurrent proposals from the same parent never collide. Reading the state
        // does not consume it, so prompt rendering stays byte-identical whether or not provenance is on.
        StableRandomState state = context.Random.CaptureState();
        return EvolutionHash.Combine(new[]
        {
            Id,
            context.Parent.Candidate.EvaluationId.ToString(CultureInfo.InvariantCulture),
            context.Generation.ToString(CultureInfo.InvariantCulture),
            context.Island.ToString(CultureInfo.InvariantCulture),
            state.State.ToString(CultureInfo.InvariantCulture),
            state.Increment.ToString(CultureInfo.InvariantCulture)
        }).Substring(0, 32);
    }

    private ProposalProvenanceRecord BuildProvenanceRecord(
        EvolutionVariationContext<ProgramGenome> context,
        string proposalId,
        int attemptNumber,
        ProgramProposalOutcome outcome,
        string detail,
        string modelId,
        ProgramPromptResult prompt,
        string promptText,
        string promptHash,
        string responseText,
        string childGenomeId,
        int inputTokens,
        int outputTokens,
        DateTimeOffset requestedAt,
        double latencyMilliseconds)
    {
        string storedPrompt = string.Empty;
        bool promptTruncated = false;
        if (_provenanceOptions.IncludePromptText && promptText.Length > 0)
        {
            storedPrompt = PromptTextRedactor.RedactAndBound(
                promptText, _provenanceOptions.MaxPromptBytes, string.Empty, out promptTruncated);
        }

        string storedResponse = string.Empty;
        bool responseTruncated = false;
        if (_provenanceOptions.IncludeResponseText && responseText.Length > 0)
        {
            storedResponse = PromptTextRedactor.RedactAndBound(
                responseText, _provenanceOptions.MaxResponseBytes, string.Empty, out responseTruncated);
        }

        return new ProposalProvenanceRecord(
            proposalId,
            context.Parent.Candidate.EvaluationId,
            context.Parent.Candidate.CanonicalGenome.Id,
            attemptNumber,
            outcome)
        {
            OperatorId = Id,
            OperatorVersionHash = VersionHash,
            ParentIds = context.Parent.Candidate.Lineage.ParentIds,
            InspirationIds = CollectInspirationIds(context),
            ChildGenomeId = childGenomeId,
            Generation = context.Generation,
            Island = context.Island,
            ModelId = modelId,
            PromptTemplateKey = prompt.UserTemplateKey.ToString(),
            PromptHash = promptHash,
            PromptText = storedPrompt,
            PromptTruncated = promptTruncated,
            ResponseText = storedResponse,
            ResponseTruncated = responseTruncated,
            InputTokens = inputTokens,
            OutputTokens = outputTokens,
            RequestedAtUtc = requestedAt,
            LatencyMilliseconds = latencyMilliseconds,
            Detail = detail
        };
    }

    private async Task RecordProvenanceAsync(ProposalProvenanceRecord record, CancellationToken cancellationToken)
    {
        IProposalProvenanceSink? sink = _provenanceSink;
        if (sink is null) return;
        if (!_provenanceOptions.RecordFailedAttempts && record.Outcome != ProgramProposalOutcome.Accepted) return;

        try
        {
            await sink.RecordAsync(record, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
#pragma warning disable CA1031
        catch (Exception)
#pragma warning restore CA1031
        {
            // A search must not end because a note could not be filed. The gap is counted instead, so a lineage
            // rebuilt from an incomplete stream can be recognised as incomplete.
            Interlocked.Increment(ref _provenanceFailures);
        }
    }

    private static IReadOnlyList<string> CollectInspirationIds(EvolutionVariationContext<ProgramGenome> context)
    {
        if (context.Inspirations.Count == 0) return Array.Empty<string>();

        var ids = new List<string>(context.Inspirations.Count);
        foreach (EvolutionArchiveEntry<ProgramGenome> inspiration in context.Inspirations)
        {
            ids.Add(inspiration.Candidate.CanonicalGenome.Id);
        }

        return ids;
    }

    private static double Elapsed(Stopwatch? timer) => timer is null ? 0.0 : timer.Elapsed.TotalMilliseconds;

    private static string RenderConversation(IReadOnlyList<ChatMessage> messages)
    {
        var builder = new StringBuilder();
        foreach (ChatMessage message in messages)
        {
            builder.Append("### ").Append(message.Role.ToString().ToUpperInvariant()).Append('\n');
            builder.Append(message.Text).Append('\n');
        }

        return builder.ToString();
    }

    private static string HashConversation(IReadOnlyList<ChatMessage> messages)
    {
        // Over the untruncated conversation, so two records whose stored prompts were both clipped can still be
        // proven to have sent the same request.
        var components = new List<string>(messages.Count * 2);
        foreach (ChatMessage message in messages)
        {
            components.Add(message.Role.ToString());
            components.Add(message.Text);
        }

        return EvolutionHash.Combine(components);
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
