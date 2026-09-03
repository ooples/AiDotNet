using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>Configures how a language model is prompted to propose the next candidate program.</summary>
/// <remarks>
/// <para>
/// Everything here is opt-in: the core library gains no model, network, or sandbox dependency, and these options
/// only take effect once a caller supplies a chat client of their own. The two settings that matter most are
/// <see cref="Mode"/>, which decides whether the model is asked for edits or for a whole file, and
/// <see cref="MaxProposalRetries"/>, which decides how many times an unusable answer is sent back with a
/// description of what went wrong. The reference OpenEvolve worker has no equivalent of the second: a response
/// whose edits do not apply produces no child, the iteration is lost, and nothing is fed back to the model.
/// </para>
/// <para>
/// The remaining settings exist to bound what an untrusted response can cost. <see cref="MaxPromptProgramChars"/>
/// caps how much program text is quoted into a prompt, <see cref="MaxInspirations"/> caps how many sibling
/// programs are shown, and <see cref="MaxOutputTokens"/> caps the answer. Prompts and diagnostics never carry API
/// keys, provider URLs, or raw exception text.
/// </para>
/// <para><b>For Beginners:</b> These settings control the conversation with the language model that writes your
/// next candidate program: what you show it, how much you let it write, and how many times you let it correct
/// itself when its answer cannot be used. The defaults ask for small edits, show up to three other good programs
/// for inspiration, and allow two retries with feedback. Raise <see cref="Temperature"/> for more adventurous
/// proposals and lower it for conservative ones.</para>
/// </remarks>
public sealed class LlmProgramVariationOptions
{
    /// <summary>Gets or sets whether the model is asked for edit blocks or a full rewrite.</summary>
    public ProgramEvolutionMode Mode { get; set; } = ProgramEvolutionMode.Diff;

    /// <summary>Gets or sets how many times an unusable response is retried with feedback describing the problem.</summary>
    public int MaxProposalRetries { get; set; } = 2;

    /// <summary>Gets or sets how many answers are drawn from one prompt before the prompt changes. Defaults to 1.</summary>
    /// <remarks>
    /// <para>
    /// An attempt is a prompt; a sample is one answer to it. With more than one sample, an unusable answer is
    /// followed by another draw from the same prompt rather than by a rewritten prompt carrying feedback about the
    /// failure. That is usually the better trade: a model that answered unusably once often answers usably on the
    /// next draw, and feedback about the first answer costs tokens on every later call and biases what comes back.
    /// Feedback is still added once every sample of an attempt has failed.
    /// </para>
    /// <para>
    /// The total number of calls a proposal may make is this multiplied by <see cref="MaxProposalRetries"/> plus one,
    /// so raising both multiplies the cost rather than adding to it.
    /// </para>
    /// <para><b>For Beginners:</b> Leave this at 1 unless the model frequently answers in a form the parser rejects.
    /// Raising it to 2 or 3 asks the same question again before explaining what went wrong.</para>
    /// </remarks>
    public int SamplesPerAttempt { get; set; } = 1;

    /// <summary>Gets or sets the maximum number of inspiration programs quoted into the prompt.</summary>
    public int MaxInspirations { get; set; } = 3;

    /// <summary>Gets or sets the maximum number of characters of any single program quoted into the prompt.</summary>
    public int MaxPromptProgramChars { get; set; } = 8_000;

    /// <summary>Gets or sets a replacement system message, or <c>null</c> to use the built-in instructions.</summary>
    public string? SystemMessage { get; set; }

    /// <summary>Gets or sets the sampling temperature passed to the chat client, or <c>null</c> for its default.</summary>
    public double? Temperature { get; set; }

    /// <summary>Gets or sets the output token cap passed to the chat client, or <c>null</c> for its default.</summary>
    public int? MaxOutputTokens { get; set; }

    /// <summary>Gets or sets a fixed sampling seed, or <c>null</c> to derive one from the proposal's random stream.</summary>
    /// <remarks>
    /// Leaving this <c>null</c> is the reproducible choice: the seed is then derived from the engine's
    /// deterministic per-proposal stream, so an identical run asks the model with an identical seed.
    /// </remarks>
    public int? Seed { get; set; }

    /// <summary>Gets or sets whether the parent's quality and descriptor values are included in the prompt.</summary>
    public bool IncludeParentMetrics { get; set; } = true;

    /// <summary>Gets or sets the archive's behaviour-dimension names, in the archive's own order.</summary>
    /// <remarks>
    /// The engine hands a variation operator the parent's cell as bare bin indices with no names attached, so a
    /// prompt can only say "cell 3,7" unless the names are supplied here. Set this to the archive's descriptor names
    /// in the order the archive declares them, and the prompt reads "length: bin 3 of 10" instead. When the count
    /// does not match the parent's cell the names are still shown but the indices are omitted, so a stale
    /// configuration degrades to less detail rather than to a wrong claim.
    /// </remarks>
    public IList<string> FeatureDimensions { get; set; } = new List<string>();

    /// <summary>Gets or sets the number of bins per dimension, aligned with <see cref="FeatureDimensions"/>.</summary>
    /// <remarks>Leave empty to omit the "of N" part; a mismatched count is ignored rather than rendered wrongly.</remarks>
    public IList<int> FeatureBinCounts { get; set; } = new List<int>();

    /// <summary>Gets or sets the diagnostic-code prefix whose entries are shown as evaluator artifacts.</summary>
    /// <remarks>
    /// Sandboxed evaluators report captured output as diagnostics; those whose code starts with this prefix are
    /// rendered as artifacts, which the prompt bounds and redacts separately from ordinary diagnostics, and they are
    /// not repeated in the diagnostics section. Set it to an empty string to treat every diagnostic as a diagnostic.
    /// </remarks>
    public string ArtifactDiagnosticPrefix { get; set; } = DefaultArtifactDiagnosticPrefix;

    /// <summary>Gets or sets how many recent proposal attempts are retained for inspection after a run.</summary>
    /// <remarks>
    /// The log is a ring buffer, so this bounds memory rather than truncating a run. Set it to zero to record only
    /// the aggregate counters, which is the right choice when proposals number in the millions.
    /// </remarks>
    public int MaxRecordedAttempts { get; set; } = 64;

    /// <summary>The default diagnostic-code prefix treated as evaluator artifacts.</summary>
    public const string DefaultArtifactDiagnosticPrefix = "program_script_artifact";

    /// <summary>Gets or sets how many of the archive's strongest programs are quoted into the prompt; 0 quotes none.</summary>
    /// <remarks>
    /// <para>
    /// These are the leaders of the whole island, chosen by quality, where <see cref="MaxInspirations"/> quotes what
    /// the selection policy picked, which is deliberately a mix of strong and distant candidates. Quoting both gives
    /// the model something to beat as well as something to vary from. Requires the engine to have supplied an
    /// archive view; a hand-built variation context simply omits the section.
    /// </para>
    /// <para><b>For Beginners:</b> This shows the model the current best answers so it can try to beat them.</para>
    /// </remarks>
    public int MaxTopPrograms { get; set; } = 3;

    /// <summary>Gets or sets how many empty neighbouring archive cells are named in the prompt; 0 names none.</summary>
    /// <remarks>
    /// <para>
    /// A neighbour is one bin away from the parent along a single behaviour axis with nothing in it yet. Naming a
    /// few lets the model aim at an unexplored region rather than drifting back toward the crowded middle of the
    /// archive, which is the difference between illuminating a space and hill-climbing in it.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the model which nearby gaps on the map nobody has filled, so it can
    /// try to reach one.</para>
    /// </remarks>
    public int MaxEmptyNeighborCells { get; set; } = 3;

    /// <summary>Gets or sets how many earlier attempts on the same parent are recounted in the prompt; 0 recounts none.</summary>
    /// <remarks>
    /// <para>
    /// Within one proposal a rejected answer is fed straight back into the conversation. Across proposals the model
    /// starts fresh from the same parent and will happily repeat an edit that already failed to parse or matched
    /// nothing, which costs a call and a retry each time. Recounting the last few outcomes for that parent is what
    /// stops the same dead end being paid for twice. Bounded by <see cref="MaxRecordedAttempts"/>, since only
    /// recorded attempts can be recounted.
    /// </para>
    /// <para><b>For Beginners:</b> This reminds the model what it already tried on this program and how it went.</para>
    /// </remarks>
    public int MaxPreviousAttempts { get; set; } = 3;

    /// <summary>Creates an independent copy so a running operator is unaffected by later mutation.</summary>
    /// <returns>A new options instance carrying the same values.</returns>
    public LlmProgramVariationOptions Clone() => new()
    {
        Mode = Mode,
        MaxProposalRetries = MaxProposalRetries,
        SamplesPerAttempt = SamplesPerAttempt,
        MaxInspirations = MaxInspirations,
        MaxTopPrograms = MaxTopPrograms,
        MaxEmptyNeighborCells = MaxEmptyNeighborCells,
        MaxPreviousAttempts = MaxPreviousAttempts,
        MaxPromptProgramChars = MaxPromptProgramChars,
        SystemMessage = SystemMessage,
        Temperature = Temperature,
        MaxOutputTokens = MaxOutputTokens,
        Seed = Seed,
        IncludeParentMetrics = IncludeParentMetrics,
        FeatureDimensions = FeatureDimensions is null ? new List<string>() : new List<string>(FeatureDimensions),
        FeatureBinCounts = FeatureBinCounts is null ? new List<int>() : new List<int>(FeatureBinCounts),
        ArtifactDiagnosticPrefix = ArtifactDiagnosticPrefix,
        MaxRecordedAttempts = MaxRecordedAttempts
    };

    /// <summary>Validates the mode, retry count, and prompt bounds.</summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <see cref="Mode"/> is undefined, a count or bound is out of range, or <see cref="Temperature"/> is not a
    /// finite value between 0 and 2.
    /// </exception>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ProgramEvolutionMode), Mode))
            throw new ArgumentOutOfRangeException(nameof(Mode), Mode, "Value must be a defined mode.");
        if (MaxProposalRetries < 0 || MaxProposalRetries > 16)
            throw new ArgumentOutOfRangeException(nameof(MaxProposalRetries), MaxProposalRetries,
                "Value must be between 0 and 16.");
        if (SamplesPerAttempt < 1 || SamplesPerAttempt > 16)
            throw new ArgumentOutOfRangeException(nameof(SamplesPerAttempt), SamplesPerAttempt,
                "Value must be between 1 and 16.");
        if (MaxInspirations < 0 || MaxInspirations > 32)
            throw new ArgumentOutOfRangeException(nameof(MaxInspirations), MaxInspirations,
                "Value must be between 0 and 32.");
        if (MaxPromptProgramChars < 256)
            throw new ArgumentOutOfRangeException(nameof(MaxPromptProgramChars), MaxPromptProgramChars,
                "Value must be at least 256.");
        if (MaxOutputTokens.HasValue && MaxOutputTokens.Value <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxOutputTokens), MaxOutputTokens.Value,
                "Value must be positive.");
        if (Temperature.HasValue
            && (double.IsNaN(Temperature.Value) || double.IsInfinity(Temperature.Value)
                || Temperature.Value < 0 || Temperature.Value > 2))
        {
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature.Value,
                "Value must be a finite number between 0 and 2.");
        }

        if (MaxRecordedAttempts < 0 || MaxRecordedAttempts > 100_000)
            throw new ArgumentOutOfRangeException(nameof(MaxRecordedAttempts), MaxRecordedAttempts,
                "Value must be between 0 and 100000.");
        if (ArtifactDiagnosticPrefix is null)
            throw new ArgumentException("ArtifactDiagnosticPrefix cannot be null; use an empty string to disable it.",
                nameof(ArtifactDiagnosticPrefix));
        if (FeatureDimensions is null)
            throw new ArgumentException("FeatureDimensions cannot be null.", nameof(FeatureDimensions));
        if (FeatureBinCounts is null)
            throw new ArgumentException("FeatureBinCounts cannot be null.", nameof(FeatureBinCounts));

        foreach (string dimension in FeatureDimensions)
        {
            if (dimension is null || dimension.Trim().Length == 0)
                throw new ArgumentException("A feature-dimension name cannot be empty or white space.",
                    nameof(FeatureDimensions));
        }

        foreach (int binCount in FeatureBinCounts)
        {
            if (binCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(FeatureBinCounts), binCount, "A bin count must be positive.");
        }
    }
}
