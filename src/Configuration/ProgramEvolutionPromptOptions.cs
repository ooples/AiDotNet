using AiDotNet.Enums;
using AiDotNet.Evolution.Prompts;

namespace AiDotNet.Configuration;

/// <summary>Controls what a program-evolution prompt says, what it shows, and how large it is allowed to get.</summary>
/// <remarks>
/// <para>
/// These options cover the whole prompt surface: which wording is used, how many example programs and past
/// attempts are quoted, whether execution output is included and how much of it, whether the model is asked for
/// edits or a rewrite, and the hard ceilings that stop an untrusted response or a runaway archive from producing
/// a prompt nobody intended to pay for. Nothing here starts a model call; the options are inert until a chat
/// client is supplied.
/// </para>
/// <para>
/// Several settings exist because the reference OpenEvolve implementation has no equivalent. It has no ceiling on
/// total prompt size, so a long parent plus five examples silently becomes an expensive request. Its
/// <c>evaluator_system_message</c> is accepted from configuration and then never read, because the controller
/// hard-codes its own. Its system message is treated as a template name when it happens to collide with one and
/// as literal text otherwise, so a typo is sent to the model verbatim. Its diff-versus-rewrite choice is a boolean
/// fixed for the whole run. Each of those is addressed here by <see cref="MaxPromptChars"/>,
/// <see cref="EvaluatorSystemMessage"/>, <see cref="SystemMessageMode"/>, and
/// <see cref="ProgramPromptEvolutionMode.AutoBySize"/> respectively.
/// </para>
/// <para><b>For Beginners:</b> This is the control panel for the message your search sends to the AI. The
/// defaults are a good starting point: three high-scoring examples, two deliberately different ones, the last
/// three attempts, and any output the program produced when it last ran. Turn sections off to make prompts
/// cheaper, raise the counts to give the model more context, or replace any wording through
/// <see cref="TemplateOverrides"/> or a folder of text files named by <see cref="TemplateDirectory"/>. Call
/// <see cref="Validate"/> (the prompt builder does this for you) to have every setting checked before a run
/// starts.</para>
/// </remarks>
public sealed class ProgramEvolutionPromptOptions
{
    /// <summary>Gets or sets a directory of <c>&lt;stem&gt;.txt</c> template overrides, or <c>null</c> for none.</summary>
    /// <remarks>Read as UTF-8 and layered over the shipped defaults; a missing directory is an error, not a silent fallback.</remarks>
    public string? TemplateDirectory { get; set; }

    /// <summary>Gets or sets the system message, interpreted according to <see cref="SystemMessageMode"/>.</summary>
    /// <remarks><c>null</c> uses <see cref="ProgramPromptTemplateKey.SystemMessage"/>.</remarks>
    public string? SystemMessage { get; set; }

    /// <summary>Gets or sets whether <see cref="SystemMessage"/> names a template or is the text itself.</summary>
    public ProgramPromptSystemMessageMode SystemMessageMode { get; set; } = ProgramPromptSystemMessageMode.TemplateKey;

    /// <summary>Gets or sets the system message used when a model judges a program, or <c>null</c> for the default template.</summary>
    /// <remarks>Unlike upstream, this value is honoured rather than accepted and ignored.</remarks>
    public string? EvaluatorSystemMessage { get; set; }

    /// <summary>Gets or sets extra instructions appended to the system message, or <c>null</c> for none.</summary>
    public string? ExtraSystemText { get; set; }

    /// <summary>Gets or sets a description of the task the program must accomplish, or <c>null</c> for none.</summary>
    public string? TaskDescription { get; set; }

    /// <summary>Gets or sets whether the model is asked for edits, a rewrite, or either depending on program size.</summary>
    /// <remarks>
    /// <c>LlmProgramVariationOptions.Mode</c> wins over this, because the operator that parses the answer is the one
    /// that has to agree with the prompt that asked for it. The single exception is
    /// <see cref="ProgramPromptEvolutionMode.AutoBySize"/>, which the operator leaves alone so the choice can be made
    /// per candidate. Set the variation mode when you want to choose, and set this one only to select automatic
    /// sizing.
    /// </remarks>
    public ProgramPromptEvolutionMode EvolutionMode { get; set; } = ProgramPromptEvolutionMode.Diff;

    /// <summary>Gets or sets whether programs are represented in prompts by their change descriptions.</summary>
    /// <remarks>
    /// Large-codebase mode. Example programs are quoted as prose summaries of what they changed rather than as
    /// source, and the model is required to keep the current summary up to date in the same answer.
    /// </remarks>
    public bool ProgramsAsChangesDescription { get; set; }

    /// <summary>Gets or sets the description given to a program that has none yet; <c>null</c> means none.</summary>
    /// <remarks>
    /// <para>
    /// A seed program has no history, so in changes-description mode the first prompt would otherwise show the model
    /// an empty summary and ask it to edit that, which is both confusing and impossible to write a SEARCH block
    /// against. Supplying a line of starting text gives the first generation something real to replace.
    /// </para>
    /// <para>
    /// It is used only where a description is missing, so it never overwrites one a run has already built up, and
    /// only in changes-description mode, so it costs nothing elsewhere.
    /// </para>
    /// <para><b>For Beginners:</b> Something like <c>"Initial version."</c>. It gives the very first prompt a
    /// sensible starting note for the model to rewrite.</para>
    /// </remarks>
    public string? InitialChangesDescription { get; set; }

    /// <summary>Gets or sets the parent length below which <see cref="ProgramPromptEvolutionMode.AutoBySize"/> asks for a rewrite.</summary>
    public int AutoFullRewriteBelowChars { get; set; } = 2_000;

    /// <summary>Gets or sets how many high-scoring programs are quoted as examples.</summary>
    public int NumTopPrograms { get; set; } = 3;

    /// <summary>Gets or sets how many additional programs are quoted for their difference rather than their score.</summary>
    public int NumDiversePrograms { get; set; } = 2;

    /// <summary>Gets or sets how many earlier attempts are summarized.</summary>
    public int NumPreviousAttempts { get; set; } = 3;

    /// <summary>Gets or sets whether the inspirations section is rendered.</summary>
    public bool IncludeInspirations { get; set; } = true;

    /// <summary>Gets or sets whether the previous-attempts section is rendered.</summary>
    public bool IncludePreviousAttempts { get; set; } = true;

    /// <summary>Gets or sets whether execution output from the parent's evaluation is quoted.</summary>
    /// <remarks>
    /// This is permission rather than a demand: it lets the prompt carry artifacts, and the engine decides whether
    /// there are any to carry. Artifact capture is off by default (<c>EvolutionEngineOptions.Artifacts.Enabled</c>),
    /// so leaving this on costs nothing and changes nothing until capture is enabled as well. If a prompt has no
    /// artifact section when you expected one, that engine option is the setting to check.
    /// </remarks>
    public bool IncludeArtifacts { get; set; } = true;

    /// <summary>Gets or sets the maximum UTF-8 bytes of a single artifact that are quoted.</summary>
    public int MaxArtifactBytes { get; set; } = 20 * 1024;

    /// <summary>Gets or sets the maximum number of artifacts quoted.</summary>
    public int MaxArtifactCount { get; set; } = 8;

    /// <summary>Gets or sets whether artifact text is scrubbed of control sequences and credential-shaped values.</summary>
    /// <remarks>Leave this on. Artifacts are the output of untrusted generated code.</remarks>
    public bool ArtifactSecurityFilter { get; set; } = true;

    /// <summary>Gets or sets whether the parent's descriptor coordinates are shown.</summary>
    public bool IncludeFeatureCoordinates { get; set; } = true;

    /// <summary>Gets or sets whether bounded evaluation diagnostics are shown.</summary>
    public bool IncludeDiagnostics { get; set; } = true;

    /// <summary>Gets or sets the maximum number of diagnostics shown.</summary>
    public int MaxDiagnostics { get; set; } = 5;

    /// <summary>Gets or sets whether nearby unoccupied archive cells are suggested as targets.</summary>
    public bool IncludeCoverageHints { get; set; }

    /// <summary>Gets or sets whether a placeholder in the user template may be filled from <see cref="TemplateVariations"/>.</summary>
    public bool UseTemplateStochasticity { get; set; } = true;

    /// <summary>Gets or sets the alternative wordings each variation placeholder may be filled with.</summary>
    /// <remarks>
    /// The choice is drawn from the proposal's deterministic random stream, so two runs with the same engine seed
    /// choose the same wording in the same order and a resumed run continues the sequence. Upstream draws from the
    /// process-global generator, which no seed controls.
    /// </remarks>
    public IDictionary<string, IReadOnlyList<string>> TemplateVariations { get; set; } =
        new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);

    /// <summary>Gets or sets extra named values any template may reference.</summary>
    public IDictionary<string, string> CustomVariables { get; set; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Gets or sets replacement text for individual templates.</summary>
    public IDictionary<ProgramPromptTemplateKey, string> TemplateOverrides { get; set; } =
        new Dictionary<ProgramPromptTemplateKey, string>();

    /// <summary>Gets or sets replacement text for individual phrases.</summary>
    public IDictionary<ProgramPromptFragmentKey, string> FragmentOverrides { get; set; } =
        new Dictionary<ProgramPromptFragmentKey, string>();

    /// <summary>Gets or sets the program length past which simplification is suggested, or <c>null</c> to never suggest it.</summary>
    public int? SuggestSimplificationAfterChars { get; set; } = 500;

    /// <summary>Gets or sets the absolute fitness difference treated as unchanged.</summary>
    public double FitnessStableBand { get; set; } = 1e-6;

    /// <summary>Gets or sets the length under which an example's change description is quoted, or <c>null</c> to never quote it.</summary>
    public int? IncludeChangesUnderChars { get; set; } = 100;

    /// <summary>Gets or sets the line count at or below which an example is labelled concise, or <c>null</c> to skip the label.</summary>
    public int? ConciseImplementationMaxLines { get; set; } = 10;

    /// <summary>Gets or sets the line count at or above which an example is labelled comprehensive, or <c>null</c> to skip the label.</summary>
    public int? ComprehensiveImplementationMinLines { get; set; } = 50;

    /// <summary>Gets or sets the maximum characters of any one program quoted into the prompt.</summary>
    public int MaxProgramSnippetChars { get; set; } = 8_000;

    /// <summary>Gets or sets the maximum characters of the whole rendered user message.</summary>
    /// <remarks>
    /// The last hard stop before a request is sent. Upstream has no equivalent, so an unusually long parent plus a
    /// full complement of examples becomes an expensive call nobody chose to make.
    /// </remarks>
    public int MaxPromptChars { get; set; } = 200_000;

    /// <summary>Gets or sets the number of decimal places used when a score is rendered.</summary>
    public int ScoreDecimals { get; set; } = 4;

    /// <summary>Creates an independent copy so a running builder is unaffected by later mutation.</summary>
    /// <returns>A new options instance carrying the same values and copied collections.</returns>
    public ProgramEvolutionPromptOptions Clone()
    {
        var variations = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);
        if (TemplateVariations is not null)
        {
            foreach (KeyValuePair<string, IReadOnlyList<string>> pair in TemplateVariations)
            {
                variations[pair.Key] = pair.Value is null
                    ? new List<string>()
                    : new List<string>(pair.Value);
            }
        }

        var custom = new Dictionary<string, string>(StringComparer.Ordinal);
        if (CustomVariables is not null)
        {
            foreach (KeyValuePair<string, string> pair in CustomVariables) custom[pair.Key] = pair.Value;
        }

        var templates = new Dictionary<ProgramPromptTemplateKey, string>();
        if (TemplateOverrides is not null)
        {
            foreach (KeyValuePair<ProgramPromptTemplateKey, string> pair in TemplateOverrides) templates[pair.Key] = pair.Value;
        }

        var fragments = new Dictionary<ProgramPromptFragmentKey, string>();
        if (FragmentOverrides is not null)
        {
            foreach (KeyValuePair<ProgramPromptFragmentKey, string> pair in FragmentOverrides) fragments[pair.Key] = pair.Value;
        }

        return new ProgramEvolutionPromptOptions
        {
            TemplateDirectory = TemplateDirectory,
            SystemMessage = SystemMessage,
            SystemMessageMode = SystemMessageMode,
            EvaluatorSystemMessage = EvaluatorSystemMessage,
            ExtraSystemText = ExtraSystemText,
            TaskDescription = TaskDescription,
            EvolutionMode = EvolutionMode,
            ProgramsAsChangesDescription = ProgramsAsChangesDescription,
            InitialChangesDescription = InitialChangesDescription,
            AutoFullRewriteBelowChars = AutoFullRewriteBelowChars,
            NumTopPrograms = NumTopPrograms,
            NumDiversePrograms = NumDiversePrograms,
            NumPreviousAttempts = NumPreviousAttempts,
            IncludeInspirations = IncludeInspirations,
            IncludePreviousAttempts = IncludePreviousAttempts,
            IncludeArtifacts = IncludeArtifacts,
            MaxArtifactBytes = MaxArtifactBytes,
            MaxArtifactCount = MaxArtifactCount,
            ArtifactSecurityFilter = ArtifactSecurityFilter,
            IncludeFeatureCoordinates = IncludeFeatureCoordinates,
            IncludeDiagnostics = IncludeDiagnostics,
            MaxDiagnostics = MaxDiagnostics,
            IncludeCoverageHints = IncludeCoverageHints,
            UseTemplateStochasticity = UseTemplateStochasticity,
            TemplateVariations = variations,
            CustomVariables = custom,
            TemplateOverrides = templates,
            FragmentOverrides = fragments,
            SuggestSimplificationAfterChars = SuggestSimplificationAfterChars,
            FitnessStableBand = FitnessStableBand,
            IncludeChangesUnderChars = IncludeChangesUnderChars,
            ConciseImplementationMaxLines = ConciseImplementationMaxLines,
            ComprehensiveImplementationMinLines = ComprehensiveImplementationMinLines,
            MaxProgramSnippetChars = MaxProgramSnippetChars,
            MaxPromptChars = MaxPromptChars,
            ScoreDecimals = ScoreDecimals
        };
    }

    /// <summary>Validates every count, bound, and collection entry.</summary>
    /// <exception cref="ArgumentException">A collection is <c>null</c>, or a variation or custom-variable name is unusable.</exception>
    /// <exception cref="ArgumentOutOfRangeException">An enumeration value is undefined or a numeric value is out of range.</exception>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ProgramPromptSystemMessageMode), SystemMessageMode))
            throw new ArgumentOutOfRangeException(nameof(SystemMessageMode), SystemMessageMode, "Value must be a defined mode.");
        if (!Enum.IsDefined(typeof(ProgramPromptEvolutionMode), EvolutionMode))
            throw new ArgumentOutOfRangeException(nameof(EvolutionMode), EvolutionMode, "Value must be a defined mode.");

        RequireRange(AutoFullRewriteBelowChars, 0, 1_000_000, nameof(AutoFullRewriteBelowChars));
        RequireRange(NumTopPrograms, 0, 64, nameof(NumTopPrograms));
        RequireRange(NumDiversePrograms, 0, 64, nameof(NumDiversePrograms));
        RequireRange(NumPreviousAttempts, 0, 64, nameof(NumPreviousAttempts));
        RequireRange(MaxArtifactBytes, 0, 8 * 1024 * 1024, nameof(MaxArtifactBytes));
        RequireRange(MaxArtifactCount, 0, 256, nameof(MaxArtifactCount));
        RequireRange(MaxDiagnostics, 0, 64, nameof(MaxDiagnostics));
        RequireRange(MaxProgramSnippetChars, 64, 1_048_576, nameof(MaxProgramSnippetChars));
        RequireRange(MaxPromptChars, 256, 8_388_608, nameof(MaxPromptChars));
        RequireRange(ScoreDecimals, 0, 15, nameof(ScoreDecimals));

        if (SuggestSimplificationAfterChars.HasValue)
            RequireRange(SuggestSimplificationAfterChars.Value, 1, 8_388_608, nameof(SuggestSimplificationAfterChars));
        if (IncludeChangesUnderChars.HasValue)
            RequireRange(IncludeChangesUnderChars.Value, 1, 65_536, nameof(IncludeChangesUnderChars));
        if (ConciseImplementationMaxLines.HasValue)
            RequireRange(ConciseImplementationMaxLines.Value, 1, 1_000_000, nameof(ConciseImplementationMaxLines));
        if (ComprehensiveImplementationMinLines.HasValue)
            RequireRange(ComprehensiveImplementationMinLines.Value, 1, 1_000_000, nameof(ComprehensiveImplementationMinLines));

        if (double.IsNaN(FitnessStableBand) || double.IsInfinity(FitnessStableBand) || FitnessStableBand < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(FitnessStableBand), FitnessStableBand,
                "Value must be a finite, non-negative number.");
        }

        if (TemplateVariations is null) throw new ArgumentException("TemplateVariations cannot be null.", nameof(TemplateVariations));
        if (CustomVariables is null) throw new ArgumentException("CustomVariables cannot be null.", nameof(CustomVariables));
        if (TemplateOverrides is null) throw new ArgumentException("TemplateOverrides cannot be null.", nameof(TemplateOverrides));
        if (FragmentOverrides is null) throw new ArgumentException("FragmentOverrides cannot be null.", nameof(FragmentOverrides));

        foreach (KeyValuePair<string, IReadOnlyList<string>> pair in TemplateVariations)
        {
            RequirePlaceholderName(pair.Key, nameof(TemplateVariations));
            if (pair.Value is null || pair.Value.Count == 0)
            {
                throw new ArgumentException(
                    $"The template variation '{pair.Key}' must offer at least one wording.", nameof(TemplateVariations));
            }

            foreach (string variation in pair.Value)
            {
                if (variation is null)
                {
                    throw new ArgumentException(
                        $"The template variation '{pair.Key}' contains a null wording.", nameof(TemplateVariations));
                }
            }
        }

        foreach (KeyValuePair<string, string> pair in CustomVariables)
        {
            RequirePlaceholderName(pair.Key, nameof(CustomVariables));
            if (pair.Value is null)
            {
                throw new ArgumentException($"The custom variable '{pair.Key}' is null.", nameof(CustomVariables));
            }
        }

        foreach (KeyValuePair<ProgramPromptTemplateKey, string> pair in TemplateOverrides)
        {
            if (!Enum.IsDefined(typeof(ProgramPromptTemplateKey), pair.Key))
                throw new ArgumentException($"'{pair.Key}' is not a defined template key.", nameof(TemplateOverrides));
            if (pair.Value is null)
                throw new ArgumentException($"The override for '{pair.Key}' is null.", nameof(TemplateOverrides));
        }

        foreach (KeyValuePair<ProgramPromptFragmentKey, string> pair in FragmentOverrides)
        {
            if (!Enum.IsDefined(typeof(ProgramPromptFragmentKey), pair.Key))
                throw new ArgumentException($"'{pair.Key}' is not a defined fragment key.", nameof(FragmentOverrides));
            if (pair.Value is null)
                throw new ArgumentException($"The override for '{pair.Key}' is null.", nameof(FragmentOverrides));
        }

        if (SystemMessageMode == ProgramPromptSystemMessageMode.TemplateKey
            && SystemMessage is { } named
            && named.Trim().Length > 0
            && !TryParseTemplateKey(named, out _))
        {
            throw new ArgumentException(
                $"SystemMessage is '{named}', which is not a template key. Set SystemMessageMode to Literal to send " +
                "that text to the model, or use one of: " + string.Join(", ", DescribeTemplateStems()) + ".",
                nameof(SystemMessage));
        }
    }

    /// <summary>Builds the validated template set implied by the directory and the per-key overrides.</summary>
    /// <returns>A validated set with the shipped defaults, then the directory, then the overrides applied.</returns>
    /// <exception cref="ArgumentException">An override is malformed or drops a structurally required placeholder.</exception>
    /// <exception cref="DirectoryNotFoundException"><see cref="TemplateDirectory"/> is set but does not exist.</exception>
    /// <exception cref="InvalidDataException">The directory's fragments file is not a JSON object of string values.</exception>
    public ProgramPromptTemplateSet BuildTemplateSet()
    {
        ProgramPromptTemplateSet set = ProgramPromptTemplateSet.CreateDefault();
        if (TemplateDirectory is { } directory && directory.Trim().Length > 0)
        {
            set = ProgramPromptTemplateSet.LoadFromDirectory(directory, set);
        }

        if (TemplateOverrides is not null)
        {
            foreach (KeyValuePair<ProgramPromptTemplateKey, string> pair in TemplateOverrides)
            {
                set = set.With(pair.Key, pair.Value);
            }
        }

        if (FragmentOverrides is not null)
        {
            foreach (KeyValuePair<ProgramPromptFragmentKey, string> pair in FragmentOverrides)
            {
                set = set.WithFragment(pair.Key, pair.Value);
            }
        }

        return set;
    }

    /// <summary>Decides whether a parent of a given size is edited or rewritten.</summary>
    /// <param name="parentLength">The parent program's length in characters.</param>
    /// <returns>
    /// <see cref="ProgramPromptEvolutionMode.Diff"/> or <see cref="ProgramPromptEvolutionMode.FullRewrite"/>,
    /// never <see cref="ProgramPromptEvolutionMode.AutoBySize"/>.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="parentLength"/> is negative.</exception>
    public ProgramPromptEvolutionMode ResolveEvolutionMode(int parentLength)
    {
        if (parentLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(parentLength), parentLength, "Value cannot be negative.");
        }

        if (EvolutionMode != ProgramPromptEvolutionMode.AutoBySize) return EvolutionMode;
        return parentLength < AutoFullRewriteBelowChars
            ? ProgramPromptEvolutionMode.FullRewrite
            : ProgramPromptEvolutionMode.Diff;
    }

    /// <summary>Parses a template key from the file stem used for directory overrides.</summary>
    /// <param name="name">The stem, such as <c>diff_user</c>.</param>
    /// <param name="key">The matching key, when the stem is recognized.</param>
    /// <returns><c>true</c> when <paramref name="name"/> names a template.</returns>
    public static bool TryParseTemplateKey(string? name, out ProgramPromptTemplateKey key)
    {
        key = ProgramPromptTemplateKey.SystemMessage;
        if (name is null) return false;
        string trimmed = name.Trim();
        if (trimmed.Length == 0) return false;

        foreach (ProgramPromptTemplateKey candidate in ProgramPromptTemplateSet.TemplateKeys)
        {
            if (string.Equals(ProgramPromptTemplateSet.TemplateFileStem(candidate), trimmed, StringComparison.OrdinalIgnoreCase)
                || string.Equals(candidate.ToString(), trimmed, StringComparison.OrdinalIgnoreCase))
            {
                key = candidate;
                return true;
            }
        }

        return false;
    }

    private static IEnumerable<string> DescribeTemplateStems()
    {
        foreach (ProgramPromptTemplateKey key in ProgramPromptTemplateSet.TemplateKeys)
        {
            yield return ProgramPromptTemplateSet.TemplateFileStem(key);
        }
    }

    private static void RequireRange(int value, int minimum, int maximum, string name)
    {
        if (value < minimum || value > maximum)
        {
            throw new ArgumentOutOfRangeException(name, value, $"Value must be between {minimum} and {maximum}.");
        }
    }

    private static void RequirePlaceholderName(string name, string parameterName)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("A placeholder name cannot be empty.", parameterName);
        }

        foreach (char character in name)
        {
            if (character == '_' || char.IsLetterOrDigit(character)) continue;
            throw new ArgumentException(
                $"The placeholder name '{name}' may only contain letters, digits, and underscores.", parameterName);
        }
    }
}
