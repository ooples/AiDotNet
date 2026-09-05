using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.Configuration;

/// <summary>The single configuration surface for evolving source code: seeds, bounds, prompting, sandbox, and search.</summary>
/// <remarks>
/// <para>
/// These options gather every knob a program-evolution run needs. The first group describes the program text
/// itself — which comment markers fence the editable region, whether edits outside that region are refused, how
/// large a program may grow, and how SEARCH/REPLACE edits are parsed and applied. The second group ties the rest of
/// the run together: <see cref="SeedPrograms"/> and <see cref="TaskDescription"/> say what is being improved and
/// why, <see cref="TestCases"/> and <see cref="EvaluatorScript"/> say how a candidate is scored,
/// <see cref="Descriptors"/> says which behaviour axes the archive maps, and <see cref="Prompt"/>,
/// <see cref="Variation"/>, <see cref="Sandbox"/>, and <see cref="Engine"/> configure the four subsystems that do
/// the work.
/// </para>
/// <para>
/// Nothing here activates an LLM, a sandbox, a Python interpreter, or a network call. Every subsystem stays inert
/// until a caller supplies a chat client or an execution engine of its own, so a program that merely constructs
/// these options acquires no new dependency. The nested subsystem options are created on first use, which keeps the
/// common case — a bare instance handed to <see cref="AiDotNet.Evolution.Programs.ProgramDiff"/> just to read its
/// markers — as cheap as it was before the aggregate members existed.
/// </para>
/// <para>
/// <see cref="EvolveBlockStartMarker"/> and <see cref="EvolveBlockEndMarker"/> are <c>null</c> by default, which
/// means "use the comment syntax that is valid in <see cref="Language"/>"; set them to override that choice. The
/// reference OpenEvolve implementation hard-codes Python-style markers regardless of language and caps program
/// text at 10,000 characters, whereas <see cref="MaxProgramChars"/> is explicit and applies to every language.
/// </para>
/// <para><b>For Beginners:</b> This is the control panel for evolving source code rather than numbers. Set
/// <see cref="Language"/> to the language you are evolving, put your starting program in
/// <see cref="SeedPrograms"/>, describe the goal in <see cref="TaskDescription"/>, and give the search a way to
/// score a candidate — either a list of input/output <see cref="TestCases"/> or an
/// <see cref="EvaluatorScript"/>. Everything else has a working default. The character limit stops a runaway model
/// from producing a megabyte of code, and <see cref="EnforceEvolveBlocks"/> is worth turning on once your files
/// have start and end markers, because it stops the model from editing the parts you wanted left alone.</para>
/// </remarks>
public sealed class ProgramEvolutionOptions
{
    /// <summary>The default number of elites whose bounded source appears in a run summary.</summary>
    public const int DefaultIncludeEliteSourceCount = 10;

    /// <summary>The default per-elite source bound in a run summary, in characters.</summary>
    public const int DefaultMaxEliteSourceChars = 4_000;

    private ProgramDiffOptions? _diff;
    private ProgramEvolutionPromptOptions? _prompt;
    private LlmProgramVariationOptions? _variation;
    private ProgramSandboxOptions? _sandbox;
    private ScriptProgramEvaluationOptions? _script;
    private LlmFeedbackOptions? _llmFeedback;
    private EvolutionEngineOptions? _engine;
    private ProgramMetricAggregationOptions? _metrics;
    private ProposalProvenanceOptions? _provenance;
    private ProgramArtifactStoreOptions? _artifactStore;
    private ProgramRunOutputOptions? _runOutput;
    private IList<string>? _seedPrograms;
    private IList<ProgramInputOutputExample>? _testCases;
    private IList<IProgramDescriptor>? _descriptors;

    /// <summary>Gets or sets the language of the programs being evolved.</summary>
    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    /// <summary>Gets or sets the start marker that opens the editable region, or <c>null</c> to use the language default.</summary>
    public string? EvolveBlockStartMarker { get; set; }

    /// <summary>Gets or sets the end marker that closes the editable region, or <c>null</c> to use the language default.</summary>
    public string? EvolveBlockEndMarker { get; set; }

    /// <summary>Gets or sets whether edits that fall outside every evolve block are refused.</summary>
    /// <remarks>
    /// When <c>true</c> and the parent program contains at least one evolve block, an edit whose search text
    /// matches only outside those blocks is rejected with
    /// <see cref="AiDotNet.Enums.ProgramDiffFailureReason.OutsideEvolveBlock"/> instead of being applied.
    /// </remarks>
    public bool EnforceEvolveBlocks { get; set; }

    /// <summary>Gets or sets the maximum number of characters a candidate program may contain.</summary>
    public int MaxProgramChars { get; set; } = 100_000;

    /// <summary>Gets or sets how SEARCH/REPLACE edit blocks are parsed and applied.</summary>
    public ProgramDiffOptions Diff
    {
        get => _diff ??= new ProgramDiffOptions();
        set => _diff = value;
    }

    /// <summary>Gets or sets the program sources the run starts from.</summary>
    /// <remarks>
    /// Each entry becomes one seed genome. Supplying several is useful when you already have distinct approaches
    /// worth keeping apart, because the archive will place them in different behaviour cells and evolve each. An
    /// empty list is legal only when the caller supplies seed genomes to the engine directly.
    /// </remarks>
    public IList<string> SeedPrograms
    {
        get => _seedPrograms ??= new List<string>();
        set => _seedPrograms = value;
    }

    /// <summary>Gets or sets the plain-language statement of what the evolved program must achieve.</summary>
    /// <remarks>It is quoted verbatim into every prompt, so state the goal and the constraints, not the method.</remarks>
    public string? TaskDescription { get; set; }

    /// <summary>Gets or sets the input/output examples a candidate is scored against.</summary>
    /// <remarks>
    /// These drive <see cref="AiDotNet.Evolution.Programs.SandboxedProgramFitnessEvaluator"/>: a candidate's score
    /// is the fraction of examples it reproduces. Leave the list empty when scoring through
    /// <see cref="EvaluatorScript"/> or through an evaluator of your own instead.
    /// </remarks>
    public IList<ProgramInputOutputExample> TestCases
    {
        get => _testCases ??= new List<ProgramInputOutputExample>();
        set => _testCases = value;
    }

    /// <summary>Gets or sets the behaviour axes the archive maps candidates onto.</summary>
    /// <remarks>
    /// These decide what "diverse" means for this run. With none, the archive degenerates to a single cell and the
    /// search keeps only the overall best program, losing the illumination that makes MAP-Elites worth running.
    /// </remarks>
    public IList<IProgramDescriptor> Descriptors
    {
        get => _descriptors ??= new List<IProgramDescriptor>();
        set => _descriptors = value;
    }

    /// <summary>Gets or sets the evaluator source that scores candidates inside the sandbox, or <c>null</c> for none.</summary>
    /// <remarks>
    /// The script receives a candidate's source on standard input and prints one JSON object of metrics. It is
    /// itself untrusted code and runs under the same limits as the candidates, unlike the reference implementation,
    /// which executes its evaluator unsandboxed inside its worker processes.
    /// </remarks>
    public string? EvaluatorScript
    {
        get => Script.EvaluatorScript;
        set => Script.EvaluatorScript = value;
    }

    /// <summary>Gets or sets the language <see cref="EvaluatorScript"/> is written in.</summary>
    public ProgramLanguage EvaluatorScriptLanguage
    {
        get => Script.EvaluatorScriptLanguage;
        set => Script.EvaluatorScriptLanguage = value;
    }

    /// <summary>Gets or sets how many elites a run summary retains, best first.</summary>
    public int IncludeEliteSourceCount { get; set; } = DefaultIncludeEliteSourceCount;

    /// <summary>Gets or sets the per-elite source bound applied when building a run summary, in characters.</summary>
    public int MaxEliteSourceChars { get; set; } = DefaultMaxEliteSourceChars;

    /// <summary>Gets or sets how prompts are templated, bounded, and varied.</summary>
    public ProgramEvolutionPromptOptions Prompt
    {
        get => _prompt ??= new ProgramEvolutionPromptOptions();
        set => _prompt = value;
    }

    /// <summary>Gets or sets how the language model is asked for a change and how unusable answers are retried.</summary>
    public LlmProgramVariationOptions Variation
    {
        get => _variation ??= new LlmProgramVariationOptions();
        set => _variation = value;
    }

    /// <summary>Gets or sets the execution boundary and resource limits applied to untrusted program text.</summary>
    public ProgramSandboxOptions Sandbox
    {
        get => _sandbox ??= new ProgramSandboxOptions();
        set => _sandbox = value;
    }

    /// <summary>Gets whether a caller set <see cref="Sandbox"/>, as opposed to only reading its default.</summary>
    /// <remarks>
    /// Reading the property creates the default instance, so "is it non-null" cannot answer this. The builder needs
    /// the real answer to tell a caller that configured the sandbox twice from one that configured it nowhere.
    /// </remarks>
    internal bool HasExplicitSandbox => _sandbox is not null;

    /// <summary>Gets or sets how a caller-supplied evaluator script is run and how its metrics are read.</summary>
    public ScriptProgramEvaluationOptions Script
    {
        get => _script ??= new ScriptProgramEvaluationOptions();
        set => _script = value;
    }

    /// <summary>Gets or sets how a language model's judgement of a candidate is blended into its measured score.</summary>
    /// <remarks>
    /// Inert until a chat client is supplied to
    /// <see cref="AiDotNet.Evolution.Programs.LlmJudgeProgramFitnessEvaluator{T}"/>; configuring it alone contacts
    /// no model.
    /// </remarks>
    public LlmFeedbackOptions LlmFeedback
    {
        get => _llmFeedback ??= new LlmFeedbackOptions();
        set => _llmFeedback = value;
    }

    /// <summary>Gets or sets how an evaluator's metric dictionary is reduced to the single quality the archive ranks.</summary>
    /// <remarks>
    /// <para>
    /// Applied only when an evaluator reports metrics without a quality of its own, so an evaluator that already
    /// returns a quality is never second-guessed. The default reproduces the reference implementation's rule: a
    /// metric literally named <c>combined_score</c> wins, otherwise the mean over the numeric metrics.
    /// </para>
    /// <para><b>For Beginners:</b> Scoring code often reports several numbers, such as accuracy and runtime, but the
    /// archive can only rank one. This decides how that list becomes a single score, and unlike an implicit average
    /// it tells you when a metric could not be read instead of quietly skipping it.</para>
    /// </remarks>
    public ProgramMetricAggregationOptions Metrics
    {
        get => _metrics ??= new ProgramMetricAggregationOptions();
        set => _metrics = value;
    }

    /// <summary>Gets or sets whether each proposal's prompt, answer, and parse outcome are recorded for later audit.</summary>
    /// <remarks>
    /// Inert until <see cref="ProposalProvenanceOptions.Enabled"/> is set. Records are written beneath the run's
    /// output directory, bounded and redacted; no API key is ever written.
    /// </remarks>
    public ProposalProvenanceOptions Provenance
    {
        get => _provenance ??= new ProposalProvenanceOptions();
        set => _provenance = value;
    }

    /// <summary>Gets or sets on-disk retention of evaluation artifacts; <c>null</c> keeps artifacts in memory only.</summary>
    /// <remarks>
    /// <para>
    /// Setting this promotes an artifact larger than the configured inline threshold to a file beneath the run's
    /// output directory and makes it retrievable by genome id after the run, which an in-memory artifact is not.
    /// It requires <see cref="EvolutionEngineOptions.OutputDirectory"/> to be set.
    /// </para>
    /// <para><b>For Beginners:</b> Artifacts are the notes an evaluation leaves behind, such as a compiler error.
    /// By default they live in memory and vanish when the run ends; set this to keep the large ones on disk.</para>
    /// </remarks>
    public ProgramArtifactStoreOptions? ArtifactStore
    {
        get => _artifactStore;
        set => _artifactStore = value;
    }

    /// <summary>Gets or sets writing of the best program to disk; <c>null</c> writes no program files.</summary>
    /// <remarks>
    /// <para>
    /// Setting this writes the best program with the file extension of its language, plus an information document
    /// carrying its metrics, descriptors, cell and lineage, at checkpoints and at run end as configured. It
    /// requires <see cref="EvolutionEngineOptions.OutputDirectory"/> to be set.
    /// </para>
    /// <para><b>For Beginners:</b> Without this a finished run leaves nothing on disk to open. With it you get the
    /// winning program as a real source file you can run.</para>
    /// </remarks>
    public ProgramRunOutputOptions? RunOutput
    {
        get => _runOutput;
        set => _runOutput = value;
    }

    /// <summary>Gets or sets duplicate rejection by structural distance and optionally by embedding; <c>null</c> disables it.</summary>
    /// <remarks>
    /// <para>
    /// The cheap structural rung needs no model and no network. An embedding rung is consulted only for candidates
    /// the structural rung could not settle, and only when an embedding client is supplied to the run.
    /// </para>
    /// <para><b>For Beginners:</b> Language models often propose a program that is effectively one already tried.
    /// This spots those before you pay to evaluate them.</para>
    /// </remarks>
    public EmbeddingNoveltyOptions? Novelty { get; set; }

    /// <summary>Gets or sets the generic search settings: budgets, islands, parallelism, and checkpointing.</summary>
    /// <remarks>
    /// This is the same options object the engine takes, held here so one instance describes an entire run. It is
    /// referenced rather than owned by the engine: the engine snapshots it during construction, so later mutation
    /// cannot change a run in flight.
    /// </remarks>
    public EvolutionEngineOptions Engine
    {
        get => _engine ??= new EvolutionEngineOptions();
        set => _engine = value;
    }

    /// <summary>Creates one seed genome per entry in <see cref="SeedPrograms"/>.</summary>
    /// <returns>The seed genomes, in the order the sources were supplied.</returns>
    /// <exception cref="ArgumentException">A seed source is <c>null</c>, blank, or longer than <see cref="MaxProgramChars"/>.</exception>
    public IReadOnlyList<ProgramGenome> CreateSeedGenomes()
    {
        var genomes = new List<ProgramGenome>();
        foreach (string source in SeedPrograms)
        {
            if (source is not { } text || text.Trim().Length == 0)
            {
                throw new ArgumentException("A seed program cannot be null, empty, or white space.", nameof(SeedPrograms));
            }

            if (ProgramGenome.Normalize(text).Length > MaxProgramChars)
            {
                throw new ArgumentException(
                    $"A seed program exceeds the configured MaxProgramChars limit of {MaxProgramChars}.",
                    nameof(SeedPrograms));
            }

            genomes.Add(new ProgramGenome(text, Language));
        }

        return genomes;
    }

    /// <summary>Creates the descriptor set the task merges into every completed evaluation.</summary>
    /// <returns>A set over <see cref="Descriptors"/>, or an empty set when none were configured.</returns>
    /// <exception cref="ArgumentException">A descriptor is <c>null</c> or two share a name.</exception>
    public ProgramDescriptorSet CreateDescriptorSet() =>
        Descriptors.Count == 0 ? ProgramDescriptorSet.Empty() : new ProgramDescriptorSet(Descriptors);

    /// <summary>Resolves the marker pair implied by the explicit markers and the configured language.</summary>
    /// <returns>
    /// The explicit pair when both markers are set, otherwise
    /// <see cref="EvolveBlockMarkers.ForLanguage"/> for <see cref="Language"/>.
    /// </returns>
    /// <exception cref="ArgumentException">Exactly one of the two markers is set, or a set marker is invalid.</exception>
    public EvolveBlockMarkers ResolveEvolveBlockMarkers()
    {
        string? start = EvolveBlockStartMarker;
        string? end = EvolveBlockEndMarker;
        if (start is null || start.Trim().Length == 0)
        {
            if (end is not null && end.Trim().Length != 0)
            {
                throw new ArgumentException(
                    "Set both EvolveBlockStartMarker and EvolveBlockEndMarker, or neither.",
                    nameof(EvolveBlockStartMarker));
            }

            return EvolveBlockMarkers.ForLanguage(Language);
        }

        if (end is null || end.Trim().Length == 0)
        {
            throw new ArgumentException(
                "Set both EvolveBlockStartMarker and EvolveBlockEndMarker, or neither.",
                nameof(EvolveBlockEndMarker));
        }

        return new EvolveBlockMarkers(start, end);
    }

    /// <summary>Returns a copy with evolve-block enforcement cleared, for applying edits to text that is not code.</summary>
    /// <returns>A copy when enforcement is on, otherwise this instance.</returns>
    /// <remarks>
    /// Evolve-block markers fence off the parts of a program that must not change. A changes description is prose
    /// about a program rather than a program, so it has no markers to fence, and enforcing them there would reject
    /// every edit for being outside a block that does not exist.
    /// </remarks>
    internal ProgramEvolutionOptions WithoutEvolveBlockEnforcement()
    {
        if (!EnforceEvolveBlocks) return this;
        ProgramEvolutionOptions relaxed = Clone();
        relaxed.EnforceEvolveBlocks = false;
        return relaxed;
    }

    /// <summary>Creates an independent copy so a running component is unaffected by later mutation.</summary>
    /// <returns>
    /// A new options instance carrying the same values, with every nested subsystem deep-copied. Nested options
    /// that were never touched stay uncreated in the copy, so cloning a bare instance allocates nothing extra.
    /// </returns>
    /// <remarks>
    /// <see cref="Descriptors"/> holds caller-supplied objects rather than values; the list is copied but the
    /// descriptors themselves are shared, which is safe because a descriptor is required to be stateless.
    /// </remarks>
    public ProgramEvolutionOptions Clone()
    {
        var copy = new ProgramEvolutionOptions
        {
            Language = Language,
            EvolveBlockStartMarker = EvolveBlockStartMarker,
            EvolveBlockEndMarker = EvolveBlockEndMarker,
            EnforceEvolveBlocks = EnforceEvolveBlocks,
            MaxProgramChars = MaxProgramChars,
            TaskDescription = TaskDescription,
            IncludeEliteSourceCount = IncludeEliteSourceCount,
            MaxEliteSourceChars = MaxEliteSourceChars,

            // EmbeddingNoveltyOptions validates in its constructor and exposes only get-only properties, so the
            // instance is a value and sharing the reference is safe. The mutable subsystems below are deep-copied.
            Novelty = Novelty
        };

        copy._diff = _diff is null ? null : _diff.Clone();
        copy._prompt = _prompt is null ? null : _prompt.Clone();
        copy._variation = _variation is null ? null : _variation.Clone();
        copy._sandbox = _sandbox is null ? null : _sandbox.Clone();
        copy._script = _script is null ? null : _script.Clone();
        copy._llmFeedback = _llmFeedback is null ? null : _llmFeedback.Clone();
        copy._metrics = _metrics is null ? null : _metrics.Clone();
        copy._provenance = _provenance is null ? null : _provenance.Clone();
        copy._artifactStore = _artifactStore is null ? null : _artifactStore.Clone();
        copy._runOutput = _runOutput is null ? null : _runOutput.Clone();
        copy._engine = _engine is null ? null : CopyEngineOptions(_engine);
        copy._seedPrograms = _seedPrograms is null ? null : new List<string>(_seedPrograms);
        copy._descriptors = _descriptors is null ? null : new List<IProgramDescriptor>(_descriptors);
        copy._testCases = _testCases is null ? null : CopyTestCases(_testCases);
        return copy;
    }

    /// <summary>Validates the language, marker pair, program bound, and every nested subsystem that was configured.</summary>
    /// <remarks>
    /// A nested subsystem that was never touched is not created just to be validated, because its defaults are
    /// valid by construction. Seed programs and descriptors are checked here so a misconfigured run fails at
    /// configuration time rather than after the first evaluation.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// The markers are inconsistent, a nested subsystem is invalid, a seed program is blank or oversized, or a
    /// descriptor is <c>null</c> or shares a name with another.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <see cref="Language"/> is not a defined value, <see cref="MaxProgramChars"/> is outside the range
    /// 1 to <see cref="ProgramGenome.MaxSourceLength"/>, or an elite-retention bound is out of range.
    /// </exception>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ProgramLanguage), Language))
            throw new ArgumentOutOfRangeException(nameof(Language), Language, "Value must be a defined language.");
        if (MaxProgramChars <= 0 || MaxProgramChars > ProgramGenome.MaxSourceLength)
            throw new ArgumentOutOfRangeException(nameof(MaxProgramChars), MaxProgramChars,
                $"Value must be between 1 and {ProgramGenome.MaxSourceLength}.");
        if (IncludeEliteSourceCount < 0 || IncludeEliteSourceCount > 10_000)
            throw new ArgumentOutOfRangeException(nameof(IncludeEliteSourceCount), IncludeEliteSourceCount,
                "Value must be between 0 and 10000.");
        if (MaxEliteSourceChars <= 0 || MaxEliteSourceChars > ProgramGenome.MaxSourceLength)
            throw new ArgumentOutOfRangeException(nameof(MaxEliteSourceChars), MaxEliteSourceChars,
                $"Value must be between 1 and {ProgramGenome.MaxSourceLength}.");
        if (Diff is null) throw new ArgumentException("Diff options cannot be null.", nameof(Diff));

        ResolveEvolveBlockMarkers();
        Diff.Validate();
        _prompt?.Validate();
        _variation?.Validate();
        _sandbox?.Validate();
        _script?.Validate();
        _llmFeedback?.Validate();
        _metrics?.Validate();
        _provenance?.Validate();
        _artifactStore?.Validate();
        _runOutput?.Validate();

        // Both of these write files, so they need somewhere to write. Refuse at configuration time rather than
        // after the first evaluation has already been paid for.
        if ((_artifactStore is not null || _runOutput is not null) &&
            string.IsNullOrWhiteSpace(Engine.OutputDirectory))
        {
            throw new ArgumentException(
                "ArtifactStore and RunOutput write beneath the run's output directory, so " +
                "Engine.OutputDirectory must be set when either is configured.",
                _artifactStore is not null ? nameof(ArtifactStore) : nameof(RunOutput));
        }

        // A store with nothing to store is the trap this check exists to prevent: engine artifact capture is
        // off by default, so configuring retention alone would create an empty directory and report success.
        // Saying which flag to set is better than either failing silently or flipping an engine option the
        // caller may have disabled on purpose.
        if (_artifactStore is not null && !Engine.Artifacts.Enabled)
        {
            throw new ArgumentException(
                "ArtifactStore retains the artifacts an evaluation produces, but the engine captures none " +
                "unless Engine.Artifacts.Enabled is set. Enable it, or leave ArtifactStore null.",
                nameof(ArtifactStore));
        }

        // Cascade evaluation needs a task that implements ICascadeEvolutionTask, and the task a program run builds
        // is not one. Enabling it here would validate, run, and never stage anything - and it is on by default in a
        // configuration ported straight from the reference implementation, so it is worth naming rather than ignoring.
        if (Engine.Cascade.Enabled)
        {
            throw new ArgumentException(
                "Cascade evaluation needs an evolution task that implements ICascadeEvolutionTask, and program " +
                "evolution builds a task that does not. Clear Engine.Cascade.Enabled, or evolve a genome type of " +
                "your own with a cascade-aware task.",
                nameof(Engine));
        }

        // Maintaining a changes description means editing it, and a full rewrite produces no edits to route. The two
        // settings are in different sections, so nothing else would catch the combination until every proposal
        // failed for a reason that reads like the model ignoring its instructions.
        if (Prompt.ProgramsAsChangesDescription && Variation.Mode == ProgramEvolutionMode.FullRewrite)
        {
            throw new ArgumentException(
                "Prompt.ProgramsAsChangesDescription maintains the description through edit blocks, which a full " +
                "rewrite does not produce. Use the diff variation mode, or clear ProgramsAsChangesDescription.",
                nameof(Prompt));
        }

        if (_seedPrograms is not null) CreateSeedGenomes();
        if (_testCases is not null)
        {
            foreach (ProgramInputOutputExample example in _testCases)
            {
                if (example is null) throw new ArgumentException("Test cases cannot contain null entries.", nameof(TestCases));
            }
        }

        if (_descriptors is not null && _descriptors.Count > 0)
        {
            // The descriptor set constructor is the single place that rejects nulls and duplicate names.
            _ = new ProgramDescriptorSet(_descriptors);
        }
    }

    private static List<ProgramInputOutputExample> CopyTestCases(IList<ProgramInputOutputExample> source)
    {
        var copy = new List<ProgramInputOutputExample>(source.Count);
        foreach (ProgramInputOutputExample example in source)
        {
            copy.Add(example is null
                ? throw new ArgumentException("Test cases cannot contain null entries.", nameof(source))
                : new ProgramInputOutputExample { Input = example.Input, ExpectedOutput = example.ExpectedOutput });
        }

        return copy;
    }

    // The engine options are copied by EvolutionEngineOptions.Copy() rather than by a list maintained here. The
    // hand-written copy this replaces silently dropped 19 of the 41 options, so a program-evolution run discarded
    // its cascade, early stopping, target quality, migration topology, selection policy and output directory
    // without reporting anything.
    private static EvolutionEngineOptions CopyEngineOptions(EvolutionEngineOptions source) => source.Copy();
}
