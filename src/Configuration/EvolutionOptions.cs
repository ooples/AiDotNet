using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>The user-facing control panel for an evolution run: budgets, islands, the archive grid, and persistence.</summary>
/// <remarks>
/// <para>
/// This is the options object <c>AiModelBuilder.ConfigureEvolution</c> takes. It carries every knob the engine
/// itself understands and three groups the engine cannot describe on its own: the behaviour axes the archive maps
/// candidates onto (<see cref="Descriptors"/>, <see cref="ArchiveDirection"/>, <see cref="ArchiveCapacity"/>), where
/// the run's files go (<see cref="OutputDirectory"/>, <see cref="CheckpointDirectory"/>,
/// <see cref="RetainOutput"/>), and what is written to a trace (<see cref="Trace"/>). Everything is validated when
/// <see cref="SnapshotAndValidate"/> runs, which the builder does at configure time, so a misconfigured search fails
/// on the line that configured it rather than after the first evaluation.
/// </para>
/// <para>
/// <see cref="SnapshotAndValidate"/> returns an independent copy. The builder keeps the copy, so mutating the object
/// you passed in after configuring cannot change a run that is already under way — the same defensive-copy contract
/// the engine applies to its own options.
/// </para>
/// <para>
/// Defaults are deliberately cautious: one island, no retries, no timeout, no checkpointing, no tracing, caching on.
/// A first run should be cheap and free of surprises. <see cref="CreateOpenEvolveDefaults"/> is the opt-in that
/// switches on everything the reference OpenEvolve implementation enables out of the box.
/// </para>
/// <para><b>For Beginners:</b> Evolutionary search keeps a population of candidate solutions, changes them, scores
/// them, and keeps the good ones. The budget settings say how long to search: <see cref="MaxEvaluationAttempts"/>
/// caps how many times your scoring function actually runs, and <see cref="TimeLimit"/> caps wall-clock time.
/// <see cref="Seed"/> makes a run repeatable — the same seed and settings always give the same answer.
/// <see cref="Descriptors"/> is the interesting one: each entry is a measurable property of a candidate (its size,
/// its depth, how long it takes) and the search keeps the best candidate found for every combination of those
/// properties, so you end up with a map of good and *different* answers instead of one winner. Start with the
/// defaults, add one or two descriptors, and raise the budgets once your scoring function is stable.</para>
/// </remarks>
public sealed class EvolutionOptions
{
    private IList<EvolutionDescriptorDefinition>? _descriptors;
    private EvolutionSelectionOptions? _selection;
    private EvolutionCascadeOptions? _cascade;
    private EvolutionArtifactOptions? _artifacts;
    private EvolutionEarlyStoppingOptions? _earlyStopping;
    private EvolutionTraceOptions? _trace;

    /// <summary>Gets or sets the stable run identifier that names this run's checkpoint and trace files.</summary>
    public string RunId { get; set; } = "default";

    /// <summary>Gets or sets the root seed every random stream in the run is derived from.</summary>
    public ulong Seed { get; set; } = 1234UL;

    /// <summary>Gets or sets the directory this run's checkpoint, trace, and artifact paths are derived from.</summary>
    /// <remarks>
    /// <c>null</c>, the default, means nothing is derived: the run writes only through a checkpoint store or an
    /// observer that was configured explicitly. Setting it names one folder under which
    /// <see cref="AiDotNet.Evolution.EvolutionOutputLayout"/> places a <c>checkpoints/</c> and a <c>traces/</c>
    /// subfolder named after <see cref="RunId"/>. Nothing is created until something is actually written.
    /// </remarks>
    public string? OutputDirectory { get; set; }

    /// <summary>Gets or sets the directory the checkpoint file goes in, overriding the one <see cref="OutputDirectory"/> derives.</summary>
    /// <remarks>
    /// Set this only when the checkpoint has to live somewhere other than the run's output folder — a shared volume,
    /// say. When both are <c>null</c> and checkpointing is requested, the builder derives a run-specific directory
    /// under the system temporary folder and reports it on the run summary.
    /// </remarks>
    public string? CheckpointDirectory { get; set; }

    /// <summary>Gets or sets whether a directory the builder derived for this run survives the run.</summary>
    /// <remarks>
    /// <c>true</c>, the default, keeps everything. <c>false</c> deletes a directory the builder derived itself,
    /// best-effort, once the run has finished; a directory you named through <see cref="OutputDirectory"/> or
    /// <see cref="CheckpointDirectory"/> is never deleted, because it is yours.
    /// </remarks>
    public bool RetainOutput { get; set; } = true;

    /// <summary>Gets or sets the maximum number of evaluator attempts, retries included.</summary>
    public int MaxEvaluationAttempts { get; set; } = 100;

    /// <summary>Gets or sets the maximum number of proposals, duplicates and rejections included.</summary>
    public int MaxProposals { get; set; } = 1_000;

    /// <summary>Gets or sets the maximum number of non-seed variation proposals.</summary>
    public int MaxGenerations { get; set; } = 1_000;

    /// <summary>Gets or sets how many proposals form one logical batch, independent of worker count.</summary>
    public int ProposalBatchSize { get; set; } = 8;

    /// <summary>Gets or sets how many evaluator calls may run at once.</summary>
    /// <remarks>In the default deterministic execution mode the result does not depend on this value.</remarks>
    public int MaxDegreeOfParallelism { get; set; } = 1;

    /// <summary>Gets or sets whether commits are ordered deterministically or by completion.</summary>
    public EvolutionExecutionMode ExecutionMode { get; set; } = EvolutionExecutionMode.Deterministic;

    /// <summary>Gets or sets whether evaluations run in fixed batches or in a continuously refilled window.</summary>
    /// <remarks>
    /// <see cref="EvolutionDispatchMode.Continuous"/> commits each evaluation as it finishes and dispatches a
    /// replacement immediately, so a slow candidate no longer holds the other workers idle. See
    /// <see cref="EvolutionEngineOptions.Dispatch"/> for the behaviour and the determinism it preserves.
    /// </remarks>
    public EvolutionDispatchMode Dispatch { get; set; } = EvolutionDispatchMode.Batch;

    /// <summary>Gets or sets how many evaluations may be in flight at once; zero follows the worker count.</summary>
    /// <remarks>Used only by <see cref="EvolutionDispatchMode.Continuous"/>.</remarks>
    public int MaxInFlight { get; set; }

    /// <summary>Gets or sets how many evaluations one island may have in flight; zero means no per-island limit.</summary>
    /// <remarks>
    /// Used only by <see cref="EvolutionDispatchMode.Continuous"/>. It stops one island's slow candidates from
    /// occupying the whole window while every other island stops advancing.
    /// </remarks>
    public int MaxInFlightPerIsland { get; set; }

    /// <summary>Gets or sets whether a recoverable candidate failure stops the whole run.</summary>
    public EvolutionFailurePolicy FailurePolicy { get; set; } = EvolutionFailurePolicy.Continue;

    /// <summary>Gets or sets how many further attempts a failed evaluation may have.</summary>
    public int MaxRetries { get; set; }

    /// <summary>Gets or sets which failure-like outcomes are eligible for another attempt.</summary>
    public EvolutionRetryStatuses RetryOn { get; set; } = EvolutionRetryStatuses.All;

    /// <summary>Gets or sets the pause before a retry round; zero, the default, retries immediately.</summary>
    public TimeSpan RetryBaseDelay { get; set; } = TimeSpan.Zero;

    /// <summary>Gets or sets the factor applied to <see cref="RetryBaseDelay"/> for each further attempt.</summary>
    public double RetryBackoffMultiplier { get; set; } = 1.0;

    /// <summary>Gets or sets a cooperative time limit for one evaluator attempt.</summary>
    public TimeSpan? EvaluationTimeout { get; set; }

    /// <summary>Gets or sets how long past <see cref="EvaluationTimeout"/> the run waits before abandoning a call.</summary>
    /// <remarks>Requires <see cref="EvaluationTimeout"/>; without it a slow evaluator that ignores its token blocks the batch.</remarks>
    public TimeSpan? EvaluationGracePeriod { get; set; }

    /// <summary>Gets or sets a wall-clock limit for the whole run.</summary>
    public TimeSpan? TimeLimit { get; set; }

    /// <summary>Gets or sets a quality that ends the run as soon as an elite reaches it.</summary>
    public double? TargetQuality { get; set; }

    /// <summary>Gets or sets how many committed evaluations pass between checkpoints; zero disables periodic saves.</summary>
    /// <remarks>
    /// Checkpointing needs a genome codec, so a positive value is only accepted by the
    /// <c>ConfigureEvolution</c> overload that takes one. A final checkpoint is always written when a run with a
    /// checkpoint store ends, whatever this value is.
    /// </remarks>
    public int CheckpointInterval { get; set; }

    /// <summary>Gets or sets whether a compatible checkpoint is loaded before new candidates are proposed.</summary>
    /// <remarks>Like <see cref="CheckpointInterval"/>, this needs the genome-codec overload of <c>ConfigureEvolution</c>.</remarks>
    public bool Resume { get; set; }

    /// <summary>Gets or sets whether a completed evaluation is reused when the same candidate appears again.</summary>
    public bool EnableEvaluationCache { get; set; } = true;

    /// <summary>Gets or sets whether a candidate that failed stays permanently deduplicated.</summary>
    public bool DeduplicateFailedCandidates { get; set; }

    /// <summary>Gets or sets how many independent archives the population is split across.</summary>
    public int IslandCount { get; set; } = 1;

    /// <summary>Gets or sets how many units pass between migrations; zero disables migration.</summary>
    public int MigrationInterval { get; set; } = 20;

    /// <summary>Gets or sets how many elites each island contributes to a migration round.</summary>
    public int MigrantsPerIsland { get; set; } = 2;

    /// <summary>Gets or sets the island graph migrations follow when the engine builds the migration policy.</summary>
    /// <remarks>
    /// <para>
    /// A ring passes elites to the next island only, which keeps islands genuinely separate for longer; a fully
    /// connected graph spreads a winner everywhere in a single migration, which converges faster and explores less.
    /// Ignored when a migration policy is passed to <c>ConfigureEvolution</c> directly.
    /// </para>
    /// <para><b>For Beginners:</b> Islands are separate sub-populations searching in parallel. This says who copies
    /// good solutions to whom when they exchange. Leave it on the ring unless the search converges too slowly.</para>
    /// </remarks>
    public EvolutionMigrationTopology MigrationTopology { get; set; } = EvolutionMigrationTopology.Ring;

    /// <summary>Gets or sets the fraction of a destination island's cells one migration may overwrite; zero means no cap.</summary>
    /// <remarks>
    /// A migration that overwrites most of a destination erases what that island found on its own, which is the
    /// diversity islands exist to protect. Capping the fraction keeps a migration an infusion rather than a
    /// replacement. Ignored when a migration policy is supplied directly.
    /// </remarks>
    public double MigrationRate { get; set; }

    /// <summary>Gets or sets whether a genome may migrate only once; by default it may migrate repeatedly.</summary>
    /// <remarks>Ignored when a migration policy is supplied directly.</remarks>
    public bool PreventRepeatedMigration { get; set; }

    /// <summary>Gets or sets which unit <see cref="MigrationInterval"/> counts.</summary>
    public EvolutionMigrationTrigger MigrationTrigger { get; set; } = EvolutionMigrationTrigger.CommittedBatches;

    /// <summary>Gets or sets how a new proposal is assigned to an island.</summary>
    public EvolutionIslandAssignmentStrategy IslandAssignment { get; set; } = EvolutionIslandAssignmentStrategy.RoundRobin;

    /// <summary>Gets or sets how many inspiration elites are handed to the variation operator.</summary>
    public int InspirationCount { get; set; } = 3;

    /// <summary>Gets or sets how many failure diagnostics a run keeps.</summary>
    public int MaxRetainedFailures { get; set; } = 128;

    /// <summary>Gets or sets the size of the cross-island elite index; zero disables it.</summary>
    public int GlobalEliteCount { get; set; }

    /// <summary>Gets or sets how many past evaluations each island remembers beyond its elites; zero disables it.</summary>
    public int HistorySize { get; set; }

    /// <summary>Gets or sets the structural novelty threshold; zero disables the pre-evaluation novelty gate.</summary>
    /// <remarks>A positive value needs a genome distance metric on the engine, which the facade does not configure.</remarks>
    public double NoveltyDistanceThreshold { get; set; }

    /// <summary>Gets or sets the descriptor filled from a completed evaluation's quality; <c>null</c> disables it.</summary>
    public string? QualityDescriptorName { get; set; }

    /// <summary>Gets or sets which built-in selection policy the engine builds when none is supplied.</summary>
    /// <remarks>
    /// <see cref="EvolutionSelectionPolicyKind.Ratio"/> is what makes <see cref="Selection"/> take effect; the other
    /// built-in policies ignore it. A policy passed to <c>ConfigureEvolution</c> always wins over this value.
    /// </remarks>
    public EvolutionSelectionPolicyKind SelectionPolicy { get; set; } = EvolutionSelectionPolicyKind.Uniform;

    /// <summary>Gets or sets the branch ratios and inspiration mix a ratio-based selection policy uses.</summary>
    public EvolutionSelectionOptions Selection
    {
        get => _selection ??= new EvolutionSelectionOptions();
        set => _selection = value;
    }

    /// <summary>Gets or sets staged (cascade) evaluation; disabled by default.</summary>
    /// <remarks>Enabling it requires an evolution task that implements <c>ICascadeEvolutionTask&lt;TGenome&gt;</c>.</remarks>
    public EvolutionCascadeOptions Cascade
    {
        get => _cascade ??= new EvolutionCascadeOptions();
        set => _cascade = value;
    }

    /// <summary>Gets or sets retention and sanitizing of untrusted evaluator artifacts; disabled by default.</summary>
    public EvolutionArtifactOptions Artifacts
    {
        get => _artifacts ??= new EvolutionArtifactOptions();
        set => _artifacts = value;
    }

    /// <summary>Gets or sets plateau-based early stopping; disabled by default.</summary>
    public EvolutionEarlyStoppingOptions EarlyStopping
    {
        get => _earlyStopping ??= new EvolutionEarlyStoppingOptions();
        set => _earlyStopping = value;
    }

    /// <summary>Gets or sets what a run records to its trace file; disabled by default.</summary>
    /// <remarks>
    /// Enabling this without a <see cref="EvolutionTraceOptions.Path"/> is legal here: the builder derives the path
    /// from <see cref="OutputDirectory"/> and <see cref="RunId"/> and reports it on the run summary.
    /// </remarks>
    public EvolutionTraceOptions Trace
    {
        get => _trace ??= new EvolutionTraceOptions();
        set => _trace = value;
    }

    /// <summary>Gets or sets the behaviour axes the archive maps candidates onto.</summary>
    /// <remarks>
    /// Each definition names a descriptor the evaluation must return, the range it spans, and how many bins that
    /// range is divided into. At least one is required for a typed-genome run, and <c>ConfigureEvolution</c> refuses
    /// an empty list at configure time rather than failing when the first archive is built. A program-evolution run
    /// may leave this empty, in which case the builder supplies a single documented axis — program length in
    /// sixteen clamped bins, spanning four times the longest seed program — because that is the one behaviour axis
    /// every program has without any domain knowledge.
    /// </remarks>
    public IList<EvolutionDescriptorDefinition> Descriptors
    {
        get => _descriptors ??= new List<EvolutionDescriptorDefinition>();
        set => _descriptors = value;
    }

    /// <summary>Gets or sets whether a higher or a lower quality is better.</summary>
    public EvolutionOptimizationDirection ArchiveDirection { get; set; } = EvolutionOptimizationDirection.Maximize;

    /// <summary>Gets or sets the maximum number of filled cells one island keeps; zero means unbounded.</summary>
    public int ArchiveCapacity { get; set; }

    /// <summary>Gets or sets the ceiling on the archive's total cell count, which guards against an unusable grid.</summary>
    /// <remarks>The product of every descriptor's bin count must stay at or below this value.</remarks>
    public long MaximumArchiveGridCells { get; set; } = 10_000_000L;

    /// <summary>Creates options equivalent to the shipped defaults of OpenEvolve 0.3.2.</summary>
    /// <returns>A new instance; the property defaults of this class are unchanged.</returns>
    /// <remarks>
    /// <para>
    /// Every value is taken from <see cref="EvolutionEngineOptions.CreateOpenEvolveDefaults"/>, which cites the
    /// upstream source line for each one, so the two factories cannot drift apart. The archive settings this class
    /// adds keep their own defaults, because upstream's feature grid is described per run rather than globally.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when you are reproducing an OpenEvolve experiment, or when you simply
    /// want the fuller configuration without looking every number up. The object you get back is an ordinary
    /// options instance, so change whatever you like afterwards.</para>
    /// </remarks>
    public static EvolutionOptions CreateOpenEvolveDefaults()
    {
        // Copying wholesale is what keeps the two factories from drifting: a value added to the engine's upstream
        // defaults reaches this one without anybody remembering to restate it here.
        return FromEngineOptions(EvolutionEngineOptions.CreateOpenEvolveDefaults());
    }

    /// <summary>Validates every value and returns an independent copy that later mutation cannot reach.</summary>
    /// <returns>A validated deep copy, safe to hand to a run.</returns>
    /// <exception cref="ArgumentException">
    /// A required string is blank, a descriptor is <c>null</c> or shares a name with another, or a nested options
    /// group is <c>null</c> or internally inconsistent.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">A numeric or enumeration value is outside its permitted range.</exception>
    public EvolutionOptions SnapshotAndValidate()
    {
        Guard.NotNullOrWhiteSpace(RunId);
        Guard.NotNull(Selection);
        Guard.NotNull(Cascade);
        Guard.NotNull(Artifacts);
        Guard.NotNull(EarlyStopping);
        Guard.NotNull(Trace);
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), ArchiveDirection))
            throw new ArgumentOutOfRangeException(nameof(ArchiveDirection), ArchiveDirection, "Value must be a defined direction.");
        if (ArchiveCapacity < 0)
            throw new ArgumentOutOfRangeException(nameof(ArchiveCapacity), ArchiveCapacity, "Value cannot be negative.");
        if (MaximumArchiveGridCells <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaximumArchiveGridCells), MaximumArchiveGridCells, "Value must be positive.");

        List<EvolutionDescriptorDefinition> descriptors = CopyDescriptors();

        // Building the engine options runs the engine's own validation over every shared knob, so one rule set
        // covers both surfaces and the facade cannot accept a value the engine would later reject.
        EvolutionEngineOptions engine = ToEngineOptions().SnapshotAndValidate();
        EvolutionTraceOptions trace = SnapshotTrace();

        // Every engine-shared setting is copied by FromEngineOptions, so this method only adds what the facade owns.
        // Listing the shared ones a second time here is how one of them silently stops being carried across.
        EvolutionOptions snapshot = FromEngineOptions(engine);
        snapshot.CheckpointDirectory = ResolveDirectory(CheckpointDirectory, nameof(CheckpointDirectory));
        snapshot.RetainOutput = RetainOutput;
        snapshot.Trace = trace;
        snapshot.ArchiveDirection = ArchiveDirection;
        snapshot.ArchiveCapacity = ArchiveCapacity;
        snapshot.MaximumArchiveGridCells = MaximumArchiveGridCells;
        snapshot.Descriptors = descriptors;
        return snapshot;
    }

    /// <summary>Projects the shared knobs onto the options type the engine takes.</summary>
    /// <returns>A new engine options instance; the archive, trace, and output-retention settings stay behind.</returns>
    /// <remarks>
    /// The engine defensively copies and validates what it is given, so the returned instance is deliberately not
    /// snapshotted here; callers that need a validated copy call <see cref="SnapshotAndValidate"/> first.
    /// </remarks>
    internal EvolutionEngineOptions ToEngineOptions() => new()
    {
        RunId = RunId,
        Seed = Seed,
        OutputDirectory = OutputDirectory,
        MaxEvaluationAttempts = MaxEvaluationAttempts,
        MaxProposals = MaxProposals,
        MaxGenerations = MaxGenerations,
        ProposalBatchSize = ProposalBatchSize,
        MaxDegreeOfParallelism = MaxDegreeOfParallelism,
        ExecutionMode = ExecutionMode,
        Dispatch = Dispatch,
        MaxInFlight = MaxInFlight,
        MaxInFlightPerIsland = MaxInFlightPerIsland,
        FailurePolicy = FailurePolicy,
        MaxRetries = MaxRetries,
        RetryOn = RetryOn,
        RetryBaseDelay = RetryBaseDelay,
        RetryBackoffMultiplier = RetryBackoffMultiplier,
        EvaluationTimeout = EvaluationTimeout,
        EvaluationGracePeriod = EvaluationGracePeriod,
        TimeLimit = TimeLimit,
        TargetQuality = TargetQuality,
        CheckpointInterval = CheckpointInterval,
        Resume = Resume,
        EnableEvaluationCache = EnableEvaluationCache,
        DeduplicateFailedCandidates = DeduplicateFailedCandidates,
        IslandCount = IslandCount,
        MigrationInterval = MigrationInterval,
        MigrantsPerIsland = MigrantsPerIsland,
        MigrationTrigger = MigrationTrigger,
        MigrationTopology = MigrationTopology,
        MigrationRate = MigrationRate,
        PreventRepeatedMigration = PreventRepeatedMigration,
        IslandAssignment = IslandAssignment,
        InspirationCount = InspirationCount,
        MaxRetainedFailures = MaxRetainedFailures,
        GlobalEliteCount = GlobalEliteCount,
        HistorySize = HistorySize,
        NoveltyDistanceThreshold = NoveltyDistanceThreshold,
        QualityDescriptorName = QualityDescriptorName,
        SelectionPolicy = SelectionPolicy,
        Selection = Selection,
        Cascade = Cascade,
        Artifacts = Artifacts,
        EarlyStopping = EarlyStopping
    };

    /// <summary>Creates one empty archive for an island, using the configured grid.</summary>
    /// <typeparam name="TGenome">The genome type the archive stores.</typeparam>
    /// <returns>A fresh <see cref="MapElitesArchive{TGenome}"/>; every island gets its own instance.</returns>
    internal MapElitesArchive<TGenome> CreateArchive<TGenome>() =>
        new(Descriptors, ArchiveDirection, ArchiveCapacity, MaximumArchiveGridCells);

    /// <summary>Returns whether this run needs somewhere to write a checkpoint or a trace.</summary>
    /// <returns><c>true</c> when checkpointing, resume, or tracing is requested.</returns>
    internal bool NeedsOutputLocation() =>
        Resume || CheckpointInterval > 0 || CheckpointDirectory is not null || Trace.Enabled;

    /// <summary>Mirrors an engine options instance back onto the facade surface.</summary>
    /// <param name="engine">The engine options to copy; every shared knob is carried across by value.</param>
    /// <returns>A new facade options instance carrying the same engine settings and this class's own defaults.</returns>
    /// <remarks>
    /// The archive, trace, and output-retention settings this class adds have no counterpart on the engine options,
    /// so they keep their defaults. This is how a program-evolution run that configured only
    /// <c>ProgramEvolutionOptions.Engine</c> reaches the facade's build path without the caller having to restate
    /// the same numbers through <c>ConfigureEvolution</c>.
    /// </remarks>
    internal static EvolutionOptions FromEngineOptions(EvolutionEngineOptions engine)
    {
        Guard.NotNull(engine);
        return new EvolutionOptions
        {
            RunId = engine.RunId,
            Seed = engine.Seed,
            OutputDirectory = engine.OutputDirectory,
            MaxEvaluationAttempts = engine.MaxEvaluationAttempts,
            MaxProposals = engine.MaxProposals,
            MaxGenerations = engine.MaxGenerations,
            ProposalBatchSize = engine.ProposalBatchSize,
            MaxDegreeOfParallelism = engine.MaxDegreeOfParallelism,
            ExecutionMode = engine.ExecutionMode,
            Dispatch = engine.Dispatch,
            MaxInFlight = engine.MaxInFlight,
            MaxInFlightPerIsland = engine.MaxInFlightPerIsland,
            FailurePolicy = engine.FailurePolicy,
            MaxRetries = engine.MaxRetries,
            RetryOn = engine.RetryOn,
            RetryBaseDelay = engine.RetryBaseDelay,
            RetryBackoffMultiplier = engine.RetryBackoffMultiplier,
            EvaluationTimeout = engine.EvaluationTimeout,
            EvaluationGracePeriod = engine.EvaluationGracePeriod,
            TimeLimit = engine.TimeLimit,
            TargetQuality = engine.TargetQuality,
            CheckpointInterval = engine.CheckpointInterval,
            Resume = engine.Resume,
            EnableEvaluationCache = engine.EnableEvaluationCache,
            DeduplicateFailedCandidates = engine.DeduplicateFailedCandidates,
            IslandCount = engine.IslandCount,
            MigrationInterval = engine.MigrationInterval,
            MigrantsPerIsland = engine.MigrantsPerIsland,
            MigrationTrigger = engine.MigrationTrigger,
            MigrationTopology = engine.MigrationTopology,
            MigrationRate = engine.MigrationRate,
            PreventRepeatedMigration = engine.PreventRepeatedMigration,
            IslandAssignment = engine.IslandAssignment,
            InspirationCount = engine.InspirationCount,
            MaxRetainedFailures = engine.MaxRetainedFailures,
            GlobalEliteCount = engine.GlobalEliteCount,
            HistorySize = engine.HistorySize,
            NoveltyDistanceThreshold = engine.NoveltyDistanceThreshold,
            QualityDescriptorName = engine.QualityDescriptorName,
            SelectionPolicy = engine.SelectionPolicy,
            Selection = engine.Selection,
            Cascade = engine.Cascade,
            Artifacts = engine.Artifacts,
            EarlyStopping = engine.EarlyStopping
        };
    }

    /// <summary>Validates the trace settings while allowing an enabled trace to leave its path unset.</summary>
    /// <returns>A validated copy whose path is still <c>null</c> when the builder is expected to derive one.</returns>
    /// <remarks>
    /// <see cref="EvolutionTraceOptions"/> insists on an explicit path for an enabled trace, because the engine has
    /// no notion of a run's output folder. The facade does, so every other trace field is validated here against a
    /// stand-in path that is never written, and the real one is derived from <see cref="OutputDirectory"/> and
    /// <see cref="RunId"/> when the run starts.
    /// </remarks>
    private EvolutionTraceOptions SnapshotTrace()
    {
        if (!Trace.Enabled || !string.IsNullOrWhiteSpace(Trace.Path)) return Trace.SnapshotAndValidate();

        EvolutionTraceOptions probe = CopyTrace(Trace);
        probe.Path = "evolution-trace";
        EvolutionTraceOptions validated = probe.SnapshotAndValidate();
        validated.Path = null;
        return validated;
    }

    private static EvolutionTraceOptions CopyTrace(EvolutionTraceOptions source) => new()
    {
        Enabled = source.Enabled,
        Path = source.Path,
        Format = source.Format,
        Compress = source.Compress,
        FlushEveryRecords = source.FlushEveryRecords,
        MaxBytes = source.MaxBytes,
        MaxRecords = source.MaxRecords,
        IncludeDescriptors = source.IncludeDescriptors,
        IncludeLineage = source.IncludeLineage,
        IncludeDiagnostics = source.IncludeDiagnostics,
        ParentQualityCacheSize = source.ParentQualityCacheSize
    };

    /// <summary>Copies the trace settings with a resolved file path in place of the configured one.</summary>
    /// <param name="path">The path the trace file is written to.</param>
    /// <returns>A copy ready to hand to an <see cref="AiDotNet.Evolution.EvolutionTraceObserver{TGenome}"/>.</returns>
    internal EvolutionTraceOptions CreateTraceOptions(string path)
    {
        EvolutionTraceOptions copy = CopyTrace(Trace);
        copy.Path = path;
        return copy;
    }

    private List<EvolutionDescriptorDefinition> CopyDescriptors()
    {
        var copy = new List<EvolutionDescriptorDefinition>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (EvolutionDescriptorDefinition descriptor in Descriptors)
        {
            if (descriptor is null)
                throw new ArgumentException("Descriptors cannot contain a null entry.", nameof(Descriptors));
            if (!names.Add(descriptor.Name))
                throw new ArgumentException(
                    $"Two descriptors share the name '{descriptor.Name}'; behaviour axes must be distinct.",
                    nameof(Descriptors));
            copy.Add(descriptor);
        }

        return copy;
    }

    private static string? ResolveDirectory(string? directory, string parameterName)
    {
        if (directory is null) return null;
        if (string.IsNullOrWhiteSpace(directory))
            throw new ArgumentException("The directory cannot be empty.", parameterName);
        try
        {
            return Path.GetFullPath(directory.Trim());
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException or PathTooLongException)
        {
            throw new ArgumentException("The directory is not a valid path.", parameterName, exception);
        }
    }
}
