using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>Configures deterministic orchestration, budgets, parallelism, islands, and persistence.</summary>
/// <remarks>
/// <para>
/// <see cref="EvolutionEngine{TGenome}"/> validates and defensively copies these options in its constructor, so
/// mutating the original object afterwards has no effect on a running engine. Every option that can change the
/// logical search trajectory (seed, budgets, batch size, execution mode, failure policy, retries, evaluation
/// timeout, caching, deduplication, islands, migration, inspirations, the failure-retention bound, island
/// assignment, the migration trigger, the global elite and history bounds, the novelty threshold, the derived
/// quality descriptor, the selection ratios, the cascade, artifact, early-stopping and retry settings, the
/// evaluation grace period, and the target quality) is folded into the engine's configuration and compatibility
/// hashes, so a checkpoint written under one configuration refuses to resume under another. <see cref="RunId"/>, <see cref="MaxDegreeOfParallelism"/>,
/// <see cref="TimeLimit"/>, <see cref="CheckpointInterval"/>, <see cref="Resume"/>, and <see cref="OutputDirectory"/>
/// are deliberately excluded because they affect only speed, the identity of the stored run, persistence, and where
/// files are written, never the result of a given step. Excluding <see cref="OutputDirectory"/> matters especially:
/// it resolves to a machine-specific absolute path, so hashing it would make a run's configuration hash differ
/// between machines and stop a checkpoint from moving with its output folder.
/// </para>
/// <para><b>For Beginners:</b> This is the control panel for an evolutionary quality-diversity search. The budget
/// settings say how long to search: <see cref="MaxEvaluationAttempts"/> caps how many times your evaluator is
/// actually called, <see cref="MaxProposals"/> caps how many candidates are generated (including duplicates that
/// are never evaluated), and <see cref="TimeLimit"/> caps wall-clock time; the run stops at whichever limit is hit
/// first. <see cref="Seed"/> makes the run repeatable: the same seed and options always produce the same archive.
/// <see cref="MaxDegreeOfParallelism"/> lets several candidates be scored at once on a multi-core machine without
/// changing the answer, and <see cref="IslandCount"/> splits the population into semi-independent groups that
/// periodically exchange their best members, which discourages everyone converging on one idea. Start with the
/// defaults, raise the budgets once your evaluator is stable, and for long runs set <see cref="CheckpointInterval"/>
/// together with <see cref="Resume"/> (plus a checkpoint store and genome codec on the engine) so a crash does not
/// lose the work.</para>
/// <para>
/// Background: the one-archive-per-island layout follows MAP-Elites (Mouret &amp; Clune, 2015, "Illuminating search
/// spaces by mapping elites", arXiv:1504.04909) and periodic elite exchange follows the island model of parallel
/// genetic algorithms (Whitley, Rana &amp; Heckendorn, 1999). Candidate-local random streams are derived from
/// <see cref="Seed"/> and the candidate's evaluation identifier with a stable generator, which is why results do not
/// depend on <see cref="MaxDegreeOfParallelism"/> in <see cref="EvolutionExecutionMode.Deterministic"/> mode.
/// </para>
/// </remarks>
public sealed class EvolutionEngineOptions
{
    /// <summary>Gets or sets the stable run identifier used by checkpoint stores.</summary>
    public string RunId { get; set; } = "default";

    /// <summary>Gets or sets the root seed used to derive candidate-local streams.</summary>
    public ulong Seed { get; set; } = 1234UL;

    /// <summary>Gets or sets one directory under which this run's checkpoint and trace paths are derived; <c>null</c> derives nothing.</summary>
    /// <remarks>
    /// <para>
    /// The default of <c>null</c> preserves today's behaviour exactly: nothing is derived and the engine writes only
    /// through the checkpoint store and observer it was handed. Setting it names a root that
    /// <see cref="EvolutionOutputLayout"/> turns into deterministic per-run paths - a checkpoint under
    /// <c>checkpoints/</c> and a trace under <c>traces/</c>, both named after <see cref="RunId"/> - so a caller
    /// configures one folder instead of three file names, and a resumed run finds its own checkpoint without being
    /// told where it went.
    /// </para>
    /// <para>
    /// The value is resolved to an absolute path when the options are validated but no directory is created here;
    /// the checkpoint store and the trace observer each create their own parent directory the first time they write,
    /// so a configured output directory costs nothing until something is actually saved. It is deliberately excluded
    /// from the configuration hash because an absolute path is machine-specific. OpenEvolve derives the same three
    /// locations from its own <c>output_dir</c> but creates the directory in the controller constructor and gives
    /// every run in that directory the same trace file name (controller.py:52-56, 142-160).
    /// </para>
    /// </remarks>
    public string? OutputDirectory { get; set; }

    /// <summary>Gets or sets the maximum number of actual evaluator attempts, including retries.</summary>
    public int MaxEvaluationAttempts { get; set; } = 100;

    /// <summary>Gets or sets the maximum number of proposals, including duplicates and rejected proposals.</summary>
    public int MaxProposals { get; set; } = 1_000;

    /// <summary>Gets or sets the maximum number of non-seed variation proposals.</summary>
    public int MaxGenerations { get; set; } = 1_000;

    /// <summary>Gets or sets the deterministic logical batch size, independent of worker count.</summary>
    public int ProposalBatchSize { get; set; } = 8;

    /// <summary>Gets or sets the maximum number of concurrently running evaluator calls.</summary>
    /// <remarks>
    /// Excluded from the compatibility hash; in <see cref="EvolutionExecutionMode.Deterministic"/> mode the result
    /// is independent of this value.
    /// </remarks>
    public int MaxDegreeOfParallelism { get; set; } = 1;

    /// <summary>Gets or sets deterministic or opportunistic commit behavior.</summary>
    public EvolutionExecutionMode ExecutionMode { get; set; } = EvolutionExecutionMode.Deterministic;

    /// <summary>Gets or sets whether recoverable candidate failures stop the run.</summary>
    public EvolutionFailurePolicy FailurePolicy { get; set; } = EvolutionFailurePolicy.Continue;

    /// <summary>Gets or sets the maximum retries after the first evaluator attempt.</summary>
    public int MaxRetries { get; set; }

    /// <summary>Gets or sets an optional cooperative timeout per evaluator attempt.</summary>
    public TimeSpan? EvaluationTimeout { get; set; }

    /// <summary>Gets or sets how long past <see cref="EvaluationTimeout"/> the engine keeps waiting before abandoning a call.</summary>
    /// <remarks>
    /// <c>null</c>, the default, keeps the purely cooperative behaviour: the engine signals the evaluator's token and
    /// then awaits the call for as long as it takes, so an evaluator that ignores its token blocks the batch. When a
    /// grace period is set alongside <see cref="EvaluationTimeout"/>, the engine stops waiting after the sum of the two,
    /// records the attempt as <see cref="EvolutionEvaluationStatus.TimedOut"/>, counts it in
    /// <c>EvolutionRunCounters.AbandonedEvaluations</c>, and lets the batch continue. The abandoned call keeps running
    /// and its parallelism slot is released immediately, so a run with many abandoned calls can briefly exceed
    /// <see cref="MaxDegreeOfParallelism"/> concurrent evaluators; that is the deliberate price of not blocking the run.
    /// </remarks>
    public TimeSpan? EvaluationGracePeriod { get; set; }

    /// <summary>Gets or sets which failure-like statuses are eligible for another evaluation attempt.</summary>
    /// <remarks>
    /// The default <see cref="EvolutionRetryStatuses.All"/> preserves the engine's historical behaviour. Retries are
    /// still bounded by <see cref="MaxRetries"/> and by the run's evaluation-attempt budget, and each retry is charged
    /// exactly once.
    /// </remarks>
    public EvolutionRetryStatuses RetryOn { get; set; } = EvolutionRetryStatuses.All;

    /// <summary>Gets or sets the pause before a retry round; zero, the default, retries immediately.</summary>
    public TimeSpan RetryBaseDelay { get; set; } = TimeSpan.Zero;

    /// <summary>Gets or sets the factor applied to <see cref="RetryBaseDelay"/> for each additional attempt.</summary>
    /// <remarks>
    /// The delay before the attempt numbered <c>n</c> (one-based) is <see cref="RetryBaseDelay"/> multiplied by this
    /// factor raised to the power <c>n - 2</c>, capped at one minute. The sequence is a pure function of the attempt
    /// number, so it adds no randomness and cannot change a run's outcome. OpenEvolve instead sleeps a hard-coded one
    /// second between attempts (evaluator.py:283-285).
    /// </remarks>
    public double RetryBackoffMultiplier { get; set; } = 1.0;

    /// <summary>Gets or sets a quality that ends the run once an elite reaches it; <c>null</c> disables the check.</summary>
    /// <remarks>
    /// Compared against the best elite across every island in the archive's optimization direction after each committed
    /// batch, so a minimizing run stops when the best quality falls to or below the target. The run then reports
    /// <see cref="EvolutionStopReason.TargetReached"/> and always writes a final checkpoint. OpenEvolve only ever
    /// compares a metric literally named <c>combined_score</c> and only in the maximizing direction, and reports a
    /// generic completion (process_parallel.py:736-745).
    /// </remarks>
    public double? TargetQuality { get; set; }

    /// <summary>Gets or sets an optional wall-clock limit for the overall run.</summary>
    public TimeSpan? TimeLimit { get; set; }

    /// <summary>Gets or sets the number of committed evaluations between checkpoints; zero disables periodic saves.</summary>
    /// <remarks>
    /// Periodic saves require a checkpoint store and genome codec on the engine. A final checkpoint is always written
    /// when a run with a checkpoint store ends, regardless of this value.
    /// </remarks>
    public int CheckpointInterval { get; set; }

    /// <summary>Gets or sets whether a compatible checkpoint is loaded before proposing new candidates.</summary>
    /// <remarks>
    /// Requires a checkpoint store and genome codec. The checkpoint must have been written by an engine with an
    /// identical compatibility hash, and any seeds supplied to the run must match the checkpointed seed sequence.
    /// </remarks>
    public bool Resume { get; set; }

    /// <summary>Gets or sets whether completed canonical evaluations are reused across islands and proposals.</summary>
    public bool EnableEvaluationCache { get; set; } = true;

    /// <summary>Gets or sets whether final failed/timed-out identities remain permanently deduplicated.</summary>
    public bool DeduplicateFailedCandidates { get; set; }

    /// <summary>Gets or sets the number of independent archives.</summary>
    public int IslandCount { get; set; } = 1;

    /// <summary>Gets or sets the number of committed logical batches between migrations; zero disables migration.</summary>
    public int MigrationInterval { get; set; } = 20;

    /// <summary>Gets or sets the maximum elites copied from each source during migration.</summary>
    public int MigrantsPerIsland { get; set; } = 2;

    /// <summary>Gets or sets the number of inspiration elites supplied to variation.</summary>
    public int InspirationCount { get; set; } = 3;

    /// <summary>Gets or sets the maximum retained failure diagnostics in checkpoint state.</summary>
    public int MaxRetainedFailures { get; set; } = 128;

    /// <summary>Gets or sets how the engine assigns a new proposal to an island.</summary>
    /// <remarks>
    /// The default <see cref="EvolutionIslandAssignmentStrategy.RoundRobin"/> keeps islands balanced;
    /// <see cref="EvolutionIslandAssignmentStrategy.InheritParent"/> keeps a child on the island its parent was
    /// drawn from, which matters when the preferred island was empty and a parent had to be borrowed.
    /// </remarks>
    public EvolutionIslandAssignmentStrategy IslandAssignment { get; set; } = EvolutionIslandAssignmentStrategy.RoundRobin;

    /// <summary>Gets or sets the unit counted before an island migration round runs.</summary>
    /// <remarks>
    /// <see cref="MigrationInterval"/> is interpreted in this unit. The default counts committed logical batches;
    /// <see cref="EvolutionMigrationTrigger.IslandGenerations"/> instead waits until the busiest island has advanced
    /// by that many variation proposals, which is how OpenEvolve schedules migration.
    /// </remarks>
    public EvolutionMigrationTrigger MigrationTrigger { get; set; } = EvolutionMigrationTrigger.CommittedBatches;

    /// <summary>Gets or sets the size of the cross-island global elite index; zero disables it.</summary>
    /// <remarks>
    /// When positive, every completed evaluation whose descriptors map to a cell is offered to the index, which keeps
    /// the best entries across all islands in <see cref="EvolutionRunResult{TGenome}.GlobalElites"/>, checkpoints
    /// them, and folds them into the run state hash.
    /// <para>
    /// One derived exception: when the engine builds the selection policy itself from
    /// <see cref="SelectionPolicy"/> and that policy draws its exploitation pool from the cross-island index
    /// (<see cref="EvolutionSelectionPolicyKind.Ratio"/> with
    /// <see cref="EvolutionExploitationSource.GlobalTopK"/>), the engine raises the effective capacity to the
    /// policy's <see cref="EvolutionSelectionOptions.ExploitationEliteCount"/> so that branch is not silently dead.
    /// The derivation is a pure function of these options and of the constructed policy, both of which are in the
    /// compatibility hash, so a resumed run derives the same capacity. A policy the caller supplies directly never
    /// triggers it.
    /// </para>
    /// </remarks>
    public int GlobalEliteCount { get; set; }

    /// <summary>Gets or sets the bounded per-island population history size; zero disables it.</summary>
    /// <remarks>
    /// When positive, each island retains up to this many completed evaluations beyond its current elites, evicting
    /// homeless entries worst-first and never evicting the island best or the entry just added, so total memory is
    /// bounded by <see cref="IslandCount"/> times this value.
    /// </remarks>
    public int HistorySize { get; set; }

    /// <summary>Gets or sets the structural novelty threshold; zero disables the pre-evaluation novelty gate.</summary>
    /// <remarks>
    /// Requires a genome distance metric on the engine. A proposal whose smallest distance to an elite of its target
    /// island is below this value is rejected before the evaluator runs, at a cost of at most one distance call per
    /// occupied cell and no network traffic at all.
    /// </remarks>
    public double NoveltyDistanceThreshold { get; set; }

    /// <summary>Gets or sets the descriptor name filled from a completed evaluation's quality; <c>null</c> disables it.</summary>
    /// <remarks>
    /// This supplies the genome-agnostic equivalent of OpenEvolve's built-in <c>score</c> feature dimension. A task
    /// that already returns a descriptor with this name keeps its own value.
    /// </remarks>
    public string? QualityDescriptorName { get; set; }

    /// <summary>Gets or sets the branch ratios and inspiration mix used by ratio-based selection policies.</summary>
    /// <remarks>
    /// The engine validates and copies this object and folds it into the configuration hash even when the configured
    /// selection policy ignores it, so a facade can expose one selection knob per run.
    /// </remarks>
    public EvolutionSelectionOptions Selection { get; set; } = new();

    /// <summary>Gets or sets which built-in selection policy the engine builds when the caller supplies none.</summary>
    /// <remarks>
    /// <para>
    /// The default <see cref="EvolutionSelectionPolicyKind.Uniform"/> is the policy the engine has always used, so
    /// leaving this alone preserves existing behaviour exactly. A policy passed to the
    /// <see cref="EvolutionEngine{TGenome}"/> constructor always wins over this value, which the engine reads only
    /// when that argument is <c>null</c>. <see cref="EvolutionSelectionPolicyKind.Ratio"/> is what makes
    /// <see cref="Selection"/> take effect, because the ratio policy is the only built-in policy that reads it.
    /// </para>
    /// <para>
    /// This value is deliberately absent from the canonical configuration string: the engine folds the constructed
    /// policy's own identifier and version hash into its compatibility hash instead, which is strictly more precise
    /// because it also covers a policy the caller supplied directly. Two engines that behave identically therefore
    /// still resume each other's checkpoints, and two that do not, do not.
    /// </para>
    /// </remarks>
    public EvolutionSelectionPolicyKind SelectionPolicy { get; set; } = EvolutionSelectionPolicyKind.Uniform;

    /// <summary>Gets or sets staged (cascade) evaluation; disabled by default.</summary>
    /// <remarks>
    /// Enabling this requires the task to implement <c>ICascadeEvolutionTask&lt;TGenome&gt;</c>; the thresholds and
    /// per-stage timeouts are validated against that task's stage count when the engine is constructed.
    /// </remarks>
    public EvolutionCascadeOptions Cascade { get; set; } = new();

    /// <summary>Gets or sets retention, sanitizing, and replay of untrusted evaluator artifacts; disabled by default.</summary>
    public EvolutionArtifactOptions Artifacts { get; set; } = new();

    /// <summary>Gets or sets plateau-based early stopping; disabled by default.</summary>
    public EvolutionEarlyStoppingOptions EarlyStopping { get; set; } = new();

    /// <summary>Creates options equivalent to the shipped defaults of OpenEvolve 0.3.2.</summary>
    /// <returns>A new instance; the property defaults of this class are unchanged.</returns>
    /// <remarks>
    /// <para>
    /// The property defaults on this class are deliberately conservative, so several features OpenEvolve enables out
    /// of the box - islands, retries, an evaluation timeout, the cross-island elite index, per-island history, and
    /// artifact retention - are off unless a caller asks for them. Changing those defaults would silently change
    /// every run that already exists, so this factory is the opt-in instead: one call configures a run the way
    /// upstream would, and nobody else moves.
    /// </para>
    /// <para>
    /// Every value is taken from openevolve 0.3.2 <c>config.py</c>. <see cref="Seed"/> is 42 (<c>random_seed</c>,
    /// line 424); <see cref="MaxEvaluationAttempts"/>, <see cref="MaxProposals"/> and <see cref="MaxGenerations"/>
    /// are 10000 (<c>max_iterations</c>, line 420); <see cref="CheckpointInterval"/> is 100
    /// (<c>checkpoint_interval</c>, line 421); <see cref="MaxRetries"/> is 3 and <see cref="EvaluationTimeout"/> is
    /// 300 seconds (<c>evaluator.max_retries</c> and <c>evaluator.timeout</c>, lines 376-377);
    /// <see cref="RetryBaseDelay"/> is one second with <see cref="RetryBackoffMultiplier"/> 1.0, matching the
    /// hard-coded <c>asyncio.sleep(1.0)</c> of evaluator.py:283-285, and <see cref="RetryOn"/> covers failures only
    /// because upstream returns a timeout without retrying it (evaluator.py:252-265);
    /// <see cref="IslandCount"/> is 5 (<c>database.num_islands</c>, line 323); <see cref="MigrationInterval"/> is 50
    /// counted in island generations, matching "Migrate every N generations"
    /// (<c>database.migration_interval</c>, line 351); <see cref="GlobalEliteCount"/> is 100
    /// (<c>database.archive_size</c>, line 322); <see cref="HistorySize"/> is 1000
    /// (<c>database.population_size</c>, line 321); artifacts are enabled
    /// (<c>evaluator.enable_artifacts</c>, line 398); <see cref="MaxDegreeOfParallelism"/> is 1
    /// (<c>evaluator.parallel_evaluations</c>, line 389) and <see cref="ProposalBatchSize"/> is 1 so each candidate
    /// is committed before the next parent is drawn, which is how upstream's controller refills work
    /// (process_parallel.py:588-602); <see cref="InspirationCount"/> is 5 with three top and two diverse picks
    /// (<c>prompt.num_top_programs</c> 3 and <c>prompt.num_diverse_programs</c> 2, lines 268-269); and
    /// <see cref="SelectionPolicy"/> is <see cref="EvolutionSelectionPolicyKind.Ratio"/> with the 0.2 / 0.7 / 0.1
    /// mixture of <c>database.exploration_ratio</c>, <c>exploitation_ratio</c> and <c>elite_selection_ratio</c>
    /// (lines 326-328). <see cref="EvolutionEarlyStoppingOptions.MinimumImprovement"/> already defaults to the
    /// upstream <c>convergence_threshold</c> of 0.001 (line 442), and early stopping stays off because upstream's
    /// <c>early_stopping_patience</c> defaults to <c>None</c> (line 441).
    /// </para>
    /// <para>
    /// Three upstream defaults have no faithful counterpart here and are therefore left alone.
    /// <c>evaluator.cascade_evaluation</c> is <c>true</c> upstream, but <see cref="Cascade"/> stays disabled because
    /// enabling it requires a task that implements <c>ICascadeEvolutionTask&lt;TGenome&gt;</c> and the engine
    /// rejects the combination when it is constructed; switch it on yourself once your task has stages.
    /// <c>database.migration_rate</c> is a fraction of a population, which the whole-elite count
    /// <see cref="MigrantsPerIsland"/> cannot express, so that keeps its own default of 2. And upstream's
    /// <c>max_iterations</c> counts iterations rather than evaluator attempts, so a run that retries heavily
    /// exhausts <see cref="MaxEvaluationAttempts"/> before upstream would reach its iteration limit.
    /// </para>
    /// <para><b>For Beginners:</b> Our defaults are cautious on purpose: a first run should be cheap, single
    /// threaded, and free of surprises, so features such as islands and retries start switched off. OpenEvolve makes
    /// the opposite choice and turns most of them on. If you are reproducing an OpenEvolve experiment, or you simply
    /// want the fuller configuration without looking every number up, start from this factory and then change what
    /// you need: <c>var options = EvolutionEngineOptions.CreateOpenEvolveDefaults(); options.RunId = "my-run";</c>.
    /// The returned object is an ordinary options instance, so nothing is locked down.</para>
    /// </remarks>
    public static EvolutionEngineOptions CreateOpenEvolveDefaults() => new()
    {
        Seed = 42UL,
        MaxEvaluationAttempts = 10_000,
        MaxProposals = 10_000,
        MaxGenerations = 10_000,
        ProposalBatchSize = 1,
        MaxDegreeOfParallelism = 1,
        MaxRetries = 3,
        RetryOn = EvolutionRetryStatuses.Failed,
        RetryBaseDelay = TimeSpan.FromSeconds(1),
        RetryBackoffMultiplier = 1.0,
        EvaluationTimeout = TimeSpan.FromSeconds(300),
        CheckpointInterval = 100,
        IslandCount = 5,
        MigrationInterval = 50,
        MigrationTrigger = EvolutionMigrationTrigger.IslandGenerations,
        GlobalEliteCount = 100,
        HistorySize = 1_000,
        InspirationCount = 5,
        SelectionPolicy = EvolutionSelectionPolicyKind.Ratio,
        Selection = new EvolutionSelectionOptions
        {
            ExplorationRatio = 0.2,
            ExploitationRatio = 0.7,
            EliteRatio = 0.1,
            TopInspirationCount = 3,
            DiverseInspirationCount = 2
        },
        Artifacts = new EvolutionArtifactOptions { Enabled = true }
    };

    internal EvolutionEngineOptions SnapshotAndValidate()
    {
        Guard.NotNullOrWhiteSpace(RunId);
        if (MaxEvaluationAttempts < 0) throw new ArgumentOutOfRangeException(nameof(MaxEvaluationAttempts));
        if (MaxProposals < 0) throw new ArgumentOutOfRangeException(nameof(MaxProposals));
        if (MaxGenerations < 0) throw new ArgumentOutOfRangeException(nameof(MaxGenerations));
        Guard.Positive(ProposalBatchSize);
        Guard.Positive(MaxDegreeOfParallelism);
        if (!Enum.IsDefined(typeof(EvolutionExecutionMode), ExecutionMode)) throw new ArgumentOutOfRangeException(nameof(ExecutionMode));
        if (!Enum.IsDefined(typeof(EvolutionFailurePolicy), FailurePolicy)) throw new ArgumentOutOfRangeException(nameof(FailurePolicy));
        if (MaxRetries < 0) throw new ArgumentOutOfRangeException(nameof(MaxRetries));
        ValidateDuration(EvaluationTimeout, nameof(EvaluationTimeout));
        ValidateDuration(EvaluationGracePeriod, nameof(EvaluationGracePeriod));
        ValidateDuration(TimeLimit, nameof(TimeLimit));
        if (EvaluationGracePeriod.HasValue && !EvaluationTimeout.HasValue)
            throw new ArgumentException("An evaluation grace period requires an evaluation timeout.", nameof(EvaluationGracePeriod));
        if ((RetryOn & ~EvolutionRetryStatuses.All) != 0) throw new ArgumentOutOfRangeException(nameof(RetryOn));
        if (RetryBaseDelay < TimeSpan.Zero || RetryBaseDelay.TotalMilliseconds > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(RetryBaseDelay));
        if (!IsFinite(RetryBackoffMultiplier) || RetryBackoffMultiplier < 1.0)
            throw new ArgumentOutOfRangeException(nameof(RetryBackoffMultiplier),
                "The retry backoff multiplier must be a finite value of at least one.");
        if (TargetQuality.HasValue && !IsFinite(TargetQuality.Value))
            throw new ArgumentOutOfRangeException(nameof(TargetQuality), "The target quality must be finite.");
        if (CheckpointInterval < 0) throw new ArgumentOutOfRangeException(nameof(CheckpointInterval));
        Guard.Positive(IslandCount);
        if (MigrationInterval < 0) throw new ArgumentOutOfRangeException(nameof(MigrationInterval));
        Guard.Positive(MigrantsPerIsland);
        if (InspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(InspirationCount));
        Guard.Positive(MaxRetainedFailures);
        if (!Enum.IsDefined(typeof(EvolutionIslandAssignmentStrategy), IslandAssignment))
            throw new ArgumentOutOfRangeException(nameof(IslandAssignment));
        if (!Enum.IsDefined(typeof(EvolutionMigrationTrigger), MigrationTrigger))
            throw new ArgumentOutOfRangeException(nameof(MigrationTrigger));
        if (!Enum.IsDefined(typeof(EvolutionSelectionPolicyKind), SelectionPolicy))
            throw new ArgumentOutOfRangeException(nameof(SelectionPolicy));
        if (GlobalEliteCount < 0) throw new ArgumentOutOfRangeException(nameof(GlobalEliteCount));
        if (HistorySize < 0) throw new ArgumentOutOfRangeException(nameof(HistorySize));
        if (!IsFinite(NoveltyDistanceThreshold) || NoveltyDistanceThreshold < 0)
            throw new ArgumentOutOfRangeException(nameof(NoveltyDistanceThreshold));
        if (QualityDescriptorName is not null && string.IsNullOrWhiteSpace(QualityDescriptorName))
            throw new ArgumentException("The quality descriptor name cannot be empty.", nameof(QualityDescriptorName));
        string? outputDirectory = ResolveOutputDirectory();
        Guard.NotNull(Selection);
        Guard.NotNull(Cascade);
        Guard.NotNull(Artifacts);
        Guard.NotNull(EarlyStopping);
        EvolutionSelectionOptions selection = Selection.SnapshotAndValidate();
        EvolutionCascadeOptions cascade = Cascade.SnapshotAndValidate();
        EvolutionArtifactOptions artifacts = Artifacts.SnapshotAndValidate();
        EvolutionEarlyStoppingOptions earlyStopping = EarlyStopping.SnapshotAndValidate();

        return new EvolutionEngineOptions
        {
            Cascade = cascade,
            Artifacts = artifacts,
            EarlyStopping = earlyStopping,
            EvaluationGracePeriod = EvaluationGracePeriod,
            RetryOn = RetryOn,
            RetryBaseDelay = RetryBaseDelay,
            RetryBackoffMultiplier = RetryBackoffMultiplier,
            TargetQuality = TargetQuality,
            IslandAssignment = IslandAssignment,
            MigrationTrigger = MigrationTrigger,
            GlobalEliteCount = GlobalEliteCount,
            HistorySize = HistorySize,
            NoveltyDistanceThreshold = NoveltyDistanceThreshold,
            QualityDescriptorName = QualityDescriptorName?.Trim(),
            Selection = selection,
            SelectionPolicy = SelectionPolicy,
            RunId = RunId.Trim(),
            Seed = Seed,
            OutputDirectory = outputDirectory,
            MaxEvaluationAttempts = MaxEvaluationAttempts,
            MaxProposals = MaxProposals,
            MaxGenerations = MaxGenerations,
            ProposalBatchSize = ProposalBatchSize,
            MaxDegreeOfParallelism = MaxDegreeOfParallelism,
            ExecutionMode = ExecutionMode,
            FailurePolicy = FailurePolicy,
            MaxRetries = MaxRetries,
            EvaluationTimeout = EvaluationTimeout,
            TimeLimit = TimeLimit,
            CheckpointInterval = CheckpointInterval,
            Resume = Resume,
            EnableEvaluationCache = EnableEvaluationCache,
            DeduplicateFailedCandidates = DeduplicateFailedCandidates,
            IslandCount = IslandCount,
            MigrationInterval = MigrationInterval,
            MigrantsPerIsland = MigrantsPerIsland,
            InspirationCount = InspirationCount,
            MaxRetainedFailures = MaxRetainedFailures
        };
    }

    internal string ToCanonicalString() => string.Join("|", new[]
    {
        StableRandom.AlgorithmId,
        Seed.ToString(CultureInfo.InvariantCulture),
        MaxEvaluationAttempts.ToString(CultureInfo.InvariantCulture),
        MaxProposals.ToString(CultureInfo.InvariantCulture),
        MaxGenerations.ToString(CultureInfo.InvariantCulture),
        ProposalBatchSize.ToString(CultureInfo.InvariantCulture),
        ((int)ExecutionMode).ToString(CultureInfo.InvariantCulture),
        ((int)FailurePolicy).ToString(CultureInfo.InvariantCulture),
        MaxRetries.ToString(CultureInfo.InvariantCulture),
        EvaluationTimeout?.Ticks.ToString(CultureInfo.InvariantCulture) ?? "none",
        EnableEvaluationCache ? "cache" : "no-cache",
        DeduplicateFailedCandidates ? "dedup-failed" : "retry-failed",
        IslandCount.ToString(CultureInfo.InvariantCulture),
        MigrationInterval.ToString(CultureInfo.InvariantCulture),
        MigrantsPerIsland.ToString(CultureInfo.InvariantCulture),
        InspirationCount.ToString(CultureInfo.InvariantCulture),
        MaxRetainedFailures.ToString(CultureInfo.InvariantCulture),
        ((int)IslandAssignment).ToString(CultureInfo.InvariantCulture),
        ((int)MigrationTrigger).ToString(CultureInfo.InvariantCulture),
        GlobalEliteCount.ToString(CultureInfo.InvariantCulture),
        HistorySize.ToString(CultureInfo.InvariantCulture),
        NoveltyDistanceThreshold.ToString("R", CultureInfo.InvariantCulture),
        QualityDescriptorName is null ? "no-quality-descriptor" : "quality:" + QualityDescriptorName,
        Selection is null ? "default-selection" : Selection.ToCanonicalString(),
        Cascade is null ? "default-cascade" : Cascade.ToCanonicalString(),
        Artifacts is null ? "default-artifacts" : Artifacts.ToCanonicalString(),
        EarlyStopping is null ? "default-early-stopping" : EarlyStopping.ToCanonicalString(),
        EvaluationGracePeriod?.Ticks.ToString(CultureInfo.InvariantCulture) ?? "none",
        ((int)RetryOn).ToString(CultureInfo.InvariantCulture),
        RetryBaseDelay.Ticks.ToString(CultureInfo.InvariantCulture),
        RetryBackoffMultiplier.ToString("R", CultureInfo.InvariantCulture),
        TargetQuality?.ToString("R", CultureInfo.InvariantCulture) ?? "none"
    });

    /// <summary>Validates <see cref="OutputDirectory"/> and resolves it to an absolute path without creating it.</summary>
    /// <returns>The resolved directory, or <c>null</c> when none is configured.</returns>
    /// <exception cref="ArgumentException">The configured directory is blank or is not a valid path.</exception>
    private string? ResolveOutputDirectory()
    {
        if (OutputDirectory is null) return null;
        if (string.IsNullOrWhiteSpace(OutputDirectory))
            throw new ArgumentException("The output directory cannot be empty.", nameof(OutputDirectory));
        try
        {
            return Path.GetFullPath(OutputDirectory.Trim());
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException or PathTooLongException)
        {
            throw new ArgumentException("The output directory is not a valid path.", nameof(OutputDirectory), exception);
        }
    }

    private static void ValidateDuration(TimeSpan? duration, string parameterName)
    {
        if (!duration.HasValue) return;
        if (duration.Value <= TimeSpan.Zero || duration.Value.TotalMilliseconds > int.MaxValue)
            throw new ArgumentOutOfRangeException(parameterName,
                "Durations must be positive and within the cross-target cancellation timer range.");
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}
