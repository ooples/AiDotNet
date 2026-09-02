using System.Globalization;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>Configures deterministic orchestration, budgets, parallelism, islands, and persistence.</summary>
/// <remarks>
/// <para>
/// <see cref="EvolutionEngine{TGenome}"/> validates and defensively copies these options in its constructor, so
/// mutating the original object afterwards has no effect on a running engine. The options divide into two groups.
/// Semantic options change what the search means (seed, batch size, execution mode, failure policy, retries,
/// evaluation timeout, caching, deduplication, islands, migration topology, rate and schedule, inspirations, the
/// failure-retention bound, island assignment, the global elite and history bounds, the novelty threshold, the
/// derived quality descriptor, the selection ratios, the cascade, artifact, early-stopping and retry settings, the
/// evaluation grace period, and the target quality); they are folded into the engine's configuration and
/// compatibility hashes, and a checkpoint written under one set refuses to resume under another, naming the option
/// that differs.
/// </para>
/// <para>
/// Budget options only bound or locate a run: <see cref="MaxEvaluationAttempts"/>, <see cref="MaxProposals"/>,
/// <see cref="MaxGenerations"/>, <see cref="TimeLimit"/>, <see cref="CheckpointInterval"/>, <see cref="Resume"/>,
/// <see cref="RunId"/>, <see cref="MaxDegreeOfParallelism"/>, and <see cref="OutputDirectory"/>. They are recorded in
/// the checkpoint for provenance but never compared, so a run that stopped at its limit can be continued simply by
/// raising that limit, which is the most common thing to want after a search ends. Lowering a limit below what the run
/// already spent is legal too: the resumed run restores its counters and stops immediately with the matching budget
/// stop reason. Excluding <see cref="OutputDirectory"/> matters especially: it resolves to a machine-specific absolute
/// path, so hashing it would make a run's configuration hash differ between machines and stop a checkpoint from moving
/// with its output folder. OpenEvolve compares nothing at all on resume and will happily continue a checkpoint under
/// incompatible settings (<c>controller.py</c>).
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

    /// <summary>Gets or sets the maximum elites one island copies to any single destination during migration.</summary>
    /// <remarks>
    /// Under the default <see cref="EvolutionMigrationTopology.Ring"/> each island has one destination, so this is also
    /// the number of elites that leave an island per round. A broadcasting topology sends this many to each of its
    /// destinations, and the engine bounds the whole round by the ordered island pairs times this value.
    /// </remarks>
    public int MigrantsPerIsland { get; set; } = 2;

    /// <summary>Gets or sets which islands a migration round copies elites between.</summary>
    /// <remarks>
    /// The default <see cref="EvolutionMigrationTopology.Ring"/> reproduces the historical behaviour exactly: each
    /// island feeds the next one only. Denser topologies emit up to <c>IslandCount - 1</c> destinations per source, so
    /// the engine bounds a round by the ordered island pairs rather than by <see cref="MigrantsPerIsland"/> alone. This
    /// option configures the engine's built-in policy; supplying an explicit <c>IMigrationPolicy</c> to the engine
    /// overrides it, because that policy then decides every transfer itself.
    /// </remarks>
    public EvolutionMigrationTopology MigrationTopology { get; set; } = EvolutionMigrationTopology.Ring;

    /// <summary>Gets or sets the fraction of a source island's elites that migrate; zero uses <see cref="MigrantsPerIsland"/>.</summary>
    /// <remarks>
    /// A positive rate resolves to <c>max(1, floor(eliteCount * rate))</c> capped at <see cref="MigrantsPerIsland"/>, so
    /// the number of travellers grows with a filling island but can never exceed the bound you configured. Zero, the
    /// default, keeps the fixed per-island count. OpenEvolve uses a rate of <c>0.1</c> with no upper bound
    /// (<c>config.py</c> <c>migration_rate</c>).
    /// </remarks>
    public double MigrationRate { get; set; }

    /// <summary>Gets or sets whether elites that arrived by an earlier migration are skipped as migration sources.</summary>
    /// <remarks>
    /// <see langword="false"/>, the default, migrates the best elites whatever their origin. Setting it reproduces
    /// OpenEvolve's hard-coded guard against re-migrating an already-migrated program, which it added to stop
    /// identical copies multiplying across islands (<c>database.py</c> <c>migrate_programs</c>). It is optional here
    /// because a MAP-Elites archive already keeps one entry per cell, so the copies OpenEvolve feared are usually
    /// rejected on arrival rather than accumulated.
    /// </remarks>
    public bool PreventRepeatedMigration { get; set; }

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
        if (!Enum.IsDefined(typeof(EvolutionMigrationTopology), MigrationTopology))
            throw new ArgumentOutOfRangeException(nameof(MigrationTopology));
        if (!IsFinite(MigrationRate) || MigrationRate < 0 || MigrationRate > 1)
            throw new ArgumentOutOfRangeException(nameof(MigrationRate),
                "The migration rate must be a finite fraction between zero and one.");
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
            MigrationTopology = MigrationTopology,
            MigrationRate = MigrationRate,
            PreventRepeatedMigration = PreventRepeatedMigration,
            InspirationCount = InspirationCount,
            MaxRetainedFailures = MaxRetainedFailures
        };
    }

    /// <summary>Lists every option that changes what the search means, as ordered name/value pairs.</summary>
    /// <remarks>
    /// These are exactly the settings a checkpoint compares on resume: changing one of them would make the restored
    /// state describe a different search, so continuing under the new value would corrupt it. Budget settings are
    /// deliberately absent, which is what lets a finished run be resumed with a raised limit. The pairs are named so
    /// that a rejected resume can say which option differs instead of only reporting a hash mismatch.
    /// </remarks>
    internal IReadOnlyList<KeyValuePair<string, string>> SemanticFields() => new[]
    {
        Field("random-algorithm", StableRandom.AlgorithmId),
        Field("seed", Seed.ToString(CultureInfo.InvariantCulture)),
        Field("proposal-batch-size", ProposalBatchSize.ToString(CultureInfo.InvariantCulture)),
        Field("execution-mode", ((int)ExecutionMode).ToString(CultureInfo.InvariantCulture)),
        Field("failure-policy", ((int)FailurePolicy).ToString(CultureInfo.InvariantCulture)),
        Field("max-retries", MaxRetries.ToString(CultureInfo.InvariantCulture)),
        Field("evaluation-timeout", EvaluationTimeout?.Ticks.ToString(CultureInfo.InvariantCulture) ?? "none"),
        Field("evaluation-cache", EnableEvaluationCache ? "cache" : "no-cache"),
        Field("failed-candidate-dedup", DeduplicateFailedCandidates ? "dedup-failed" : "retry-failed"),
        Field("island-count", IslandCount.ToString(CultureInfo.InvariantCulture)),
        Field("migration-interval", MigrationInterval.ToString(CultureInfo.InvariantCulture)),
        Field("migrants-per-island", MigrantsPerIsland.ToString(CultureInfo.InvariantCulture)),
        Field("migration-topology", ((int)MigrationTopology).ToString(CultureInfo.InvariantCulture)),
        Field("migration-rate", MigrationRate.ToString("R", CultureInfo.InvariantCulture)),
        Field("prevent-repeated-migration", PreventRepeatedMigration ? "once" : "repeatable"),
        Field("inspiration-count", InspirationCount.ToString(CultureInfo.InvariantCulture)),
        Field("max-retained-failures", MaxRetainedFailures.ToString(CultureInfo.InvariantCulture)),
        Field("island-assignment", ((int)IslandAssignment).ToString(CultureInfo.InvariantCulture)),
        Field("migration-trigger", ((int)MigrationTrigger).ToString(CultureInfo.InvariantCulture)),
        Field("global-elite-count", GlobalEliteCount.ToString(CultureInfo.InvariantCulture)),
        Field("history-size", HistorySize.ToString(CultureInfo.InvariantCulture)),
        Field("novelty-distance-threshold", NoveltyDistanceThreshold.ToString("R", CultureInfo.InvariantCulture)),
        Field("quality-descriptor", QualityDescriptorName ?? "none"),
        Field("selection", Selection is null ? "default-selection" : Selection.ToCanonicalString()),
        Field("cascade", Cascade is null ? "default-cascade" : Cascade.ToCanonicalString()),
        Field("artifacts", Artifacts is null ? "default-artifacts" : Artifacts.ToCanonicalString()),
        Field("early-stopping", EarlyStopping is null ? "default-early-stopping" : EarlyStopping.ToCanonicalString()),
        Field("evaluation-grace-period", EvaluationGracePeriod?.Ticks.ToString(CultureInfo.InvariantCulture) ?? "none"),
        Field("retry-on", ((int)RetryOn).ToString(CultureInfo.InvariantCulture)),
        Field("retry-base-delay", RetryBaseDelay.Ticks.ToString(CultureInfo.InvariantCulture)),
        Field("retry-backoff-multiplier", RetryBackoffMultiplier.ToString("R", CultureInfo.InvariantCulture)),
        Field("target-quality", TargetQuality?.ToString("R", CultureInfo.InvariantCulture) ?? "none")
    };

    /// <summary>Lists every option that only bounds or locates a run, as ordered name/value pairs.</summary>
    /// <remarks>
    /// A checkpoint records these for provenance and never compares them, so a completed run can be continued simply by
    /// raising a limit. Lowering a limit below what a run already spent is legal too: the resumed run restores its
    /// counters and stops immediately with the matching budget stop reason.
    /// </remarks>
    internal IReadOnlyList<KeyValuePair<string, string>> BudgetFields() => new[]
    {
        Field("run-id", RunId),
        Field("max-evaluation-attempts", MaxEvaluationAttempts.ToString(CultureInfo.InvariantCulture)),
        Field("max-proposals", MaxProposals.ToString(CultureInfo.InvariantCulture)),
        Field("max-generations", MaxGenerations.ToString(CultureInfo.InvariantCulture)),
        Field("time-limit", TimeLimit?.Ticks.ToString(CultureInfo.InvariantCulture) ?? "none"),
        Field("checkpoint-interval", CheckpointInterval.ToString(CultureInfo.InvariantCulture)),
        Field("resume", Resume ? "resume" : "fresh"),
        Field("max-degree-of-parallelism", MaxDegreeOfParallelism.ToString(CultureInfo.InvariantCulture)),
        Field("output-directory", OutputDirectory ?? "none")
    };

    /// <summary>Encodes the semantic options into the string the configuration hash is computed from.</summary>
    internal string ToSemanticCanonicalString() => Encode(SemanticFields());

    /// <summary>Encodes the budget options into the provenance string stored in a checkpoint.</summary>
    internal string ToBudgetCanonicalString() => Encode(BudgetFields());

    /// <summary>Names the first semantic option whose recorded value differs from the current one.</summary>
    /// <param name="recorded">The name/value pairs read back from a checkpoint.</param>
    /// <returns>A description naming the option, or <c>null</c> when every recorded option matches.</returns>
    internal string? DescribeSemanticDifference(IReadOnlyList<KeyValuePair<string, string>> recorded)
    {
        IReadOnlyList<KeyValuePair<string, string>> current = SemanticFields();
        var currentByName = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> field in current) currentByName[field.Key] = field.Value;
        foreach (KeyValuePair<string, string> field in recorded)
        {
            if (!currentByName.TryGetValue(field.Key, out string? value))
            {
                return string.Format(CultureInfo.InvariantCulture,
                    "the checkpoint records the option '{0}', which this engine no longer defines", field.Key);
            }
            if (!string.Equals(value, field.Value, StringComparison.Ordinal))
            {
                return string.Format(CultureInfo.InvariantCulture,
                    "the option '{0}' changed from '{1}' to '{2}'", field.Key, field.Value, value);
            }
        }
        var recordedNames = new HashSet<string>(recorded.Select(field => field.Key), StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> field in current)
        {
            if (!recordedNames.Contains(field.Key))
            {
                return string.Format(CultureInfo.InvariantCulture,
                    "this engine defines the option '{0}', which the checkpoint does not record", field.Key);
            }
        }
        return null;
    }

    private static KeyValuePair<string, string> Field(string name, string value) => new(name, value);

    private static string Encode(IReadOnlyList<KeyValuePair<string, string>> fields)
    {
        var builder = new StringBuilder();
        foreach (KeyValuePair<string, string> field in fields)
        {
            builder.Append(field.Key.Length.ToString(CultureInfo.InvariantCulture)).Append(':').Append(field.Key)
                .Append('=')
                .Append(field.Value.Length.ToString(CultureInfo.InvariantCulture)).Append(':').Append(field.Value)
                .Append(';');
        }
        return builder.ToString();
    }

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
