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
    /// <remarks>
    /// <para>
    /// A batch is the engine's unit of commit and of checkpointing, so the batch size fixes how much work a crash can
    /// cost and how coarsely a run can be resumed, independently of how many evaluations run at once. It also sets a
    /// boundary condition on resume determinism worth knowing before choosing a large value. The engine stops filling
    /// a batch as soon as the proposals already in it would exhaust <see cref="MaxEvaluationAttempts"/>, which avoids
    /// paying for evaluations the budget will not allow. A run stopped by a smaller budget therefore ends on a
    /// truncated batch, and the resumed run under a larger budget starts a fresh full batch from there, so the two
    /// legs together propose a slightly different sequence from one uninterrupted run and reach a different
    /// <c>StateHash</c> even though both are internally deterministic and reproduce themselves exactly. That is
    /// visible only when the first leg's budget lands mid-batch: with a batch size of 1 every stop is a batch
    /// boundary, so a resumed run always reproduces the uninterrupted hash.
    /// </para>
    /// <para><b>For Beginners:</b> Larger batches mean fewer commits and less checkpoint overhead; smaller batches
    /// mean finer-grained saving and a resume that matches an uninterrupted run exactly. Leave this alone unless you
    /// are comparing a resumed run against an uninterrupted one, in which case use 1.</para>
    /// </remarks>
    public int ProposalBatchSize { get; set; } = 8;

    /// <summary>Gets or sets the maximum number of concurrently running evaluator calls.</summary>
    /// <remarks>
    /// Excluded from the compatibility hash; in <see cref="EvolutionExecutionMode.Deterministic"/> mode the result
    /// is independent of this value.
    /// </remarks>
    public int MaxDegreeOfParallelism { get; set; } = 1;

    /// <summary>Gets or sets deterministic or opportunistic commit behavior.</summary>
    public EvolutionExecutionMode ExecutionMode { get; set; } = EvolutionExecutionMode.Deterministic;

    /// <summary>Gets or sets whether evaluations run in fixed batches or in a continuously refilled window.</summary>
    /// <remarks>
    /// <para>
    /// <see cref="EvolutionDispatchMode.Batch"/>, the default, keeps the historical behaviour: fill a batch of
    /// <see cref="ProposalBatchSize"/> proposals, evaluate them together, commit them as one transaction.
    /// <see cref="EvolutionDispatchMode.Continuous"/> instead keeps <see cref="MaxInFlight"/> evaluations running,
    /// committing each one as it finishes and dispatching a replacement immediately, which is what the reference
    /// implementation does and what keeps workers busy when candidates differ widely in cost.
    /// </para>
    /// <para>
    /// The mode changes what a run costs in wall-clock time, never what it produces: under
    /// <see cref="EvolutionExecutionMode.Deterministic"/> both modes commit in evaluation-id order, and continuous
    /// dispatch prepares the proposal for one evaluation only after the evaluation one window earlier has committed,
    /// so the schedule is a function of the identifier sequence rather than of timing. Two continuous runs with the
    /// same seed and options therefore agree on the state hash at any worker count. The two modes do not produce the
    /// same hash as each other, because a proposal sees a different amount of the archive, which is exactly the
    /// staleness continuous dispatch exists to reduce; the mode is part of the compatibility hash for that reason.
    /// </para>
    /// <para><b>For Beginners:</b> Leave this alone until worker utilisation matters. Switch to
    /// <see cref="EvolutionDispatchMode.Continuous"/> when some candidates take far longer to score than others, or
    /// when you raise <see cref="MaxDegreeOfParallelism"/> and want every worker fed.</para>
    /// </remarks>
    public EvolutionDispatchMode Dispatch { get; set; } = EvolutionDispatchMode.Batch;

    /// <summary>Gets or sets how many evaluations may be in flight at once; zero follows the worker count.</summary>
    /// <remarks>
    /// <para>
    /// Used only by <see cref="EvolutionDispatchMode.Continuous"/>. Zero means the window is
    /// <see cref="MaxDegreeOfParallelism"/>, which is the natural choice: exactly enough work to keep every worker
    /// busy and no more. A larger window absorbs bursts of quick failures without draining, at the cost of proposing
    /// from an archive that is further behind. Because the window size decides which committed evaluation a proposal
    /// is prepared after, it changes results and is part of the compatibility hash.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many candidates are being scored at the same time. Leave it at zero.</para>
    /// </remarks>
    public int MaxInFlight { get; set; }

    /// <summary>Returns the window continuous dispatch actually uses, resolving zero to the worker count.</summary>
    /// <returns>The number of evaluations that may be in flight at once.</returns>
    /// <remarks>
    /// The resolved value, not the raw setting, is what the compatibility hash records. Leaving this at zero makes
    /// the window follow <see cref="MaxDegreeOfParallelism"/>, which is otherwise a budget setting a resume may
    /// change freely; hashing the raw zero would let that change the window, and with it the results, without the
    /// resume noticing.
    /// </remarks>
    internal int ResolveInFlightWindow() =>
        Math.Max(1, MaxInFlight > 0 ? MaxInFlight : MaxDegreeOfParallelism);

    /// <summary>Gets or sets how many evaluations one island may have in flight; zero means no per-island limit.</summary>
    /// <remarks>
    /// <para>
    /// Used only by <see cref="EvolutionDispatchMode.Continuous"/>. Without a per-island cap, one island whose
    /// candidates happen to be slow can occupy the whole window, so the other islands stop advancing and the
    /// separation that islands exist to provide quietly disappears. Setting this to roughly the window size divided
    /// by <see cref="IslandCount"/> keeps every island advancing.
    /// </para>
    /// <para><b>For Beginners:</b> Islands are separate sub-populations that explore independently. This stops one of
    /// them from hogging all the workers. Zero, the default, means no limit.</para>
    /// </remarks>
    public int MaxInFlightPerIsland { get; set; }

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
    /// (<c>evaluator.parallel_evaluations</c>, line 389), <see cref="ProposalBatchSize"/> is 1, and
    /// <see cref="Dispatch"/> is <see cref="EvolutionDispatchMode.Continuous"/> so each candidate is committed before
    /// the next parent is drawn and a freed worker is refilled immediately, which is how upstream's controller works
    /// (process_parallel.py:588-602); the window follows <see cref="MaxDegreeOfParallelism"/>, so raising the worker
    /// count on these defaults keeps that refill behaviour instead of reverting to a barrier;
    /// <see cref="InspirationCount"/> is 5 with three top and two diverse picks
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
        Dispatch = EvolutionDispatchMode.Continuous,
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
        if (!Enum.IsDefined(typeof(EvolutionDispatchMode), Dispatch)) throw new ArgumentOutOfRangeException(nameof(Dispatch));
        if (MaxInFlight < 0) throw new ArgumentOutOfRangeException(nameof(MaxInFlight));
        if (MaxInFlightPerIsland < 0) throw new ArgumentOutOfRangeException(nameof(MaxInFlightPerIsland));
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

        EvolutionEngineOptions snapshot = Copy();
        snapshot.Cascade = cascade;
        snapshot.Artifacts = artifacts;
        snapshot.EarlyStopping = earlyStopping;
        snapshot.Selection = selection;
        snapshot.RunId = RunId.Trim();
        snapshot.OutputDirectory = outputDirectory;
        snapshot.QualityDescriptorName = QualityDescriptorName?.Trim();
        return snapshot;
    }

    /// <summary>Copies every option without validating any of them.</summary>
    /// <returns>An independent instance carrying the same values, with each nested subsystem deep-copied.</returns>
    /// <remarks>
    /// This is the single place that enumerates the options, so a new one cannot be forgotten by a second
    /// hand-maintained copy elsewhere. <see cref="SnapshotAndValidate"/> builds on it and then substitutes the
    /// validated nested subsystems. Before this existed a separate copy in
    /// <see cref="ProgramEvolutionOptions"/> silently dropped 19 of the 41 options, so a program-evolution run
    /// discarded its cascade, early stopping, target quality, migration topology, selection policy and output
    /// directory without any error.
    /// </remarks>
    internal EvolutionEngineOptions Copy()
    {
        return new EvolutionEngineOptions
        {
            Cascade = Cascade.SnapshotAndValidate(),
            Artifacts = Artifacts.SnapshotAndValidate(),
            EarlyStopping = EarlyStopping.SnapshotAndValidate(),
            Selection = Selection.SnapshotAndValidate(),
            QualityDescriptorName = QualityDescriptorName,
            OutputDirectory = OutputDirectory,
            RunId = RunId,
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
            SelectionPolicy = SelectionPolicy,
            Seed = Seed,
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
        Field("dispatch", ((int)Dispatch).ToString(CultureInfo.InvariantCulture)),
        // The RESOLVED window, not the raw setting. A window of zero defers to the worker count, which is a budget
        // field and is never compared, so hashing the raw zero would let a resume quietly change the window - and the
        // window decides which committed evaluation a proposal is prepared after. Batch dispatch reads neither
        // setting, so both collapse to a constant there rather than refusing a resume over a value nothing uses.
        Field("in-flight-window", Dispatch == EvolutionDispatchMode.Continuous
            ? ResolveInFlightWindow().ToString(CultureInfo.InvariantCulture)
            : "batch"),
        Field("max-in-flight-per-island", Dispatch == EvolutionDispatchMode.Continuous
            ? MaxInFlightPerIsland.ToString(CultureInfo.InvariantCulture)
            : "batch"),
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
