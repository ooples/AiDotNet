namespace AiDotNet.Evolution;

/// <summary>Defines validation, canonical identity, and evaluation for a domain-specific genome.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionTask<TGenome>
{
    /// <summary>Gets a stable task identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash that changes whenever task semantics change.</summary>
    string VersionHash { get; }

    /// <summary>Gets a version hash that changes whenever evaluator semantics or data change.</summary>
    string EvaluatorVersionHash { get; }

    /// <summary>Validates, snapshots, and canonicalizes a proposed genome.</summary>
    ValueTask<EvolutionCanonicalGenome<TGenome>> CanonicalizeAsync(TGenome genome, CancellationToken cancellationToken = default);

    /// <summary>Evaluates one canonical candidate.</summary>
    ValueTask<EvolutionTaskResult> EvaluateAsync(
        EvolutionCandidate<TGenome> candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default);
}

/// <summary>Proposes mutation, crossover, or another variation without evaluator knowledge.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IVariationOperator<TGenome>
{
    /// <summary>Gets a stable operator identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Proposes a new genome.</summary>
    ValueTask<TGenome> ProposeAsync(EvolutionVariationContext<TGenome> context, CancellationToken cancellationToken = default);
}

/// <summary>Optionally improves a proposed genome while returning a new immutable snapshot.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface ICandidateRefiner<TGenome>
{
    /// <summary>Gets a stable refiner identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Returns a refined genome without modifying the input object.</summary>
    ValueTask<TGenome> RefineAsync(TGenome genome, EvolutionRefinementContext context, CancellationToken cancellationToken = default);
}

/// <summary>Selects parents and inspirations from an archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface ISelectionPolicy<TGenome>
{
    /// <summary>Gets a stable policy identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Selects a parent and optional inspiration set.</summary>
    EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount);
}

/// <summary>Optional state and outcome feedback implemented by adaptive selection policies.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IOutcomeAwareEvolutionSelectionPolicy<TGenome> : ISelectionPolicy<TGenome>
{
    /// <summary>Updates policy state after one evaluation is committed.</summary>
    void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult);

    /// <summary>Captures deterministic policy state for a checkpoint.</summary>
    string CaptureState();

    /// <summary>Restores deterministic policy state from a checkpoint.</summary>
    void RestoreState(string state);
}

/// <summary>Creates deterministic elite transfers between independent island archives.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IMigrationPolicy<TGenome>
{
    /// <summary>Gets a stable policy identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Creates transfers without modifying any archive.</summary>
    IReadOnlyList<EvolutionMigration<TGenome>> CreateMigrations(
        IReadOnlyList<IEvolutionArchiveView<TGenome>> islands,
        int migrantsPerIsland,
        StableRandom random);
}

/// <summary>Read-only view of a deterministic quality-diversity archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionArchiveView<TGenome>
{
    /// <summary>Gets immutable descriptor definitions.</summary>
    IReadOnlyList<EvolutionDescriptorDefinition> Descriptors { get; }

    /// <summary>Gets a stable hash of every archive policy that affects insertion or restoration.</summary>
    string DefinitionHash { get; }

    /// <summary>Gets the configured optimization direction.</summary>
    EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets the current number of occupied cells.</summary>
    int Count { get; }

    /// <summary>Gets a monotonically increasing change version.</summary>
    long Version { get; }

    /// <summary>Gets elites in stable cell-key order.</summary>
    IReadOnlyList<EvolutionArchiveEntry<TGenome>> Entries { get; }

    /// <summary>Gets the globally best elite with deterministic tie-breaking.</summary>
    EvolutionArchiveEntry<TGenome>? Best { get; }

    /// <summary>Returns the elite in a specific cell, or <c>null</c>.</summary>
    EvolutionArchiveEntry<TGenome>? Get(EvolutionCellKey cell);

}

/// <summary>Mutable engine-owned quality-diversity archive contract.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionArchive<TGenome> : IEvolutionArchiveView<TGenome>
{
    /// <summary>Attempts to insert a completed candidate evaluation.</summary>
    EvolutionArchiveInsertionResult TryAdd(EvolutionCandidate<TGenome> candidate, EvolutionEvaluation evaluation);

    /// <summary>Samples uniformly from occupied cells using a caller-owned stable stream.</summary>
    EvolutionArchiveEntry<TGenome>? Sample(StableRandom random);
}

/// <summary>Allows an archive implementation to restore an exact versioned checkpoint snapshot.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface ICheckpointableEvolutionArchive<TGenome> : IEvolutionArchive<TGenome>
{
    /// <summary>Restores entries and the exact archive version into an empty archive.</summary>
    void Restore(IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries, long version);
}

/// <summary>Receives structured progress without coupling the engine to a console or logging provider.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionObserver<TGenome>
{
    /// <summary>Observes one immutable event.</summary>
    ValueTask OnEventAsync(EvolutionEvent<TGenome> evolutionEvent, CancellationToken cancellationToken = default);
}

/// <summary>Serializes task genomes for portable checkpoints.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionGenomeCodec<TGenome>
{
    /// <summary>Gets a stable codec identifier.</summary>
    string Id { get; }

    /// <summary>Gets a codec version hash.</summary>
    string VersionHash { get; }

    /// <summary>Serializes an immutable genome snapshot.</summary>
    string Serialize(TGenome genome);

    /// <summary>Deserializes an immutable genome snapshot.</summary>
    TGenome Deserialize(string payload);
}

/// <summary>Persists and loads versioned opaque evolution checkpoints.</summary>
public interface IEvolutionCheckpointStore
{
    /// <summary>Atomically saves the latest checkpoint for its run.</summary>
    Task SaveAsync(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken = default);

    /// <summary>Loads the latest valid checkpoint for a run, or <c>null</c> when none exists.</summary>
    Task<EvolutionCheckpoint?> LoadLatestAsync(string runId, CancellationToken cancellationToken = default);
}

/// <summary>Deterministic evaluator context for one candidate.</summary>
public sealed class EvolutionEvaluationContext
{
    /// <summary>Initializes an evaluator context.</summary>
    public EvolutionEvaluationContext(long evaluationId, ulong rootSeed, ulong seedStream, int attemptCount)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        if (attemptCount <= 0) throw new ArgumentOutOfRangeException(nameof(attemptCount));
        EvaluationId = evaluationId;
        RootSeed = rootSeed;
        SeedStream = seedStream;
        AttemptCount = attemptCount;
    }

    /// <summary>Gets the evaluation ID.</summary>
    public long EvaluationId { get; }
    /// <summary>Gets the run root seed.</summary>
    public ulong RootSeed { get; }
    /// <summary>Gets the stable stream identifier.</summary>
    public ulong SeedStream { get; }
    /// <summary>Gets the one-based attempt count.</summary>
    public int AttemptCount { get; }

    /// <summary>Creates a fresh task-local stream whose sequence is independent of worker scheduling.</summary>
    public StableRandom CreateRandom() => StableRandom.CreateStream(RootSeed, SeedStream);
}

/// <summary>Inputs available to one variation proposal.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionVariationContext<TGenome>
{
    /// <summary>Initializes a variation context.</summary>
    public EvolutionVariationContext(
        EvolutionArchiveEntry<TGenome> parent,
        IReadOnlyList<EvolutionArchiveEntry<TGenome>> inspirations,
        StableRandom random,
        long generation,
        int island)
    {
        Parent = parent ?? throw new ArgumentNullException(nameof(parent));
        Inspirations = inspirations ?? throw new ArgumentNullException(nameof(inspirations));
        Random = random ?? throw new ArgumentNullException(nameof(random));
        if (generation < 0) throw new ArgumentOutOfRangeException(nameof(generation));
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        Generation = generation;
        Island = island;
    }

    /// <summary>Gets the selected parent.</summary>
    public EvolutionArchiveEntry<TGenome> Parent { get; }
    /// <summary>Gets selected inspiration elites.</summary>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Inspirations { get; }
    /// <summary>Gets a proposal-local stable random stream.</summary>
    public StableRandom Random { get; }
    /// <summary>Gets the logical generation.</summary>
    public long Generation { get; }
    /// <summary>Gets the target island.</summary>
    public int Island { get; }
}

/// <summary>Inputs available to an optional candidate refiner.</summary>
public sealed class EvolutionRefinementContext
{
    /// <summary>Initializes a refinement context.</summary>
    public EvolutionRefinementContext(long evaluationId, StableRandom random)
    {
        if (evaluationId < 0) throw new ArgumentOutOfRangeException(nameof(evaluationId));
        EvaluationId = evaluationId;
        Random = random ?? throw new ArgumentNullException(nameof(random));
    }

    /// <summary>Gets the assigned evaluation ID.</summary>
    public long EvaluationId { get; }
    /// <summary>Gets a refiner-local stable random stream.</summary>
    public StableRandom Random { get; }
}

/// <summary>A selected parent and its inspiration elites.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionSelection<TGenome>
{
    /// <summary>Initializes a selection.</summary>
    public EvolutionSelection(EvolutionArchiveEntry<TGenome> parent, IReadOnlyList<EvolutionArchiveEntry<TGenome>> inspirations)
    {
        Parent = parent ?? throw new ArgumentNullException(nameof(parent));
        Inspirations = inspirations ?? throw new ArgumentNullException(nameof(inspirations));
    }

    /// <summary>Gets the selected parent.</summary>
    public EvolutionArchiveEntry<TGenome> Parent { get; }
    /// <summary>Gets selected inspirations.</summary>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Inspirations { get; }
}

/// <summary>One immutable elite transfer between islands.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionMigration<TGenome>
{
    /// <summary>Initializes a migration transfer.</summary>
    public EvolutionMigration(int sourceIsland, int destinationIsland, EvolutionArchiveEntry<TGenome> entry)
    {
        if (sourceIsland < 0) throw new ArgumentOutOfRangeException(nameof(sourceIsland));
        if (destinationIsland < 0) throw new ArgumentOutOfRangeException(nameof(destinationIsland));
        if (sourceIsland == destinationIsland) throw new ArgumentException("Migration requires distinct islands.", nameof(destinationIsland));
        SourceIsland = sourceIsland;
        DestinationIsland = destinationIsland;
        Entry = entry ?? throw new ArgumentNullException(nameof(entry));
    }

    /// <summary>Gets the source island.</summary>
    public int SourceIsland { get; }
    /// <summary>Gets the destination island.</summary>
    public int DestinationIsland { get; }
    /// <summary>Gets the immutable elite to copy.</summary>
    public EvolutionArchiveEntry<TGenome> Entry { get; }
}

/// <summary>One structured observer notification.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionEvent<TGenome>
{
    /// <summary>Initializes an observer event.</summary>
    public EvolutionEvent(EvolutionEventKind kind, long sequence, EvolutionCandidate<TGenome>? candidate = null,
        EvolutionEvaluation? evaluation = null, EvolutionArchiveInsertionResult? insertionResult = null, string? message = null)
    {
        if (sequence < 0) throw new ArgumentOutOfRangeException(nameof(sequence));
        Kind = kind;
        Sequence = sequence;
        Candidate = candidate;
        Evaluation = evaluation;
        InsertionResult = insertionResult;
        Message = message;
    }

    /// <summary>Gets the event kind.</summary>
    public EvolutionEventKind Kind { get; }
    /// <summary>Gets the monotonically increasing event sequence.</summary>
    public long Sequence { get; }
    /// <summary>Gets the related candidate, when applicable.</summary>
    public EvolutionCandidate<TGenome>? Candidate { get; }
    /// <summary>Gets the related evaluation, when applicable.</summary>
    public EvolutionEvaluation? Evaluation { get; }
    /// <summary>Gets the archive insertion result, when applicable.</summary>
    public EvolutionArchiveInsertionResult? InsertionResult { get; }
    /// <summary>Gets an optional bounded message.</summary>
    public string? Message { get; }
}
