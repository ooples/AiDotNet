using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>Configures deterministic orchestration, budgets, parallelism, islands, and persistence.</summary>
public sealed class EvolutionEngineOptions
{
    /// <summary>Gets or sets the stable run identifier used by checkpoint stores.</summary>
    public string RunId { get; set; } = "default";

    /// <summary>Gets or sets the root seed used to derive candidate-local streams.</summary>
    public ulong Seed { get; set; } = 1234UL;

    /// <summary>Gets or sets the maximum number of actual evaluator attempts, including retries.</summary>
    public int MaxEvaluationAttempts { get; set; } = 100;

    /// <summary>Gets or sets the maximum number of proposals, including duplicates and rejected proposals.</summary>
    public int MaxProposals { get; set; } = 1_000;

    /// <summary>Gets or sets the maximum number of non-seed variation proposals.</summary>
    public int MaxGenerations { get; set; } = 1_000;

    /// <summary>Gets or sets the deterministic logical batch size, independent of worker count.</summary>
    public int ProposalBatchSize { get; set; } = 8;

    /// <summary>Gets or sets the maximum number of concurrently running evaluator calls.</summary>
    public int MaxDegreeOfParallelism { get; set; } = 1;

    /// <summary>Gets or sets deterministic or opportunistic commit behavior.</summary>
    public EvolutionExecutionMode ExecutionMode { get; set; } = EvolutionExecutionMode.Deterministic;

    /// <summary>Gets or sets whether recoverable candidate failures stop the run.</summary>
    public EvolutionFailurePolicy FailurePolicy { get; set; } = EvolutionFailurePolicy.Continue;

    /// <summary>Gets or sets the maximum retries after the first evaluator attempt.</summary>
    public int MaxRetries { get; set; }

    /// <summary>Gets or sets an optional cooperative timeout per evaluator attempt.</summary>
    public TimeSpan? EvaluationTimeout { get; set; }

    /// <summary>Gets or sets an optional wall-clock limit for the overall run.</summary>
    public TimeSpan? TimeLimit { get; set; }

    /// <summary>Gets or sets the number of committed evaluations between checkpoints; zero disables periodic saves.</summary>
    public int CheckpointInterval { get; set; }

    /// <summary>Gets or sets whether a compatible checkpoint is loaded before proposing new candidates.</summary>
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
        ValidateDuration(TimeLimit, nameof(TimeLimit));
        if (CheckpointInterval < 0) throw new ArgumentOutOfRangeException(nameof(CheckpointInterval));
        Guard.Positive(IslandCount);
        if (MigrationInterval < 0) throw new ArgumentOutOfRangeException(nameof(MigrationInterval));
        Guard.Positive(MigrantsPerIsland);
        if (InspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(InspirationCount));
        Guard.Positive(MaxRetainedFailures);

        return new EvolutionEngineOptions
        {
            RunId = RunId.Trim(),
            Seed = Seed,
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
        MaxRetainedFailures.ToString(CultureInfo.InvariantCulture)
    });

    private static void ValidateDuration(TimeSpan? duration, string parameterName)
    {
        if (!duration.HasValue) return;
        if (duration.Value <= TimeSpan.Zero || duration.Value.TotalMilliseconds > int.MaxValue)
            throw new ArgumentOutOfRangeException(parameterName,
                "Durations must be positive and within the cross-target cancellation timer range.");
    }
}
