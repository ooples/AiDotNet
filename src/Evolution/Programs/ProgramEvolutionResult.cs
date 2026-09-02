using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>The reportable summary of a finished program-evolution run.</summary>
/// <remarks>
/// <para>
/// <see cref="EvolutionRunResult{TGenome}"/> is the engine's own result and holds live archive snapshots of every
/// island. This type is the flat, bounded projection of it that a caller can log, serialize, or hang off a model
/// result: the best program, the top elites with their coordinates and a size-capped copy of their source, the
/// language-model bill, the run counters, and where the checkpoint was written. Building it never re-reads the
/// archives, so it stays valid after the engine is disposed.
/// </para>
/// <para>
/// Ordering is deterministic and direction-aware: elites are sorted by quality in the archive's optimisation
/// direction, unscored entries last, ties broken by genome identity, so the same run produces the same list on every
/// machine. <see cref="ArchiveCount"/> is the number of filled cells across every island — the search's coverage —
/// while <see cref="Elites"/> holds only the top <c>includeEliteSourceCount</c> of them, because a full archive can
/// hold thousands of programs.
/// </para>
/// <para><b>For Beginners:</b> This is the report card for a run. It answers the four questions you actually ask
/// afterwards: what is the best program found, what other good and different programs were found, how much did the
/// AI calls cost, and can I resume this later. It deliberately keeps only a capped amount of each program's text,
/// because these summaries end up in logs and a generated program can be arbitrarily long. Use
/// <see cref="BestProgram"/> for the winner and <see cref="Elites"/> to see the alternatives it beat.</para>
/// </remarks>
public sealed class ProgramEvolutionResult
{
    /// <summary>The default number of elites whose bounded source is retained.</summary>
    public const int DefaultEliteCount = 10;

    /// <summary>The default per-elite source bound, in characters.</summary>
    public const int DefaultEliteSourceChars = 4_000;

    private static readonly ReadOnlyDictionary<string, double> NoDescriptors =
        new(new Dictionary<string, double>(StringComparer.Ordinal));

    private readonly ReadOnlyCollection<ProgramEvolutionElite> _elites;
    private readonly ReadOnlyDictionary<string, double> _bestDescriptors;

    /// <summary>Initializes a run summary.</summary>
    /// <param name="stopReason">Why the run stopped.</param>
    /// <param name="stateHash">The engine's deterministic state hash.</param>
    /// <param name="counters">The engine's run counters.</param>
    /// <param name="direction">The archive's optimisation direction.</param>
    /// <param name="bestProgram">The best program found, or <c>null</c> when nothing was archived.</param>
    /// <param name="bestQuality">Its score, or <c>null</c> when it was never scored.</param>
    /// <param name="bestDescriptors">Its archive coordinates, or <c>null</c> for none.</param>
    /// <param name="elites">The retained elites, already ordered; <c>null</c> means none.</param>
    /// <param name="archiveCount">How many cells were filled across every island.</param>
    /// <param name="islandCount">How many islands the run used.</param>
    /// <param name="llmUsage">The language-model totals, or <c>null</c> for <see cref="ProgramEvolutionLlmUsage.Empty"/>.</param>
    /// <param name="checkpointPath">Where the checkpoint was written, or <c>null</c> when checkpointing was off.</param>
    /// <exception cref="ArgumentNullException"><paramref name="counters"/> or <paramref name="stateHash"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="stateHash"/> is blank, or <paramref name="elites"/> holds a null entry.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A count is negative, or <paramref name="bestQuality"/> is not finite.</exception>
    public ProgramEvolutionResult(
        EvolutionStopReason stopReason,
        string stateHash,
        EvolutionRunCounters counters,
        EvolutionOptimizationDirection direction,
        ProgramGenome? bestProgram,
        double? bestQuality,
        IReadOnlyDictionary<string, double>? bestDescriptors,
        IReadOnlyList<ProgramEvolutionElite>? elites,
        int archiveCount,
        int islandCount,
        ProgramEvolutionLlmUsage? llmUsage = null,
        string? checkpointPath = null)
    {
        Guard.NotNullOrWhiteSpace(stateHash);
        Guard.NotNull(counters);
        if (!Enum.IsDefined(typeof(EvolutionStopReason), stopReason))
        {
            throw new ArgumentOutOfRangeException(nameof(stopReason), stopReason, "Value must be a defined stop reason.");
        }

        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction))
        {
            throw new ArgumentOutOfRangeException(nameof(direction), direction, "Value must be a defined direction.");
        }

        if (archiveCount < 0) throw new ArgumentOutOfRangeException(nameof(archiveCount), archiveCount, "Value cannot be negative.");
        if (islandCount < 0) throw new ArgumentOutOfRangeException(nameof(islandCount), islandCount, "Value cannot be negative.");
        if (bestQuality.HasValue && (double.IsNaN(bestQuality.Value) || double.IsInfinity(bestQuality.Value)))
        {
            throw new ArgumentOutOfRangeException(nameof(bestQuality), bestQuality.Value, "Value must be a finite number.");
        }

        var eliteCopy = new List<ProgramEvolutionElite>();
        if (elites is not null)
        {
            foreach (ProgramEvolutionElite elite in elites)
            {
                if (elite is null) throw new ArgumentException("Elites cannot contain null entries.", nameof(elites));
                eliteCopy.Add(elite);
            }
        }

        StopReason = stopReason;
        StateHash = stateHash.Trim();
        Counters = counters;
        Direction = direction;
        BestProgram = bestProgram;
        BestQuality = bestQuality;
        ArchiveCount = archiveCount;
        IslandCount = islandCount;
        LlmUsage = llmUsage ?? ProgramEvolutionLlmUsage.Empty;
        CheckpointPath = checkpointPath;
        _elites = new ReadOnlyCollection<ProgramEvolutionElite>(eliteCopy);
        _bestDescriptors = bestDescriptors is null
            ? NoDescriptors
            : new ReadOnlyDictionary<string, double>(
                bestDescriptors.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal));
    }

    /// <summary>Gets why the run stopped.</summary>
    public EvolutionStopReason StopReason { get; }

    /// <summary>Gets the engine's deterministic state hash; equal hashes mean two runs were identical.</summary>
    public string StateHash { get; }

    /// <summary>Gets how many proposals, evaluation attempts, and completions the run recorded.</summary>
    public EvolutionRunCounters Counters { get; }

    /// <summary>Gets whether higher or lower quality values are better.</summary>
    public EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets the best program found, or <c>null</c> when nothing was archived.</summary>
    /// <remarks>This is the full program, not a bounded copy; only <see cref="Elites"/> is size-capped.</remarks>
    public ProgramGenome? BestProgram { get; }

    /// <summary>Gets the best program's score, or <c>null</c> when there is none.</summary>
    public double? BestQuality { get; }

    /// <summary>Gets the best program's archive coordinates; empty when none were recorded.</summary>
    public IReadOnlyDictionary<string, double> BestDescriptors => _bestDescriptors;

    /// <summary>Gets the retained elites, best first, each with a bounded copy of its source.</summary>
    public IReadOnlyList<ProgramEvolutionElite> Elites => _elites;

    /// <summary>Gets how many archive cells were filled across every island.</summary>
    public int ArchiveCount { get; }

    /// <summary>Gets how many islands the run used.</summary>
    public int IslandCount { get; }

    /// <summary>Gets the language-model totals; <see cref="ProgramEvolutionLlmUsage.Empty"/> when none were used.</summary>
    public ProgramEvolutionLlmUsage LlmUsage { get; }

    /// <summary>Gets where the run's checkpoint was written, or <c>null</c> when checkpointing was off.</summary>
    public string? CheckpointPath { get; }

    /// <summary>Gets whether the run archived at least one program.</summary>
    public bool HasBestProgram => BestProgram is not null;

    /// <summary>Summarizes a finished engine run, bounding the retained program text.</summary>
    /// <param name="runResult">The engine's own run result.</param>
    /// <param name="llmUsage">The language-model totals, or <c>null</c> when none were used.</param>
    /// <param name="checkpointPath">Where the checkpoint was written, or <c>null</c>.</param>
    /// <param name="includeEliteSourceCount">How many elites to retain, best first.</param>
    /// <param name="maxEliteSourceChars">The per-elite source bound, in characters.</param>
    /// <returns>A flat, bounded summary that does not reference the live archives.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="runResult"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="includeEliteSourceCount"/> is negative, or <paramref name="maxEliteSourceChars"/> is not positive.
    /// </exception>
    public static ProgramEvolutionResult Create(
        EvolutionRunResult<ProgramGenome> runResult,
        ProgramEvolutionLlmUsage? llmUsage = null,
        string? checkpointPath = null,
        int includeEliteSourceCount = DefaultEliteCount,
        int maxEliteSourceChars = DefaultEliteSourceChars)
    {
        Guard.NotNull(runResult);
        if (includeEliteSourceCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(includeEliteSourceCount), includeEliteSourceCount,
                "Value cannot be negative.");
        }

        if (maxEliteSourceChars <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxEliteSourceChars), maxEliteSourceChars,
                "Value must be positive.");
        }

        EvolutionOptimizationDirection direction = runResult.Islands.Count == 0
            ? EvolutionOptimizationDirection.Maximize
            : runResult.Islands[0].Direction;

        var ranked = new List<RankedEntry>();
        for (int island = 0; island < runResult.Islands.Count; island++)
        {
            foreach (EvolutionArchiveEntry<ProgramGenome> entry in runResult.Islands[island].Entries)
            {
                ranked.Add(new RankedEntry(island, entry));
            }
        }

        ranked.Sort((left, right) => Compare(left, right, direction));

        var elites = new List<ProgramEvolutionElite>();
        int retained = Math.Min(includeEliteSourceCount, ranked.Count);
        for (int index = 0; index < retained; index++)
        {
            RankedEntry candidate = ranked[index];
            ProgramGenome genome = candidate.Entry.Candidate.CanonicalGenome.Genome;
            string bounded = ProgramText.Bound(genome.Source, maxEliteSourceChars);
            elites.Add(new ProgramEvolutionElite(
                genome.Id,
                bounded,
                bounded.Length < genome.Source.Length,
                genome.Language,
                candidate.Entry.Evaluation.Quality,
                candidate.Entry.Evaluation.Descriptors,
                candidate.Entry.Cell.Bins,
                candidate.Island,
                candidate.Entry.Evaluation.EvaluationId));
        }

        EvolutionArchiveEntry<ProgramGenome>? best = runResult.Best;
        return new ProgramEvolutionResult(
            runResult.StopReason,
            runResult.StateHash,
            runResult.Counters,
            direction,
            best?.Candidate.CanonicalGenome.Genome,
            best?.Evaluation.Quality,
            best?.Evaluation.Descriptors,
            elites,
            ranked.Count,
            runResult.Islands.Count,
            llmUsage,
            checkpointPath);
    }

    /// <summary>Summarizes a finished engine run using the retention limits held on the run's options.</summary>
    /// <param name="runResult">The engine's own run result.</param>
    /// <param name="options">The options whose elite-retention limits shape the summary.</param>
    /// <param name="llmUsage">The language-model totals, or <c>null</c> when none were used.</param>
    /// <param name="checkpointPath">Where the checkpoint was written, or <c>null</c>.</param>
    /// <returns>A flat, bounded summary that does not reference the live archives.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="runResult"/> or <paramref name="options"/> is <c>null</c>.</exception>
    public static ProgramEvolutionResult Create(
        EvolutionRunResult<ProgramGenome> runResult,
        ProgramEvolutionOptions options,
        ProgramEvolutionLlmUsage? llmUsage = null,
        string? checkpointPath = null)
    {
        Guard.NotNull(options);
        return Create(runResult, llmUsage, checkpointPath, options.IncludeEliteSourceCount, options.MaxEliteSourceChars);
    }

    /// <summary>Returns a description that never echoes program text.</summary>
    /// <returns>The stop reason, the best score, and the archive coverage.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "ProgramEvolutionResult({0}, best={1}, elites={2}/{3}, {4})",
        StopReason,
        BestQuality.HasValue ? BestQuality.Value.ToString("R", CultureInfo.InvariantCulture) : "none",
        _elites.Count,
        ArchiveCount,
        LlmUsage);

    private static int Compare(RankedEntry left, RankedEntry right, EvolutionOptimizationDirection direction)
    {
        double? leftQuality = left.Entry.Evaluation.Quality;
        double? rightQuality = right.Entry.Evaluation.Quality;
        if (leftQuality.HasValue != rightQuality.HasValue) return leftQuality.HasValue ? -1 : 1;
        if (leftQuality.HasValue && rightQuality.HasValue)
        {
            int byQuality = direction == EvolutionOptimizationDirection.Maximize
                ? rightQuality.Value.CompareTo(leftQuality.Value)
                : leftQuality.Value.CompareTo(rightQuality.Value);
            if (byQuality != 0) return byQuality;
        }

        int byGenome = string.CompareOrdinal(
            left.Entry.Evaluation.GenomeId, right.Entry.Evaluation.GenomeId);
        if (byGenome != 0) return byGenome;
        int byIsland = left.Island.CompareTo(right.Island);
        return byIsland != 0 ? byIsland : left.Entry.Cell.CompareTo(right.Entry.Cell);
    }

    private readonly struct RankedEntry
    {
        public RankedEntry(int island, EvolutionArchiveEntry<ProgramGenome> entry)
        {
            Island = island;
            Entry = entry;
        }

        public int Island { get; }

        public EvolutionArchiveEntry<ProgramGenome> Entry { get; }
    }
}
