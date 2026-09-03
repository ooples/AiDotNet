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

    /// <summary>Returns the retained elite that scored best on a named metric.</summary>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <param name="direction">Which way the metric reads; defaults to <see cref="Direction"/>.</param>
    /// <returns>The best retained elite reporting the metric, or <c>null</c> when none does.</returns>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    /// <remarks>
    /// <para>
    /// The search optimised one number, and <see cref="BestProgram"/> is the winner by that number. An evaluator
    /// usually reports several, so this ranks the retained elites by any one of them: the most accurate program
    /// rather than the highest blended score, or the fastest of the ones that were accurate enough.
    /// </para>
    /// <para>
    /// An elite that never reported the metric is left out rather than treated as having scored zero, which would
    /// otherwise hand a minimising query to whichever program simply failed to measure. Direction defaults to the
    /// run's own; pass it explicitly for a metric that reads the other way, such as a runtime inside a maximising
    /// run. Only <see cref="Elites"/> is searched, so the answer is drawn from the elites the run was configured to
    /// retain rather than from the whole archive.
    /// </para>
    /// <para><b>For Beginners:</b> <c>result.BestBy("accuracy")</c> gives the most accurate program kept, and
    /// <c>result.MetricNames()</c> lists what you can ask for.</para>
    /// </remarks>
    public ProgramEvolutionElite? BestBy(string metric, EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNullOrWhiteSpace(metric);

        EvolutionOptimizationDirection resolved = direction ?? Direction;
        ProgramEvolutionElite? best = null;
        foreach (ProgramEvolutionElite elite in _elites)
        {
            if (!Reports(elite, metric)) continue;
            if (best is null || CompareByMetric(resolved, metric, elite, best) < 0) best = elite;
        }

        return best;
    }

    /// <summary>Returns the retained elites that scored best on a named metric, best first.</summary>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <param name="count">How many to return at most.</param>
    /// <param name="direction">Which way the metric reads; defaults to <see cref="Direction"/>.</param>
    /// <returns>Up to <paramref name="count"/> retained elites, best first, shorter when fewer reported the metric.</returns>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
    public IReadOnlyList<ProgramEvolutionElite> TopBy(string metric, int count,
        EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNullOrWhiteSpace(metric);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), count, "Value cannot be negative.");
        if (count == 0) return Array.Empty<ProgramEvolutionElite>();

        EvolutionOptimizationDirection resolved = direction ?? Direction;
        var reporting = new List<ProgramEvolutionElite>();
        foreach (ProgramEvolutionElite elite in _elites)
        {
            if (Reports(elite, metric)) reporting.Add(elite);
        }

        reporting.Sort((left, right) => CompareByMetric(resolved, metric, left, right));
        if (reporting.Count > count) reporting.RemoveRange(count, reporting.Count - count);
        return reporting;
    }

    /// <summary>Returns every metric name any retained elite reported, ordered for stable display.</summary>
    /// <returns>The ordinal-sorted union of reported metric names; empty when nothing was retained.</returns>
    /// <remarks>
    /// Names originate in an evaluated program and are untrusted: display them rather than acting on them. Not every
    /// elite necessarily reported every name.
    /// </remarks>
    public IReadOnlyList<string> MetricNames()
    {
        var names = new SortedSet<string>(StringComparer.Ordinal);
        foreach (ProgramEvolutionElite elite in _elites)
        {
            foreach (KeyValuePair<string, double> metric in elite.Metrics)
            {
                if (IsRankable(metric.Value)) names.Add(metric.Key);
            }
        }

        return new List<string>(names);
    }

    /// <summary>Reports whether an elite carries a usable value for a metric.</summary>
    private static bool Reports(ProgramEvolutionElite elite, string metric) =>
        elite.Metrics.TryGetValue(metric, out double value) && IsRankable(value);

    /// <summary>Reports whether a metric value can take part in an ordering at all.</summary>
    private static bool IsRankable(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    /// <summary>
    /// Orders two elites best first by a named metric, breaking ties on the same chain the archive uses so a query
    /// against this result and one against the live archive cannot disagree.
    /// </summary>
    private static int CompareByMetric(EvolutionOptimizationDirection direction, string metric,
        ProgramEvolutionElite left, ProgramEvolutionElite right)
    {
        double leftValue = left.Metrics[metric];
        double rightValue = right.Metrics[metric];
        int value = direction == EvolutionOptimizationDirection.Maximize
            ? rightValue.CompareTo(leftValue)
            : leftValue.CompareTo(rightValue);
        if (value != 0) return value;
        int genome = StringComparer.Ordinal.Compare(left.GenomeId, right.GenomeId);
        if (genome != 0) return genome;
        return left.EvaluationId.CompareTo(right.EvaluationId);
    }

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
                candidate.Entry.Evaluation.EvaluationId,
                candidate.Entry.Evaluation.Metrics));
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
