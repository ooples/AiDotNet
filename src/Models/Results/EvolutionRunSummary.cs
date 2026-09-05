using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Validation;

namespace AiDotNet.Models.Results;

/// <summary>A redacted, serializable account of one finished evolution run.</summary>
/// <remarks>
/// <para>
/// The engine's own <c>EvolutionRunResult&lt;TGenome&gt;</c> holds live archive snapshots and is generic over the
/// genome type, so it can be neither serialized nor exposed on a non-generic facade. This is its flat projection:
/// why the run stopped, how much work it did, how each island fared, which cells the best candidates occupy, what
/// failed, what the language model cost, and where the run's files were written. It carries identifiers and
/// coordinates rather than genomes, so it is safe to log or to save inside a model file. The typed result stays
/// reachable through <c>AiModelResult.GetEvolutionRunResult&lt;TGenome&gt;()</c> for callers that want the genomes
/// themselves.
/// </para>
/// <para>
/// <see cref="StateHash"/> and <see cref="CompatibilityHash"/> are the two identity values worth keeping. Two runs
/// with the same seed, options, and components produce the same state hash, which makes it the cheapest possible
/// determinism check. The compatibility hash instead describes the configuration: a checkpoint can only be resumed
/// by a run whose compatibility hash matches, so recording it tells you later whether a saved run can be continued.
/// </para>
/// <para><b>For Beginners:</b> This is the report card for a search. Read <see cref="StopReason"/> first — it says
/// whether the run finished because it ran out of budget, ran out of time, reached the quality you asked for, or
/// stopped improving. Then <see cref="BestQuality"/> for the score of the winner, <see cref="Elites"/> for the good
/// and genuinely different candidates it collected, and <see cref="RetainedFailures"/> when the answer is
/// disappointing and you want to know what went wrong.</para>
/// </remarks>
public sealed class EvolutionRunSummary
{
    /// <summary>The default number of elites a summary retains, best first.</summary>
    public const int DefaultEliteCount = 10;

    private IDictionary<string, long>? _statusCounts;
    private IList<EvolutionEliteSummary>? _elites;
    private IList<EvolutionIslandSummary>? _islands;
    private IList<EvolutionFailureSummary>? _retainedFailures;

    /// <summary>Gets or sets the run identifier the checkpoint and trace files were named after.</summary>
    public string RunId { get; set; } = string.Empty;

    /// <summary>Gets or sets why the run stopped.</summary>
    public EvolutionStopReason StopReason { get; set; }

    /// <summary>Gets or sets the deterministic state hash; equal hashes mean two runs did exactly the same work.</summary>
    public string StateHash { get; set; } = string.Empty;

    /// <summary>Gets or sets the configuration hash a checkpoint must match to be resumable by this run.</summary>
    public string CompatibilityHash { get; set; } = string.Empty;

    /// <summary>Gets or sets whether higher or lower quality values were better.</summary>
    public EvolutionOptimizationDirection Direction { get; set; } = EvolutionOptimizationDirection.Maximize;

    /// <summary>Gets or sets how many candidates were proposed, duplicates and rejections included.</summary>
    public long Proposals { get; set; }

    /// <summary>Gets or sets how many evaluator attempts were made, retries included.</summary>
    public long EvaluationAttempts { get; set; }

    /// <summary>Gets or sets how many evaluations completed and were committed.</summary>
    public long CompletedEvaluations { get; set; }

    /// <summary>Gets or sets how many evaluator calls were abandoned after outrunning their grace period.</summary>
    public long AbandonedEvaluations { get; set; }

    /// <summary>Gets or sets how many evaluations ended in each status, keyed by the status name.</summary>
    /// <remarks>Keys are the names of <see cref="EvolutionEvaluationStatus"/>, so the map survives a JSON round trip.</remarks>
    public IDictionary<string, long> StatusCounts
    {
        get => _statusCounts ??= new Dictionary<string, long>(StringComparer.Ordinal);
        set => _statusCounts = value;
    }

    /// <summary>Gets or sets the best quality found across every island, or <c>null</c> when nothing was scored.</summary>
    public double? BestQuality { get; set; }

    /// <summary>Gets or sets the identifier of the best candidate, or <c>null</c> when nothing was archived.</summary>
    public string? BestGenomeId { get; set; }

    /// <summary>Gets or sets how many archive cells were filled across every island.</summary>
    public int ArchiveCount { get; set; }

    /// <summary>Gets or sets how many islands the run used.</summary>
    public int IslandCount { get; set; }

    /// <summary>Gets or sets the retained elites, best first and bounded by the configured count.</summary>
    public IList<EvolutionEliteSummary> Elites
    {
        get => _elites ??= new List<EvolutionEliteSummary>();
        set => _elites = value;
    }

    /// <summary>Gets or sets one summary per island, in island order.</summary>
    public IList<EvolutionIslandSummary> Islands
    {
        get => _islands ??= new List<EvolutionIslandSummary>();
        set => _islands = value;
    }

    /// <summary>Gets or sets the retained failure diagnostics, oldest first.</summary>
    /// <remarks>These originate in evaluated candidates and are untrusted; display them rather than acting on them.</remarks>
    public IList<EvolutionFailureSummary> RetainedFailures
    {
        get => _retainedFailures ??= new List<EvolutionFailureSummary>();
        set => _retainedFailures = value;
    }

    /// <summary>Gets or sets the language-model totals, or <c>null</c> when no model was used.</summary>
    public ProgramEvolutionLlmUsage? LlmUsage { get; set; }

    /// <summary>Gets or sets the directory the run's files were written under, or <c>null</c> when nothing was written.</summary>
    public string? OutputDirectory { get; set; }

    /// <summary>Gets or sets the checkpoint file path, or <c>null</c> when checkpointing was off.</summary>
    public string? CheckpointPath { get; set; }

    /// <summary>Gets or sets the trace file path, or <c>null</c> when tracing was off.</summary>
    public string? TracePath { get; set; }

    /// <summary>Gets or sets how many records were written to the trace.</summary>
    public long TraceRecordCount { get; set; }

    /// <summary>Gets or sets when the run started, in UTC.</summary>
    public DateTimeOffset StartedUtc { get; set; }

    /// <summary>Gets or sets when the run finished, in UTC.</summary>
    public DateTimeOffset FinishedUtc { get; set; }

    /// <summary>Gets whether the run archived at least one candidate.</summary>
    public bool HasBest => BestGenomeId is not null;

    /// <summary>Returns the retained elite that scored best on a named metric.</summary>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <param name="direction">Which way the metric reads; defaults to <see cref="Direction"/>.</param>
    /// <returns>The best retained elite reporting the metric, or <see langword="null"/> when none does.</returns>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    /// <remarks>
    /// <para>
    /// The run optimised one number; an evaluation usually reports several. This ranks the retained elites by any
    /// one of them, so a finished search can still answer "which candidate was the most accurate" or "which was the
    /// cheapest" — and it works on a summary read back from disk, because the numbers travel with it.
    /// </para>
    /// <para>
    /// An elite that never reported the metric is left out rather than treated as having scored zero, which would
    /// otherwise hand a minimising query to whichever candidate simply failed to measure. Direction defaults to the
    /// run's own, because most secondary metrics point the same way as the objective; pass it explicitly for one
    /// that does not, such as a cost inside a maximising run. Only the retained elites are searched — the run kept
    /// as many as the configured elite count allowed, best-quality first.
    /// </para>
    /// <para><b>For Beginners:</b> <c>summary.BestBy("accuracy")</c> gives the most accurate candidate the run
    /// kept. Use <see cref="MetricNames"/> to see what you can ask for.</para>
    /// </remarks>
    public EvolutionEliteSummary? BestBy(string metric, EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNullOrWhiteSpace(metric);

        EvolutionOptimizationDirection resolved = direction ?? Direction;
        EvolutionEliteSummary? best = null;
        foreach (EvolutionEliteSummary elite in Elites)
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
    /// <returns>
    /// Up to <paramref name="count"/> retained elites, best first, shorter when fewer reported the metric.
    /// </returns>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
    public IReadOnlyList<EvolutionEliteSummary> TopBy(string metric, int count,
        EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNullOrWhiteSpace(metric);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), count, "Value cannot be negative.");
        if (count == 0) return Array.Empty<EvolutionEliteSummary>();

        EvolutionOptimizationDirection resolved = direction ?? Direction;
        var reporting = new List<EvolutionEliteSummary>();
        foreach (EvolutionEliteSummary elite in Elites)
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
    /// Names are supplied by the task and, for program evolution, originate in an evaluated candidate: display them
    /// rather than acting on them. Not every elite necessarily reported every name.
    /// </remarks>
    public IReadOnlyList<string> MetricNames()
    {
        var names = new SortedSet<string>(StringComparer.Ordinal);
        foreach (EvolutionEliteSummary elite in Elites)
        {
            foreach (KeyValuePair<string, double> metric in elite.Metrics)
            {
                if (IsRankable(metric.Value)) names.Add(metric.Key);
            }
        }

        return new List<string>(names);
    }

    /// <summary>Returns a short description that never echoes candidate content.</summary>
    /// <returns>The stop reason, the best quality, and the archive coverage.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "EvolutionRunSummary({0}, best={1}, cells={2}, islands={3})",
        StopReason,
        BestQuality.HasValue ? BestQuality.Value.ToString("R", CultureInfo.InvariantCulture) : "none",
        ArchiveCount,
        IslandCount);

    /// <summary>Projects a finished engine run onto the flat, serializable summary.</summary>
    /// <typeparam name="TGenome">The genome type the run used.</typeparam>
    /// <param name="runId">The run identifier the files were named after.</param>
    /// <param name="compatibilityHash">The engine's compatibility hash.</param>
    /// <param name="run">The engine's own run result.</param>
    /// <param name="startedUtc">When the run started.</param>
    /// <param name="finishedUtc">When the run finished.</param>
    /// <param name="maxElites">How many elites to retain, best first.</param>
    /// <returns>A summary that holds no reference to the live archives.</returns>
    internal static EvolutionRunSummary Create<TGenome>(
        string runId,
        string compatibilityHash,
        EvolutionRunResult<TGenome> run,
        DateTimeOffset startedUtc,
        DateTimeOffset finishedUtc,
        int maxElites = DefaultEliteCount)
    {
        EvolutionOptimizationDirection direction = run.Islands.Count == 0
            ? EvolutionOptimizationDirection.Maximize
            : run.Islands[0].Direction;

        var summary = new EvolutionRunSummary
        {
            RunId = runId,
            StopReason = run.StopReason,
            StateHash = run.StateHash,
            CompatibilityHash = compatibilityHash,
            Direction = direction,
            Proposals = run.Counters.Proposals,
            EvaluationAttempts = run.Counters.EvaluationAttempts,
            CompletedEvaluations = run.Counters.CompletedEvaluations,
            AbandonedEvaluations = run.Counters.AbandonedEvaluations,
            IslandCount = run.Islands.Count,
            StartedUtc = startedUtc,
            FinishedUtc = finishedUtc
        };

        foreach (KeyValuePair<EvolutionEvaluationStatus, long> pair in run.Counters.StatusCounts)
        {
            summary.StatusCounts[pair.Key.ToString()] = pair.Value;
        }

        foreach (EvolutionIslandStatus status in run.IslandStatuses)
        {
            summary.Islands.Add(new EvolutionIslandSummary
            {
                Island = status.Island,
                Generation = status.Generation,
                EliteCount = status.EliteCount,
                TotalCells = status.TotalCells,
                Coverage = status.Coverage,
                BestGenomeId = status.BestGenomeId,
                BestQuality = status.BestQuality,
                MeanQuality = status.MeanQuality,
                HistoryCount = status.HistoryCount
            });
        }

        foreach (EvolutionDiagnostic diagnostic in run.RetainedFailures)
        {
            summary.RetainedFailures.Add(new EvolutionFailureSummary
            {
                Code = diagnostic.Code,
                Message = diagnostic.Message,
                IsRedacted = diagnostic.IsRedacted
            });
        }

        var ranked = new List<RankedEntry<TGenome>>();
        for (int island = 0; island < run.Islands.Count; island++)
        {
            foreach (EvolutionArchiveEntry<TGenome> entry in run.Islands[island].Entries)
            {
                ranked.Add(new RankedEntry<TGenome>(island, entry));
            }
        }

        ranked.Sort((left, right) => Compare(left, right, direction));
        summary.ArchiveCount = ranked.Count;

        int retained = Math.Min(Math.Max(maxElites, 0), ranked.Count);
        for (int index = 0; index < retained; index++)
        {
            RankedEntry<TGenome> candidate = ranked[index];
            var elite = new EvolutionEliteSummary
            {
                GenomeId = candidate.Entry.Evaluation.GenomeId,
                Island = candidate.Island,
                Quality = candidate.Entry.Evaluation.Quality,
                EvaluationId = candidate.Entry.Evaluation.EvaluationId
            };

            foreach (KeyValuePair<string, double> pair in candidate.Entry.Evaluation.Descriptors)
            {
                elite.Descriptors[pair.Key] = pair.Value;
            }

            // Carried for the same reason as the descriptors: they are numbers the task reported, not genome
            // content, and without them a saved summary cannot answer any question but the one the run optimised.
            foreach (KeyValuePair<string, double> pair in candidate.Entry.Evaluation.Metrics)
            {
                elite.Metrics[pair.Key] = pair.Value;
            }

            foreach (int bin in candidate.Entry.Cell.Bins) elite.Cell.Add(bin);
            summary.Elites.Add(elite);
        }

        EvolutionArchiveEntry<TGenome>? best = run.Best;
        summary.BestQuality = best?.Evaluation.Quality;
        summary.BestGenomeId = best?.Evaluation.GenomeId;
        return summary;
    }

    /// <summary>Reports whether an elite carries a usable value for a metric.</summary>
    /// <remarks>
    /// A missing name and a name whose value is not a finite number are the same answer: this elite cannot be ranked
    /// by that metric, so it is left out rather than ranked as if it had scored something.
    /// </remarks>
    private static bool Reports(EvolutionEliteSummary elite, string metric) =>
        elite.Metrics.TryGetValue(metric, out double value) && IsRankable(value);

    /// <summary>Reports whether a metric value can take part in an ordering at all.</summary>
    private static bool IsRankable(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    /// <summary>
    /// Orders two elites best first by a named metric, breaking ties on the same chain the archive itself uses so a
    /// query against the summary and one against the live archive cannot disagree.
    /// </summary>
    private static int CompareByMetric(EvolutionOptimizationDirection direction, string metric,
        EvolutionEliteSummary left, EvolutionEliteSummary right)
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

    private static int Compare<TGenome>(
        RankedEntry<TGenome> left,
        RankedEntry<TGenome> right,
        EvolutionOptimizationDirection direction)
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

        int byGenome = string.CompareOrdinal(left.Entry.Evaluation.GenomeId, right.Entry.Evaluation.GenomeId);
        if (byGenome != 0) return byGenome;
        int byIsland = left.Island.CompareTo(right.Island);
        return byIsland != 0 ? byIsland : left.Entry.Cell.CompareTo(right.Entry.Cell);
    }

    /// <summary>One archive entry paired with the island it came from, so ranking can break ties by island.</summary>
    /// <typeparam name="TGenome">The candidate type being evolved.</typeparam>
    private readonly struct RankedEntry<TGenome>
    {
        public RankedEntry(int island, EvolutionArchiveEntry<TGenome> entry)
        {
            Island = island;
            Entry = entry;
        }

        public int Island { get; }

        public EvolutionArchiveEntry<TGenome> Entry { get; }
    }
}
