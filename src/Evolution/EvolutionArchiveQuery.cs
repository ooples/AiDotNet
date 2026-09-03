using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Queries an archive or a finished run by a named metric rather than by the single scalar quality.</summary>
/// <remarks>
/// <para>
/// A run optimises one number, and that number is what <see cref="IEvolutionArchiveView{TGenome}.Best"/> ranks by.
/// An evaluation usually reports several: accuracy and latency, compression and retention, a score and the seconds
/// it cost. These extensions rank by any one of them, so a search driven by a blended objective can still answer
/// "which candidate was the most accurate" or "which was the fastest of the ones that stayed accurate enough"
/// without the caller re-implementing the ordering.
/// </para>
/// <para>
/// Two rules make the answers trustworthy. A candidate that never reported the metric is <b>absent</b> from the
/// answer rather than treated as having scored zero, which would otherwise hand a minimising query to whichever
/// candidate simply failed to measure. And ranking breaks ties on the same chain the archive itself uses, so two
/// runs with the same seed return the same answer in the same order.
/// </para>
/// <para>
/// Direction defaults to the archive's own, because most secondary metrics point the same way as the objective. A
/// metric that points the other way - a cost or an error inside a maximising run - needs its direction passed
/// explicitly; nothing about the name reveals which way it should be read.
/// </para>
/// <para><b>For Beginners:</b> After a search finishes you often want a different winner than the one the search was
/// aiming at: the cheapest model rather than the most accurate, say. Call
/// <c>result.BestBy("latency", EvolutionOptimizationDirection.Minimize)</c> and you get it, or
/// <c>result.TopBy("accuracy", 5)</c> for a shortlist. <c>MetricNames</c> tells you what you can ask for.</para>
/// </remarks>
public static class EvolutionArchiveQuery
{
    /// <summary>Returns the archive's best elite by a named metric.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="archive">The archive to read.</param>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <param name="direction">
    /// Which way the metric reads; defaults to <see cref="IEvolutionArchiveView{TGenome}.Direction"/>.
    /// </param>
    /// <returns>The best elite reporting the metric, or <see langword="null"/> when none does.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="archive"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    public static EvolutionArchiveEntry<TGenome>? BestBy<TGenome>(
        this IEvolutionArchiveView<TGenome> archive,
        string metric,
        EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNull(archive);
        Guard.NotNullOrWhiteSpace(metric);

        EvolutionOptimizationDirection resolved = direction ?? archive.Direction;
        EvolutionArchiveEntry<TGenome>? best = null;
        foreach (EvolutionArchiveEntry<TGenome> entry in archive.Entries)
        {
            if (!Reports(entry, metric)) continue;
            if (best is null || EvolutionEntryOrdering.CompareByMetric(resolved, metric, entry, best) < 0) best = entry;
        }

        return best;
    }

    /// <summary>Returns the archive's best elites by a named metric, best first.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="archive">The archive to read.</param>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <param name="count">How many to return at most.</param>
    /// <param name="direction">
    /// Which way the metric reads; defaults to <see cref="IEvolutionArchiveView{TGenome}.Direction"/>.
    /// </param>
    /// <returns>
    /// Up to <paramref name="count"/> elites, best first, shorter when fewer reported the metric and empty when none
    /// did.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="archive"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
    public static IReadOnlyList<EvolutionArchiveEntry<TGenome>> TopBy<TGenome>(
        this IEvolutionArchiveView<TGenome> archive,
        string metric,
        int count,
        EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNull(archive);
        Guard.NotNullOrWhiteSpace(metric);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), count, "Value cannot be negative.");

        return Rank(archive.Entries, metric, count, direction ?? archive.Direction, deduplicate: false);
    }

    /// <summary>Returns the elites that reported a named metric, in the archive's own stable order.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="archive">The archive to read.</param>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <returns>The reporting elites, empty when none did.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="archive"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    /// <remarks>
    /// Useful for reporting how much of the archive a metric actually covers before ranking by it: a metric only two
    /// of forty elites reported gives an answer that says far less than it appears to.
    /// </remarks>
    public static IReadOnlyList<EvolutionArchiveEntry<TGenome>> WithMetric<TGenome>(
        this IEvolutionArchiveView<TGenome> archive,
        string metric)
    {
        Guard.NotNull(archive);
        Guard.NotNullOrWhiteSpace(metric);

        var reporting = new List<EvolutionArchiveEntry<TGenome>>();
        foreach (EvolutionArchiveEntry<TGenome> entry in archive.Entries)
        {
            if (Reports(entry, metric)) reporting.Add(entry);
        }

        return reporting;
    }

    /// <summary>Returns every metric name any elite in the archive reported, ordered for stable display.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="archive">The archive to read.</param>
    /// <returns>The union of reported metric names, ordinal-sorted; empty when the archive is.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="archive"/> is <see langword="null"/>.</exception>
    /// <remarks>
    /// Names come from whatever the task reported, so this is the honest list of what can be queried rather than a
    /// list the caller has to remember. Not every elite necessarily reported every name.
    /// </remarks>
    public static IReadOnlyList<string> MetricNames<TGenome>(this IEvolutionArchiveView<TGenome> archive)
    {
        Guard.NotNull(archive);
        return CollectNames(archive.Entries);
    }

    /// <summary>Returns the run's best elite by a named metric across every island.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="result">The finished run.</param>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <param name="direction">Which way the metric reads; defaults to the run's own direction.</param>
    /// <returns>The best elite reporting the metric, or <see langword="null"/> when none does.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="result"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    public static EvolutionArchiveEntry<TGenome>? BestBy<TGenome>(
        this EvolutionRunResult<TGenome> result,
        string metric,
        EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNull(result);
        Guard.NotNullOrWhiteSpace(metric);

        EvolutionOptimizationDirection resolved = direction ?? RunDirection(result);
        EvolutionArchiveEntry<TGenome>? best = null;
        foreach (IEvolutionArchiveView<TGenome> island in result.Islands)
        {
            EvolutionArchiveEntry<TGenome>? candidate = island.BestBy(metric, resolved);
            if (candidate is null) continue;
            if (best is null || EvolutionEntryOrdering.CompareByMetric(resolved, metric, candidate, best) < 0)
                best = candidate;
        }

        return best;
    }

    /// <summary>Returns the run's best elites by a named metric across every island, best first.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="result">The finished run.</param>
    /// <param name="metric">The metric name, matched exactly.</param>
    /// <param name="count">How many to return at most.</param>
    /// <param name="direction">Which way the metric reads; defaults to the run's own direction.</param>
    /// <returns>Up to <paramref name="count"/> distinct elites, best first.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="result"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="metric"/> is empty or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
    /// <remarks>
    /// One candidate can sit in more than one island once migration has copied it, so the shortlist is deduplicated
    /// by canonical genome id: asking for five gives five different candidates rather than one popular one five
    /// times.
    /// </remarks>
    public static IReadOnlyList<EvolutionArchiveEntry<TGenome>> TopBy<TGenome>(
        this EvolutionRunResult<TGenome> result,
        string metric,
        int count,
        EvolutionOptimizationDirection? direction = null)
    {
        Guard.NotNull(result);
        Guard.NotNullOrWhiteSpace(metric);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), count, "Value cannot be negative.");

        var everywhere = new List<EvolutionArchiveEntry<TGenome>>();
        foreach (IEvolutionArchiveView<TGenome> island in result.Islands) everywhere.AddRange(island.Entries);

        return Rank(everywhere, metric, count, direction ?? RunDirection(result), deduplicate: true);
    }

    /// <summary>Returns every metric name any elite of the run reported, ordered for stable display.</summary>
    /// <typeparam name="TGenome">The task-specific genome type.</typeparam>
    /// <param name="result">The finished run.</param>
    /// <returns>The union of reported metric names, ordinal-sorted.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="result"/> is <see langword="null"/>.</exception>
    public static IReadOnlyList<string> MetricNames<TGenome>(this EvolutionRunResult<TGenome> result)
    {
        Guard.NotNull(result);

        var everywhere = new List<EvolutionArchiveEntry<TGenome>>();
        foreach (IEvolutionArchiveView<TGenome> island in result.Islands) everywhere.AddRange(island.Entries);

        return CollectNames(everywhere);
    }

    /// <summary>Ranks entries reporting a metric, best first, optionally keeping one entry per genome.</summary>
    private static IReadOnlyList<EvolutionArchiveEntry<TGenome>> Rank<TGenome>(
        IEnumerable<EvolutionArchiveEntry<TGenome>> entries,
        string metric,
        int count,
        EvolutionOptimizationDirection direction,
        bool deduplicate)
    {
        if (count == 0) return Array.Empty<EvolutionArchiveEntry<TGenome>>();

        var reporting = new List<EvolutionArchiveEntry<TGenome>>();
        foreach (EvolutionArchiveEntry<TGenome> entry in entries)
        {
            if (Reports(entry, metric)) reporting.Add(entry);
        }

        // Sorted rather than partially selected: the archive is bounded by its grid, so the whole list is small, and
        // a full deterministic sort is easier to reason about than a hand-rolled selection with the same tie chain.
        reporting.Sort((x, y) => EvolutionEntryOrdering.CompareByMetric(direction, metric, x, y));

        var top = new List<EvolutionArchiveEntry<TGenome>>(Math.Min(count, reporting.Count));
        HashSet<string>? seen = deduplicate ? new HashSet<string>(StringComparer.Ordinal) : null;
        foreach (EvolutionArchiveEntry<TGenome> entry in reporting)
        {
            if (top.Count >= count) break;
            if (seen is not null && !seen.Add(entry.Evaluation.GenomeId)) continue;
            top.Add(entry);
        }

        return top;
    }

    /// <summary>Collects the ordinal-sorted union of metric names across entries.</summary>
    private static IReadOnlyList<string> CollectNames<TGenome>(IEnumerable<EvolutionArchiveEntry<TGenome>> entries)
    {
        var names = new SortedSet<string>(StringComparer.Ordinal);
        foreach (EvolutionArchiveEntry<TGenome> entry in entries)
        {
            foreach (KeyValuePair<string, double> metric in entry.Evaluation.Metrics)
            {
                if (EvolutionDescriptorDefinition.IsFinite(metric.Value)) names.Add(metric.Key);
            }
        }

        return new List<string>(names);
    }

    /// <summary>Reports whether an entry carries a usable value for a metric.</summary>
    /// <remarks>
    /// A missing name and a name whose value is not a finite number are the same answer: this entry cannot be
    /// ranked by that metric, so it is left out rather than ranked as if it had scored something.
    /// </remarks>
    private static bool Reports<TGenome>(EvolutionArchiveEntry<TGenome> entry, string metric) =>
        entry.Evaluation.Metrics.TryGetValue(metric, out double value) &&
        EvolutionDescriptorDefinition.IsFinite(value);

    /// <summary>Reads the direction the run optimised, falling back to maximisation for a run with no islands.</summary>
    private static EvolutionOptimizationDirection RunDirection<TGenome>(EvolutionRunResult<TGenome> result) =>
        result.Islands.Count > 0 ? result.Islands[0].Direction : EvolutionOptimizationDirection.Maximize;
}
