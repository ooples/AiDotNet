using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Mixes exploration, elite exploitation, and island-best selection using validated branch ratios.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// One draw from the proposal's <see cref="StableRandom"/> stream chooses the branch: below
/// <see cref="EvolutionSelectionOptions.ExplorationRatio"/> the parent is a uniformly random occupied cell, below the
/// sum of the exploration and exploitation ratios it is drawn uniformly from the strongest
/// <see cref="EvolutionSelectionOptions.ExploitationEliteCount"/> candidates, and otherwise it is the target island's
/// current best entry. The exploitation pool comes from the engine's cross-island elite index when
/// <see cref="EvolutionSelectionOptions.ExploitationSource"/> is
/// <see cref="EvolutionExploitationSource.GlobalTopK"/> and the index is populated, and from the island's own
/// best entries otherwise, so a cross-island parent is opt-in rather than the accidental leak it is in OpenEvolve.
/// Every branch falls back to uniform sampling when its pool is empty, so a non-empty archive always yields a parent.
/// </para>
/// <para>
/// Inspirations are assembled in a fixed order: the island best when
/// <see cref="EvolutionSelectionOptions.IncludeIslandBest"/> is set, then the
/// <see cref="EvolutionSelectionOptions.TopInspirationCount"/> highest-quality remaining elites, then the
/// <see cref="EvolutionSelectionOptions.DiverseInspirationCount"/> entries chosen by greedy maximum-minimum
/// Manhattan distance between cell coordinates, and finally a uniform random fill drawn from the same stream. Only
/// the exploration draw, the exploitation draw, and the fill consume randomness; the top and diverse picks are
/// deterministic functions of the archive contents, so the whole selection is reproducible and independent of the
/// order in which entries were inserted. A call costs O(n log n) in the occupied cells for ranking plus
/// O(k * n * d) for k diverse picks over d descriptors.
/// </para>
/// <para><b>For Beginners:</b> Picking which existing solution to mutate next is a balancing act. Choosing at random
/// keeps the search broad; always choosing the champion makes it converge fast but narrowly. This policy does both
/// in fixed proportions you control: by default 20 percent random, 70 percent one of the strongest solutions, and 10
/// percent the current island champion. It then hands your variation operator a few extra solutions as inspiration,
/// mixing the very best with some deliberately different ones so the operator sees contrast rather than a set of
/// near-copies. Use it when you want OpenEvolve-style sampling with reproducible results.</para>
/// </remarks>
public sealed class RatioEvolutionSelectionPolicy<TGenome> : IEliteIndexAwareEvolutionSelectionPolicy<TGenome>
{
    private readonly EvolutionSelectionOptions _options;
    private readonly string _versionHash;
    private IReadOnlyList<EvolutionEliteRecord<TGenome>> _globalElites = Array.Empty<EvolutionEliteRecord<TGenome>>();

    /// <summary>Initializes a ratio selection policy.</summary>
    /// <param name="options">
    /// The branch ratios and inspiration mix; defaults are used when <c>null</c>. The instance is validated and
    /// defensively copied, so later mutation of the argument cannot affect the policy.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">A ratio or count is outside its permitted range.</exception>
    /// <exception cref="ArgumentException">The three branch ratios do not sum to one.</exception>
    public RatioEvolutionSelectionPolicy(EvolutionSelectionOptions? options = null)
    {
        _options = (options ?? new EvolutionSelectionOptions()).SnapshotAndValidate();
        _versionHash = "ratio-selection-v1|" + _options.ToCanonicalString();
    }

    /// <inheritdoc/>
    public string Id => "ratio-selection";

    /// <inheritdoc/>
    /// <remarks>Includes the canonical branch ratios, so changing any of them rejects an older checkpoint.</remarks>
    public string VersionHash => _versionHash;

    /// <inheritdoc/>
    /// <remarks>
    /// The island index is validated but not otherwise used: the archive handed to
    /// <see cref="Select"/> already identifies the island, and
    /// <see cref="EvolutionExploitationSource.GlobalTopK"/> deliberately draws from every island.
    /// </remarks>
    public void UseEliteIndex(IReadOnlyList<EvolutionEliteRecord<TGenome>> elites, int island)
    {
        Guard.NotNull(elites);
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        _globalElites = elites;
    }

    /// <inheritdoc/>
    public EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount)
    {
        Guard.NotNull(archive);
        Guard.NotNull(random);
        if (inspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(inspirationCount));
        IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries = archive.Entries;
        if (entries.Count == 0) return null;

        double draw = random.NextDouble();
        EvolutionArchiveEntry<TGenome>? parent;
        if (draw < _options.ExplorationRatio)
        {
            parent = archive.Sample(random);
        }
        else if (draw < _options.ExplorationRatio + _options.ExploitationRatio)
        {
            parent = SampleExploitationParent(archive, entries, random);
        }
        else
        {
            parent = archive.Best;
        }

        parent ??= archive.Sample(random);
        if (parent is null) return null;
        return new EvolutionSelection<TGenome>(parent, BuildInspirations(archive, entries, parent, random, inspirationCount));
    }

    private EvolutionArchiveEntry<TGenome>? SampleExploitationParent(IEvolutionArchive<TGenome> archive,
        IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries, StableRandom random)
    {
        EvolutionArchiveEntry<TGenome>[] pool = Array.Empty<EvolutionArchiveEntry<TGenome>>();
        if (_options.ExploitationSource == EvolutionExploitationSource.GlobalTopK && _globalElites.Count > 0)
        {
            pool = _globalElites.Take(_options.ExploitationEliteCount).Select(record => record.Entry).ToArray();
        }
        if (pool.Length == 0)
        {
            pool = RankBestFirst(archive, entries).Take(_options.ExploitationEliteCount).ToArray();
        }
        return pool.Length == 0 ? null : pool[random.NextInt(pool.Length)];
    }

    private IReadOnlyList<EvolutionArchiveEntry<TGenome>> BuildInspirations(IEvolutionArchive<TGenome> archive,
        IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries, EvolutionArchiveEntry<TGenome> parent,
        StableRandom random, int inspirationCount)
    {
        var selected = new List<EvolutionArchiveEntry<TGenome>>(Math.Min(inspirationCount, entries.Count));
        if (inspirationCount == 0) return selected.AsReadOnly();
        var chosen = new HashSet<string>(StringComparer.Ordinal) { parent.Evaluation.GenomeId };

        EvolutionArchiveEntry<TGenome>? islandBest = archive.Best;
        if (_options.IncludeIslandBest && islandBest is not null && chosen.Add(islandBest.Evaluation.GenomeId))
            selected.Add(islandBest);

        int topTarget = Math.Min(inspirationCount, selected.Count + _options.TopInspirationCount);
        foreach (EvolutionArchiveEntry<TGenome> entry in RankBestFirst(archive, entries))
        {
            if (selected.Count >= topTarget) break;
            if (chosen.Add(entry.Evaluation.GenomeId)) selected.Add(entry);
        }

        AddDiverseInspirations(entries, parent, chosen, selected, inspirationCount);
        AddRandomFill(entries, chosen, selected, random, inspirationCount);
        return selected.AsReadOnly();
    }

    private void AddDiverseInspirations(IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries,
        EvolutionArchiveEntry<TGenome> parent, HashSet<string> chosen,
        List<EvolutionArchiveEntry<TGenome>> selected, int inspirationCount)
    {
        int target = Math.Min(inspirationCount, selected.Count + _options.DiverseInspirationCount);
        var anchors = new List<EvolutionCellKey> { parent.Cell };
        foreach (EvolutionArchiveEntry<TGenome> entry in selected) anchors.Add(entry.Cell);

        while (selected.Count < target)
        {
            EvolutionArchiveEntry<TGenome>? farthest = null;
            long farthestDistance = -1;
            foreach (EvolutionArchiveEntry<TGenome> entry in entries)
            {
                if (chosen.Contains(entry.Evaluation.GenomeId)) continue;
                long distance = MinimumCellDistance(entry.Cell, anchors);
                if (distance <= farthestDistance) continue;
                farthest = entry;
                farthestDistance = distance;
            }
            if (farthest is null) return;
            chosen.Add(farthest.Evaluation.GenomeId);
            selected.Add(farthest);
            anchors.Add(farthest.Cell);
        }
    }

    private static void AddRandomFill(IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries, HashSet<string> chosen,
        List<EvolutionArchiveEntry<TGenome>> selected, StableRandom random, int inspirationCount)
    {
        if (selected.Count >= inspirationCount) return;
        EvolutionArchiveEntry<TGenome>[] remaining = entries
            .Where(entry => !chosen.Contains(entry.Evaluation.GenomeId))
            .OrderBy(entry => entry.Cell.StableKey, StringComparer.Ordinal)
            .ToArray();
        int take = Math.Min(inspirationCount - selected.Count, remaining.Length);
        for (int i = 0; i < take; i++)
        {
            int index = random.NextInt(i, remaining.Length);
            EvolutionArchiveEntry<TGenome> swap = remaining[i];
            remaining[i] = remaining[index];
            remaining[index] = swap;
            chosen.Add(remaining[i].Evaluation.GenomeId);
            selected.Add(remaining[i]);
        }
    }

    private static IEnumerable<EvolutionArchiveEntry<TGenome>> RankBestFirst(IEvolutionArchive<TGenome> archive,
        IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries) =>
        entries.OrderBy(entry => entry, EvolutionEntryOrdering.BestFirst<TGenome>(archive.Direction));

    private static long MinimumCellDistance(EvolutionCellKey cell, IReadOnlyList<EvolutionCellKey> anchors)
    {
        long minimum = long.MaxValue;
        foreach (EvolutionCellKey anchor in anchors)
        {
            long distance = 0;
            int shared = Math.Min(cell.Bins.Count, anchor.Bins.Count);
            for (int i = 0; i < shared; i++) distance += Math.Abs((long)cell.Bins[i] - anchor.Bins[i]);
            if (distance < minimum) minimum = distance;
        }
        return minimum == long.MaxValue ? 0 : minimum;
    }
}
