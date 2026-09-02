using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// A deterministic scalar-best-per-cell MAP-Elites archive with explicit descriptor and capacity policies.
/// </summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// Each <see cref="EvolutionDescriptorDefinition"/> contributes one axis of a grid. A completed evaluation is
/// mapped to a cell by binning its descriptor values, and the cell keeps only the entry with the best scalar
/// quality for the configured <see cref="Direction"/>; ties are broken deterministically by genome identifier,
/// cell key, and evaluation identifier, so two archives fed the same evaluations in the same order hold
/// identical elites. When <see cref="Capacity"/> is smaller than the grid, a candidate that would open a new
/// cell must beat the archive-wide worst elite, which is then evicted. Cells live in a sorted dictionary keyed
/// by the ordinal cell key, so <see cref="Get"/> and the lookup half of <see cref="TryAdd"/> cost O(log n) in
/// the number of occupied cells, <see cref="Entries"/> materializes an O(n) copy in stable key order once per archive
/// version and then serves it from a cache,
/// <see cref="Best"/> is maintained incrementally and reads in O(1), and <see cref="Sample"/> and eviction scan the
/// occupied cells (at most O(n log n)).
/// Instances are not thread-safe; the engine performs all archive mutation from its sequential commit step.
/// </para>
/// <para><b>For Beginners:</b> Picture a wall of pigeonholes where each hole stands for one style of
/// solution, for example "small and fast" or "large and accurate", and each hole may hold only the single best
/// solution of that style found so far. That wall is a MAP-Elites archive, and this class is the standard
/// implementation the evolution engine uses. Instead of the whole population collapsing onto one lucky
/// design, the archive keeps a spread of strong, distinct designs, which gives you more choices at the end and
/// supplies diverse parents for the next round of mutations. You describe the axes of the wall with
/// descriptor definitions (name, range, number of bins, and what to do with out-of-range values), choose
/// whether higher or lower quality is better, and optionally cap how many pigeonholes may be filled at once.
/// Most users never construct this class directly because the engine and the AutoML facade build it for them;
/// reach for it when writing a custom evolution task or inspecting a finished run's elites.</para>
/// <para>
/// The algorithm follows Mouret and Clune, "Illuminating search spaces by mapping elites" (2015), restricted
/// to one scalar quality per cell. <see cref="DefinitionHash"/> captures every policy that changes insertion or
/// restoration, so a checkpoint is only restored into an archive with identical semantics.
/// </para>
/// </remarks>
public sealed class MapElitesArchive<TGenome> : ICheckpointableEvolutionArchive<TGenome>
{
    private readonly EvolutionDescriptorDefinition[] _descriptors;
    private readonly ReadOnlyCollection<EvolutionDescriptorDefinition> _descriptorView;
    private readonly SortedDictionary<string, EvolutionArchiveEntry<TGenome>> _cells = new(StringComparer.Ordinal);
    private readonly int _capacity;
    private IComparer<EvolutionArchiveEntry<TGenome>>? _comparer;
    private EvolutionArchiveEntry<TGenome>? _best;
    private ReadOnlyCollection<EvolutionArchiveEntry<TGenome>>? _entries;
    private long _entriesVersion = -1;

    /// <summary>Initializes an archive.</summary>
    /// <param name="descriptors">One or more uniquely named descriptor definitions.</param>
    /// <param name="direction">The scalar quality direction for every entry.</param>
    /// <param name="capacity">Maximum occupied cells, or zero to use the full descriptor grid.</param>
    /// <param name="maximumGridCells">Safety limit for the descriptor-grid product.</param>
    /// <exception cref="ArgumentException">
    /// No descriptors were supplied, a descriptor is <c>null</c>, descriptor names collide, or the grid exceeds
    /// <paramref name="maximumGridCells"/> or overflows a 64-bit integer.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="direction"/> is undefined, <paramref name="capacity"/> is negative or exceeds the grid, or
    /// <paramref name="maximumGridCells"/> is not positive.
    /// </exception>
    public MapElitesArchive(
        IEnumerable<EvolutionDescriptorDefinition> descriptors,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize,
        int capacity = 0,
        long maximumGridCells = 10_000_000)
    {
        Guard.NotNull(descriptors);
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        if (maximumGridCells <= 0) throw new ArgumentOutOfRangeException(nameof(maximumGridCells));

        _descriptors = descriptors.ToArray();
        if (_descriptors.Length == 0) throw new ArgumentException("At least one descriptor is required.", nameof(descriptors));
        if (_descriptors.Any(item => item is null)) throw new ArgumentException("Descriptors cannot contain null entries.", nameof(descriptors));
        if (_descriptors.Select(item => item.Name).Distinct(StringComparer.Ordinal).Count() != _descriptors.Length)
            throw new ArgumentException("Descriptor names must be unique using ordinal comparison.", nameof(descriptors));

        long gridCells = 1;
        try
        {
            foreach (EvolutionDescriptorDefinition descriptor in _descriptors)
            {
                gridCells = checked(gridCells * descriptor.EffectiveBinCount);
                if (gridCells > maximumGridCells)
                    throw new ArgumentException($"The descriptor grid exceeds the {maximumGridCells} cell safety limit.", nameof(descriptors));
            }
        }
        catch (OverflowException exception)
        {
            throw new ArgumentException("The descriptor-grid size overflowed a 64-bit integer.", nameof(descriptors), exception);
        }

        Direction = direction;
        TotalGridCells = gridCells;
        _capacity = capacity == 0 ? (int)Math.Min(gridCells, int.MaxValue) : capacity;
        if (_capacity > gridCells) throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity cannot exceed the descriptor grid.");
        _descriptorView = Array.AsReadOnly(_descriptors);
        DefinitionHash = EvolutionHash.Combine(new[] { "map-elites-scalar-v1" }
            .Concat(_descriptors.Select(item => item.ToCanonicalString())).Concat(new[]
        {
            ((int)direction).ToString(System.Globalization.CultureInfo.InvariantCulture),
            _capacity.ToString(System.Globalization.CultureInfo.InvariantCulture)
        }));
    }

    /// <inheritdoc/>
    public IReadOnlyList<EvolutionDescriptorDefinition> Descriptors => _descriptorView;

    /// <inheritdoc/>
    public EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets the full number of physical cells implied by descriptor policies.</summary>
    public long TotalGridCells { get; }

    /// <summary>Gets the maximum number of occupied cells.</summary>
    public int Capacity => _capacity;

    /// <inheritdoc/>
    /// <remarks>
    /// Combines the archive algorithm identifier, every descriptor's canonical string, the optimization
    /// direction, and the effective capacity, so any change to those settings yields a different hash.
    /// </remarks>
    public string DefinitionHash { get; }

    /// <inheritdoc/>
    public int Count => _cells.Count;

    /// <inheritdoc/>
    public long Version { get; private set; }

    /// <inheritdoc/>
    /// <remarks>
    /// Materialized once per archive version and cached, so repeated reads between insertions - which the engine
    /// performs once per proposal for selection, novelty screening, and history bookkeeping - cost O(1) instead of
    /// O(n) copies. The cached instance is read-only, so handing the same one to several callers is safe.
    /// </remarks>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Entries
    {
        get
        {
            if (_entries is null || _entriesVersion != Version)
            {
                _entries = Array.AsReadOnly(_cells.Values.ToArray());
                _entriesVersion = Version;
            }
            return _entries;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Maintained incrementally on every accepted insertion, so reading it costs O(1). The ordering is a total
    /// order, so a candidate that beats the incumbent best necessarily beats every other entry and becomes the new
    /// best without a rescan.
    /// </remarks>
    public EvolutionArchiveEntry<TGenome>? Best => _best;

    /// <inheritdoc/>
    public EvolutionArchiveInsertionResult TryAdd(EvolutionCandidate<TGenome> candidate, EvolutionEvaluation evaluation)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(evaluation);
        if (evaluation.Status != EvolutionEvaluationStatus.Completed || !evaluation.Quality.HasValue ||
            evaluation.Direction != Direction || candidate.EvaluationId != evaluation.EvaluationId ||
            candidate.CanonicalGenome.Id != evaluation.GenomeId)
        {
            return EvolutionArchiveInsertionResult.Rejected;
        }

        EvolutionCellKey? key = TryCreateKey(evaluation.Descriptors);
        if (key is null) return EvolutionArchiveInsertionResult.Rejected;

        var candidateEntry = new EvolutionArchiveEntry<TGenome>(key, candidate, evaluation);
        if (_cells.TryGetValue(key.StableKey, out EvolutionArchiveEntry<TGenome>? incumbent))
        {
            if (Comparer.Compare(candidateEntry, incumbent) >= 0) return EvolutionArchiveInsertionResult.NotImproved;
            _cells[key.StableKey] = candidateEntry;
            PromoteIfBest(candidateEntry);
            Version++;
            return EvolutionArchiveInsertionResult.Replaced;
        }

        if (_cells.Count < _capacity)
        {
            _cells.Add(key.StableKey, candidateEntry);
            PromoteIfBest(candidateEntry);
            Version++;
            return EvolutionArchiveInsertionResult.Inserted;
        }

        EvolutionArchiveEntry<TGenome> worst = _cells.Values.OrderByDescending(entry => entry, Comparer).First();
        if (Comparer.Compare(candidateEntry, worst) >= 0) return EvolutionArchiveInsertionResult.NotImproved;
        _cells.Remove(worst.Cell.StableKey);
        _cells.Add(key.StableKey, candidateEntry);
        if (ReferenceEquals(_best, worst)) _best = null;
        PromoteIfBest(candidateEntry);
        Version++;
        return EvolutionArchiveInsertionResult.InsertedWithEviction;
    }

    /// <inheritdoc/>
    public EvolutionArchiveEntry<TGenome>? Get(EvolutionCellKey cell)
    {
        Guard.NotNull(cell);
        return _cells.TryGetValue(cell.StableKey, out EvolutionArchiveEntry<TGenome>? entry) ? entry : null;
    }

    /// <inheritdoc/>
    public EvolutionArchiveEntry<TGenome>? Sample(StableRandom random)
    {
        Guard.NotNull(random);
        if (_cells.Count == 0) return null;
        return _cells.Values.ElementAt(random.NextInt(_cells.Count));
    }

    /// <summary>Computes the cell for descriptor values without modifying the archive.</summary>
    /// <param name="descriptors">The named descriptor values.</param>
    /// <returns>The cell, or <c>null</c> when a value is missing or rejected.</returns>
    public EvolutionCellKey? TryCreateKey(IReadOnlyDictionary<string, double> descriptors)
    {
        Guard.NotNull(descriptors);
        var bins = new int[_descriptors.Length];
        for (int i = 0; i < _descriptors.Length; i++)
        {
            EvolutionDescriptorDefinition definition = _descriptors[i];
            if (!descriptors.TryGetValue(definition.Name, out double value) || !definition.TryGetBin(value, out bins[i]))
                return null;
        }
        return new EvolutionCellKey(bins);
    }

    /// <inheritdoc/>
    public void Restore(IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries, long version)
    {
        Guard.NotNull(entries);
        if (_cells.Count != 0 || Version != 0) throw new InvalidOperationException("Only an empty archive can be restored.");
        foreach (EvolutionArchiveEntry<TGenome> entry in entries.OrderBy(item => item.Cell.StableKey, StringComparer.Ordinal))
        {
            EvolutionArchiveInsertionResult result = TryAdd(entry.Candidate, entry.Evaluation);
            if (result != EvolutionArchiveInsertionResult.Inserted && result != EvolutionArchiveInsertionResult.InsertedWithEviction)
                throw new InvalidDataException("The archive checkpoint contains an invalid or conflicting elite.");
        }
        if (version < Version) throw new ArgumentOutOfRangeException(nameof(version));
        Version = version;
    }

    private IComparer<EvolutionArchiveEntry<TGenome>> Comparer =>
        _comparer ??= EvolutionEntryOrdering.BestFirst<TGenome>(Direction);

    private void PromoteIfBest(EvolutionArchiveEntry<TGenome> entry)
    {
        if (_best is null || Comparer.Compare(entry, _best) < 0) _best = entry;
    }
}
