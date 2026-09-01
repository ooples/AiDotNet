using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// A deterministic scalar-best-per-cell MAP-Elites archive with explicit descriptor and capacity policies.
/// </summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class MapElitesArchive<TGenome> : ICheckpointableEvolutionArchive<TGenome>
{
    private readonly EvolutionDescriptorDefinition[] _descriptors;
    private readonly ReadOnlyCollection<EvolutionDescriptorDefinition> _descriptorView;
    private readonly SortedDictionary<string, EvolutionArchiveEntry<TGenome>> _cells = new(StringComparer.Ordinal);
    private readonly int _capacity;

    /// <summary>Initializes an archive.</summary>
    /// <param name="descriptors">One or more uniquely named descriptor definitions.</param>
    /// <param name="direction">The scalar quality direction for every entry.</param>
    /// <param name="capacity">Maximum occupied cells, or zero to use the full descriptor grid.</param>
    /// <param name="maximumGridCells">Safety limit for the descriptor-grid product.</param>
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

    /// <summary>Gets the compatibility hash of descriptor, direction, and capacity settings.</summary>
    public string DefinitionHash { get; }

    /// <inheritdoc/>
    public int Count => _cells.Count;

    /// <inheritdoc/>
    public long Version { get; private set; }

    /// <inheritdoc/>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Entries => _cells.Values.ToArray();

    /// <inheritdoc/>
    public EvolutionArchiveEntry<TGenome>? Best => _cells.Count == 0
        ? null
        : _cells.Values.OrderBy(entry => entry, Comparer).First();

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
            Version++;
            return EvolutionArchiveInsertionResult.Replaced;
        }

        if (_cells.Count < _capacity)
        {
            _cells.Add(key.StableKey, candidateEntry);
            Version++;
            return EvolutionArchiveInsertionResult.Inserted;
        }

        EvolutionArchiveEntry<TGenome> worst = _cells.Values.OrderByDescending(entry => entry, Comparer).First();
        if (Comparer.Compare(candidateEntry, worst) >= 0) return EvolutionArchiveInsertionResult.NotImproved;
        _cells.Remove(worst.Cell.StableKey);
        _cells.Add(key.StableKey, candidateEntry);
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

    private IComparer<EvolutionArchiveEntry<TGenome>> Comparer => new EntryComparer(Direction);

    private sealed class EntryComparer : IComparer<EvolutionArchiveEntry<TGenome>>
    {
        private readonly EvolutionOptimizationDirection _direction;

        public EntryComparer(EvolutionOptimizationDirection direction) => _direction = direction;

        public int Compare(EvolutionArchiveEntry<TGenome>? x, EvolutionArchiveEntry<TGenome>? y)
        {
            if (ReferenceEquals(x, y)) return 0;
            if (x is null) return 1;
            if (y is null) return -1;
            int quality = _direction == EvolutionOptimizationDirection.Maximize
                ? Nullable.Compare(y.Evaluation.Quality, x.Evaluation.Quality)
                : Nullable.Compare(x.Evaluation.Quality, y.Evaluation.Quality);
            if (quality != 0) return quality;
            int genome = StringComparer.Ordinal.Compare(x.Evaluation.GenomeId, y.Evaluation.GenomeId);
            if (genome != 0) return genome;
            int cell = x.Cell.CompareTo(y.Cell);
            if (cell != 0) return cell;
            return x.Evaluation.EvaluationId.CompareTo(y.Evaluation.EvaluationId);
        }
    }
}
