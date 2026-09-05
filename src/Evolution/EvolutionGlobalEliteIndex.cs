using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>A bounded, incrementally maintained top-K leaderboard of the best evaluations across every island.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// Each island owns its own archive, so nothing in a MAP-Elites layout tracks the strongest candidates of the run as
/// a whole. This index does: the engine offers every completed, cell-mappable evaluation to
/// <see cref="Consider"/>, which keeps at most <see cref="Capacity"/> records ordered best first by quality in the
/// configured <see cref="Direction"/>, breaking ties by ordinal genome identifier, cell, and evaluation identifier so
/// the contents never depend on arrival order. A record whose genome identifier is already present is ignored, and
/// once the index is full a newcomer only enters by beating the current worst record, which is then dropped. The
/// backing store is a <see cref="SortedSet{T}"/>, so <see cref="Consider"/> costs O(log K) rather than the O(n) rescan
/// of the whole population that OpenEvolve performs on every archive update.
/// </para>
/// <para>
/// The index is engine-owned state: it is written into every checkpoint, restored through <see cref="Restore"/>, and
/// folded into the run's deterministic state hash, so a resumed run reproduces the leaderboard of the run that wrote
/// the checkpoint. A capacity of zero disables the index entirely, which is the engine default and keeps the memory
/// cost at nothing.
/// </para>
/// <para><b>For Beginners:</b> Quality-diversity search deliberately keeps a wide map of different solutions rather
/// than a single winner, but you usually still want to see the overall top ten at the end, and the search itself
/// benefits from being able to start again from one of them. This class is that top-ten list, kept up to date as the
/// run proceeds instead of computed by sorting everything at the end. Set <c>GlobalEliteCount</c> on the engine
/// options to switch it on, then read <see cref="EvolutionRunResult{TGenome}.GlobalElites"/> when the run
/// finishes.</para>
/// </remarks>
public sealed class EvolutionGlobalEliteIndex<TGenome>
{
    private readonly SortedSet<EvolutionEliteRecord<TGenome>> _records;
    private readonly HashSet<string> _genomeIds = new(StringComparer.Ordinal);
    private readonly int _capacity;

    /// <summary>Initializes an empty index.</summary>
    /// <param name="capacity">The maximum number of retained records; zero disables the index.</param>
    /// <param name="direction">The quality direction shared by every island archive.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="capacity"/> is negative or <paramref name="direction"/> is undefined.
    /// </exception>
    public EvolutionGlobalEliteIndex(int capacity, EvolutionOptimizationDirection direction)
    {
        if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        _capacity = capacity;
        Direction = direction;
        _records = new SortedSet<EvolutionEliteRecord<TGenome>>(new RecordComparer(direction));
    }

    /// <summary>Gets the maximum number of retained records; zero means the index is disabled.</summary>
    public int Capacity => _capacity;

    /// <summary>Gets the quality direction used to order records.</summary>
    public EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets the current number of retained records.</summary>
    public int Count => _records.Count;

    /// <summary>Gets every retained record in best-first order.</summary>
    public IReadOnlyList<EvolutionEliteRecord<TGenome>> Entries => _records.ToArray();

    /// <summary>Offers a completed evaluation to the index.</summary>
    /// <param name="record">The candidate record, carrying its island and archive entry.</param>
    /// <returns><c>true</c> when the record was retained, otherwise <c>false</c>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="record"/> is <c>null</c>.</exception>
    public bool Consider(EvolutionEliteRecord<TGenome> record)
    {
        Guard.NotNull(record);
        if (_capacity == 0) return false;
        if (!_genomeIds.Add(record.Entry.Evaluation.GenomeId)) return false;
        _records.Add(record);
        if (_records.Count <= _capacity) return true;

        EvolutionEliteRecord<TGenome> worst = _records.Max ?? record;
        _records.Remove(worst);
        _genomeIds.Remove(worst.Entry.Evaluation.GenomeId);
        return !ReferenceEquals(worst, record);
    }

    /// <summary>Returns the best records, at most <paramref name="count"/> of them.</summary>
    /// <param name="count">The maximum number of records to return.</param>
    /// <returns>The leading records in best-first order.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
    public IReadOnlyList<EvolutionEliteRecord<TGenome>> Top(int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        return _records.Take(count).ToArray();
    }

    /// <summary>Replaces the whole index with a checkpointed record sequence.</summary>
    /// <param name="records">The records captured at checkpoint time.</param>
    /// <exception cref="ArgumentNullException"><paramref name="records"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">The index is not empty.</exception>
    /// <exception cref="System.IO.InvalidDataException">
    /// The sequence exceeds <see cref="Capacity"/> or repeats a genome identifier.
    /// </exception>
    public void Restore(IEnumerable<EvolutionEliteRecord<TGenome>> records)
    {
        Guard.NotNull(records);
        if (_records.Count != 0) throw new InvalidOperationException("Only an empty elite index can be restored.");
        foreach (EvolutionEliteRecord<TGenome> record in records)
        {
            if (record is null) throw new InvalidDataException("The checkpoint elite index contains a null record.");
            if (_records.Count >= _capacity) throw new InvalidDataException("The checkpoint elite index exceeds the configured capacity.");
            if (!_genomeIds.Add(record.Entry.Evaluation.GenomeId))
                throw new InvalidDataException("The checkpoint elite index repeats a genome identifier.");
            _records.Add(record);
        }
    }

    private sealed class RecordComparer : IComparer<EvolutionEliteRecord<TGenome>>
    {
        private readonly EvolutionOptimizationDirection _direction;

        internal RecordComparer(EvolutionOptimizationDirection direction) => _direction = direction;

        public int Compare(EvolutionEliteRecord<TGenome>? x, EvolutionEliteRecord<TGenome>? y)
        {
            if (ReferenceEquals(x, y)) return 0;
            if (x is null) return 1;
            if (y is null) return -1;
            int entry = EvolutionEntryOrdering.Compare(_direction, x.Entry, y.Entry);
            return entry != 0 ? entry : x.Island.CompareTo(y.Island);
        }
    }
}
