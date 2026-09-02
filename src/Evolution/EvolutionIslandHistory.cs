using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>A bounded per-island population history with deterministic worst-first eviction.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A MAP-Elites archive only keeps the single best candidate of each behaviour cell, so every other completed
/// evaluation is discarded the moment it loses its cell. This history optionally retains those candidates as well,
/// up to <see cref="Capacity"/> per island, which is what lets a selection policy or a report look beyond the
/// current elites. Because the bound is enforced on every add, the memory cost is provably
/// <c>IslandCount * Capacity</c> entries no matter how long a run continues.
/// </para>
/// <para>
/// Eviction follows the rule OpenEvolve uses, made deterministic: homeless entries, meaning those that no longer own
/// a cell of the island archive, are removed before cell owners, each group worst first by quality in the configured
/// <see cref="Direction"/> with ties broken by ordinal genome identifier, cell, and evaluation identifier. The
/// island's current best entry and the entry being added are never evicted. Adding costs O(1) and one eviction costs
/// O(h + e) for h retained entries and e occupied cells, so a full history adds one linear pass per commit and
/// nothing at all while <see cref="Capacity"/> is zero, which disables the history.
/// </para>
/// <para><b>For Beginners:</b> The archive is a wall of pigeonholes that keeps only the champion of each hole, so a
/// perfectly decent solution vanishes as soon as a better one lands in the same hole. Turning on a history keeps a
/// bounded scrapbook of those runners-up per island, which is useful for reporting and for selection policies that
/// want a wider pool than the champions alone. When the scrapbook is full the engine throws away the weakest
/// non-champion first, never the island's best and never the entry it just added, so the contents are predictable
/// and repeatable. Leave <c>HistorySize</c> at zero unless you need it; that is the default and costs no
/// memory.</para>
/// </remarks>
public sealed class EvolutionIslandHistory<TGenome>
{
    private readonly Dictionary<string, EvolutionArchiveEntry<TGenome>> _entries = new(StringComparer.Ordinal);
    private readonly int _capacity;

    /// <summary>Initializes an empty history.</summary>
    /// <param name="capacity">The maximum number of retained entries; zero disables the history.</param>
    /// <param name="direction">The quality direction shared by every island archive.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="capacity"/> is negative or <paramref name="direction"/> is undefined.
    /// </exception>
    public EvolutionIslandHistory(int capacity, EvolutionOptimizationDirection direction)
    {
        if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        _capacity = capacity;
        Direction = direction;
    }

    /// <summary>Gets the maximum number of retained entries; zero means the history is disabled.</summary>
    public int Capacity => _capacity;

    /// <summary>Gets the quality direction used to order entries.</summary>
    public EvolutionOptimizationDirection Direction { get; }

    /// <summary>Gets the current number of retained entries.</summary>
    public int Count => _entries.Count;

    /// <summary>Gets every retained entry in best-first order.</summary>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Entries =>
        _entries.Values.OrderBy(entry => entry, EvolutionEntryOrdering.BestFirst<TGenome>(Direction)).ToArray();

    /// <summary>Returns whether a canonical genome identifier is currently retained.</summary>
    /// <param name="genomeId">The canonical genome identifier to look for.</param>
    /// <returns><c>true</c> when the identifier is retained.</returns>
    public bool Contains(string genomeId)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        return _entries.ContainsKey(genomeId);
    }

    /// <summary>Adds one completed entry and evicts worst-first until the capacity bound holds again.</summary>
    /// <param name="entry">The entry to retain.</param>
    /// <param name="cellOwnerGenomeIds">Canonical identifiers that currently own a cell of the island archive.</param>
    /// <param name="protectedGenomeId">An additional identifier that must never be evicted, such as the island best.</param>
    /// <returns>The evicted entries in eviction order; empty when nothing was evicted.</returns>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="entry"/> or <paramref name="cellOwnerGenomeIds"/> is <c>null</c>.
    /// </exception>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Add(
        EvolutionArchiveEntry<TGenome> entry,
        IReadOnlyCollection<string> cellOwnerGenomeIds,
        string? protectedGenomeId)
    {
        Guard.NotNull(entry);
        Guard.NotNull(cellOwnerGenomeIds);
        if (_capacity == 0) return Array.Empty<EvolutionArchiveEntry<TGenome>>();
        string addedId = entry.Evaluation.GenomeId;
        if (!_entries.ContainsKey(addedId)) _entries.Add(addedId, entry);

        var evicted = new List<EvolutionArchiveEntry<TGenome>>();
        var owners = new HashSet<string>(cellOwnerGenomeIds, StringComparer.Ordinal);
        while (_entries.Count > _capacity)
        {
            EvolutionArchiveEntry<TGenome>? victim = SelectVictim(owners, addedId, protectedGenomeId);
            if (victim is null) break;
            _entries.Remove(victim.Evaluation.GenomeId);
            evicted.Add(victim);
        }
        return evicted.AsReadOnly();
    }

    /// <summary>Replaces the whole history with a checkpointed entry sequence.</summary>
    /// <param name="entries">The entries captured at checkpoint time.</param>
    /// <exception cref="ArgumentNullException"><paramref name="entries"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">The history is not empty.</exception>
    /// <exception cref="System.IO.InvalidDataException">
    /// The sequence exceeds <see cref="Capacity"/> or repeats a genome identifier.
    /// </exception>
    public void Restore(IEnumerable<EvolutionArchiveEntry<TGenome>> entries)
    {
        Guard.NotNull(entries);
        if (_entries.Count != 0) throw new InvalidOperationException("Only an empty island history can be restored.");
        foreach (EvolutionArchiveEntry<TGenome> entry in entries)
        {
            if (entry is null) throw new InvalidDataException("The checkpoint island history contains a null entry.");
            if (_entries.Count >= _capacity) throw new InvalidDataException("The checkpoint island history exceeds the configured capacity.");
            if (_entries.ContainsKey(entry.Evaluation.GenomeId))
                throw new InvalidDataException("The checkpoint island history repeats a genome identifier.");
            _entries.Add(entry.Evaluation.GenomeId, entry);
        }
    }

    private EvolutionArchiveEntry<TGenome>? SelectVictim(HashSet<string> owners, string addedId, string? protectedGenomeId)
    {
        EvolutionArchiveEntry<TGenome>? victim = null;
        bool victimOwnsCell = false;
        foreach (EvolutionArchiveEntry<TGenome> candidate in _entries.Values)
        {
            string candidateId = candidate.Evaluation.GenomeId;
            if (string.Equals(candidateId, addedId, StringComparison.Ordinal)) continue;
            if (protectedGenomeId is not null && string.Equals(candidateId, protectedGenomeId, StringComparison.Ordinal)) continue;
            bool ownsCell = owners.Contains(candidateId);
            if (victim is null || IsWorseVictim(ownsCell, candidate, victimOwnsCell, victim))
            {
                victim = candidate;
                victimOwnsCell = ownsCell;
            }
        }
        return victim;
    }

    private bool IsWorseVictim(bool candidateOwnsCell, EvolutionArchiveEntry<TGenome> candidate,
        bool incumbentOwnsCell, EvolutionArchiveEntry<TGenome> incumbent)
    {
        if (candidateOwnsCell != incumbentOwnsCell) return !candidateOwnsCell;
        return EvolutionEntryOrdering.Compare(Direction, candidate, incumbent) > 0;
    }
}
