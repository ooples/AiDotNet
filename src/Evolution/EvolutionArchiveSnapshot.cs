using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>An immutable point-in-time archive view safe to expose outside the single-writer engine.</summary>
/// <typeparam name="TGenome">The task-specific immutable genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine mutates its live <see cref="MapElitesArchive{TGenome}"/> instances from a single writer, so handing a
/// live reference to observers, reporting code, or another thread would expose a moving target. This class copies
/// every field of an <see cref="IEvolutionArchiveView{TGenome}"/> in one pass: the descriptors, the entries sorted by
/// <see cref="EvolutionCellKey.StableKey"/>, the definition hash, direction, and version, and a precomputed
/// <see cref="Best"/> that breaks quality ties by canonical genome id. Entries are themselves immutable, so the copy
/// is shallow yet cannot change afterwards.
/// </para>
/// <para>
/// The constructor validates the source rather than trusting it: descriptors and entries cannot be null, every entry
/// must be a completed evaluation with the archive's optimization direction, the reported
/// <see cref="IEvolutionArchiveView{TGenome}.Count"/> must match the number of entries, and cell keys must be
/// unique. Copying costs O(n log n) for n occupied cells, and <see cref="Get"/> is O(1) through a dictionary keyed by
/// stable cell key.
/// </para>
/// <para><b>For Beginners:</b> A snapshot is a photograph of the archive at one moment. The real archive keeps
/// changing while the search runs, so if you want to print a results table, compare two points in time, or hand the
/// elites to another thread, take a snapshot and work from that instead of from the live object. For example, an
/// <see cref="IEvolutionObserver{TGenome}"/> that receives an <c>ArchiveChanged</c> event can wrap the current view
/// in a snapshot and store it; the stored copy stays exactly as it was even after the search moves on. This is also
/// the type behind each island in <see cref="EvolutionRunResult{TGenome}.Islands"/> once a run finishes.</para>
/// </remarks>
public sealed class EvolutionArchiveSnapshot<TGenome> : IEvolutionArchiveView<TGenome>
{
    private readonly ReadOnlyCollection<EvolutionDescriptorDefinition> _descriptors;
    private readonly ReadOnlyCollection<EvolutionArchiveEntry<TGenome>> _entries;
    private readonly IReadOnlyDictionary<string, EvolutionArchiveEntry<TGenome>> _byCell;

    /// <summary>Copies a point-in-time archive view.</summary>
    /// <param name="source">The archive view to copy.</param>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// The view has invalid metadata, contains null descriptors or entries, or contains inconsistent entries.
    /// </exception>
    public EvolutionArchiveSnapshot(IEvolutionArchiveView<TGenome> source)
    {
        Guard.NotNull(source);
        if (string.IsNullOrWhiteSpace(source.DefinitionHash))
            throw new ArgumentException("Archive definition hashes cannot be empty.", nameof(source));
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), source.Direction) || source.Version < 0)
            throw new ArgumentException("The archive view has invalid metadata.", nameof(source));
        EvolutionDescriptorDefinition[] descriptors = source.Descriptors.ToArray();
        EvolutionArchiveEntry<TGenome>[] unorderedEntries = source.Entries.ToArray();
        if (descriptors.Any(descriptor => descriptor is null) || unorderedEntries.Any(entry => entry is null))
            throw new ArgumentException("Archive views cannot contain null values.", nameof(source));
        EvolutionArchiveEntry<TGenome>[] entries = unorderedEntries
            .OrderBy(entry => entry.Cell.StableKey, StringComparer.Ordinal)
            .ToArray();
        if (source.Count != entries.Length || entries.Any(entry =>
                entry.Evaluation.Status != EvolutionEvaluationStatus.Completed ||
                entry.Evaluation.Direction != source.Direction))
            throw new ArgumentException("The archive view contains inconsistent entries.", nameof(source));

        _descriptors = Array.AsReadOnly(descriptors);
        _entries = Array.AsReadOnly(entries);
        _byCell = new ReadOnlyDictionary<string, EvolutionArchiveEntry<TGenome>>(entries.ToDictionary(
            entry => entry.Cell.StableKey, entry => entry, StringComparer.Ordinal));
        DefinitionHash = source.DefinitionHash.Trim();
        Direction = source.Direction;
        Version = source.Version;
        Best = Direction == EvolutionOptimizationDirection.Maximize
            ? entries.OrderByDescending(entry => entry.Evaluation.Quality)
                .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal).FirstOrDefault()
            : entries.OrderBy(entry => entry.Evaluation.Quality)
                .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal).FirstOrDefault();
    }

    /// <inheritdoc/>
    public IReadOnlyList<EvolutionDescriptorDefinition> Descriptors => _descriptors;

    /// <inheritdoc/>
    public string DefinitionHash { get; }

    /// <inheritdoc/>
    public EvolutionOptimizationDirection Direction { get; }

    /// <inheritdoc/>
    public int Count => _entries.Count;

    /// <inheritdoc/>
    public long Version { get; }

    /// <inheritdoc/>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Entries => _entries;

    /// <inheritdoc/>
    public EvolutionArchiveEntry<TGenome>? Best { get; }

    /// <inheritdoc/>
    public EvolutionArchiveEntry<TGenome>? Get(EvolutionCellKey cell)
    {
        Guard.NotNull(cell);
        return _byCell.TryGetValue(cell.StableKey, out EvolutionArchiveEntry<TGenome>? entry) ? entry : null;
    }
}
