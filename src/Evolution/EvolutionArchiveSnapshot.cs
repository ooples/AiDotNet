using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>An immutable point-in-time archive view safe to expose outside the single-writer engine.</summary>
/// <typeparam name="TGenome">The task-specific immutable genome type.</typeparam>
public sealed class EvolutionArchiveSnapshot<TGenome> : IEvolutionArchiveView<TGenome>
{
    private readonly ReadOnlyCollection<EvolutionDescriptorDefinition> _descriptors;
    private readonly ReadOnlyCollection<EvolutionArchiveEntry<TGenome>> _entries;
    private readonly IReadOnlyDictionary<string, EvolutionArchiveEntry<TGenome>> _byCell;

    /// <summary>Copies a point-in-time archive view.</summary>
    /// <param name="source">The archive view to copy.</param>
    public EvolutionArchiveSnapshot(IEvolutionArchiveView<TGenome> source)
    {
        Guard.NotNull(source);
        if (string.IsNullOrWhiteSpace(source.DefinitionHash))
            throw new ArgumentException("Archive definition hashes cannot be empty.", nameof(source));
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), source.Direction) || source.Version < 0)
            throw new ArgumentException("The archive view has invalid metadata.", nameof(source));
        EvolutionDescriptorDefinition[] descriptors = source.Descriptors.ToArray();
        EvolutionArchiveEntry<TGenome>[] entries = source.Entries
            .OrderBy(entry => entry.Cell.StableKey, StringComparer.Ordinal)
            .ToArray();
        if (descriptors.Any(descriptor => descriptor is null) || entries.Any(entry => entry is null))
            throw new ArgumentException("Archive views cannot contain null values.", nameof(source));
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
