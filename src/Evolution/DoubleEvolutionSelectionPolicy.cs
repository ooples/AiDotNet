using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// Separates uniform parent selection from quality-ranked inspiration selection (double selection).
/// </summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class DoubleEvolutionSelectionPolicy<TGenome> : ISelectionPolicy<TGenome>
{
    /// <inheritdoc/>
    public string Id => "double-selection";

    /// <inheritdoc/>
    public string VersionHash => "double-selection-v1";

    /// <inheritdoc/>
    public EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount)
    {
        Guard.NotNull(archive);
        Guard.NotNull(random);
        if (inspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(inspirationCount));
        EvolutionArchiveEntry<TGenome>? parent = archive.Sample(random);
        if (parent is null) return null;
        IOrderedEnumerable<EvolutionArchiveEntry<TGenome>> ranked = archive.Direction == EvolutionOptimizationDirection.Maximize
            ? archive.Entries.OrderByDescending(entry => entry.Evaluation.Quality)
            : archive.Entries.OrderBy(entry => entry.Evaluation.Quality);
        EvolutionArchiveEntry<TGenome>[] inspirations = ranked
            .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
            .Where(entry => entry.Evaluation.GenomeId != parent.Evaluation.GenomeId)
            .Take(inspirationCount)
            .ToArray();
        return new EvolutionSelection<TGenome>(parent, Array.AsReadOnly(inspirations));
    }
}
