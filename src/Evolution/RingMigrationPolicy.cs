using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Copies each island's best distinct elites to the next island in a directed ring.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class RingMigrationPolicy<TGenome> : IMigrationPolicy<TGenome>
{
    /// <inheritdoc/>
    public string Id => "ring-best";

    /// <inheritdoc/>
    public string VersionHash => "ring-best-v1";

    /// <inheritdoc/>
    public IReadOnlyList<EvolutionMigration<TGenome>> CreateMigrations(
        IReadOnlyList<IEvolutionArchiveView<TGenome>> islands,
        int migrantsPerIsland,
        StableRandom random)
    {
        Guard.NotNull(islands);
        Guard.Positive(migrantsPerIsland);
        Guard.NotNull(random);
        if (islands.Count < 2) return Array.Empty<EvolutionMigration<TGenome>>();

        var migrations = new List<EvolutionMigration<TGenome>>();
        for (int source = 0; source < islands.Count; source++)
        {
            IEvolutionArchiveView<TGenome> archive = islands[source];
            IEnumerable<EvolutionArchiveEntry<TGenome>> ordered = archive.Direction == EvolutionOptimizationDirection.Maximize
                ? archive.Entries.OrderByDescending(entry => entry.Evaluation.Quality).ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
                : archive.Entries.OrderBy(entry => entry.Evaluation.Quality).ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal);
            int destination = (source + 1) % islands.Count;
            foreach (EvolutionArchiveEntry<TGenome> entry in ordered.Take(migrantsPerIsland))
                migrations.Add(new EvolutionMigration<TGenome>(source, destination, entry));
        }
        return migrations;
    }
}
