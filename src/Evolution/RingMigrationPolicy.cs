using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Copies each island's best distinct elites to the next island in a directed ring.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// Island <c>i</c> sends its <c>migrantsPerIsland</c> highest-quality elites to island <c>(i + 1) mod n</c>,
/// honoring the archive's <see cref="EvolutionOptimizationDirection"/> and breaking quality ties by ordinal
/// canonical genome ID so the result never depends on insertion order. The policy only reads archive views and
/// returns transfer descriptions; the engine applies them to the destination archives. Fewer than two islands
/// yields no migrations. Although the contract supplies a <see cref="StableRandom"/>, this policy is fully
/// deterministic and does not consume it. Cost is <c>O(n * m log m)</c> for <c>n</c> islands holding at most
/// <c>m</c> entries each.
/// </para>
/// <para>
/// The island model (Whitley, Rana, and Heckendorn, "The island model genetic algorithm: On separability,
/// population size and convergence", 1999) runs several sub-populations in isolation so they explore different
/// regions, with occasional migration to spread good building blocks. A one-directional ring is the classic sparse
/// topology: it slows takeover by a single lineage compared with fully connected migration while still guaranteeing
/// that every island is eventually reachable from every other.
/// </para>
/// <para><b>For Beginners:</b> Imagine several research teams working on the same problem in separate rooms.
/// Left alone, each room may settle on its own style of solution, which keeps the overall search diverse. Every so
/// often this policy lets each room pass a copy of its best few results to the next room around a circle: room 0 to
/// room 1, room 1 to room 2, and the last room back to room 0. With four islands and <c>migrantsPerIsland = 2</c>,
/// one migration round produces eight transfers. Use it when you configure more than one island and want a simple,
/// reproducible way for strong candidates to spread without one island taking over immediately.</para>
/// </remarks>
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
