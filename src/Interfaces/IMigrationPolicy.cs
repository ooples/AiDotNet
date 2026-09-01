using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Creates deterministic elite transfers between independent island archives.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IMigrationPolicy<TGenome>
{
    /// <summary>Gets a stable policy identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Creates transfers without modifying any archive.</summary>
    IReadOnlyList<EvolutionMigration<TGenome>> CreateMigrations(
        IReadOnlyList<IEvolutionArchiveView<TGenome>> islands,
        int migrantsPerIsland,
        StableRandom random);
}
