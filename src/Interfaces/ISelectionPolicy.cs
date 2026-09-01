using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Selects parents and inspirations from an archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface ISelectionPolicy<TGenome>
{
    /// <summary>Gets a stable policy identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Selects a parent and optional inspiration set.</summary>
    EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount);
}
