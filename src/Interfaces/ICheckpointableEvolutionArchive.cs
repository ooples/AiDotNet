using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Allows an archive implementation to restore an exact versioned checkpoint snapshot.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface ICheckpointableEvolutionArchive<TGenome> : IEvolutionArchive<TGenome>
{
    /// <summary>Restores entries and the exact archive version into an empty archive.</summary>
    void Restore(IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries, long version);
}
