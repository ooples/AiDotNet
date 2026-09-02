using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Allows an archive implementation to restore an exact versioned checkpoint snapshot.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine serializes each island archive as its entries plus its <c>Version</c> counter. On resume it obtains
/// fresh archives from the archive factory, verifies that they are empty, and calls <see cref="Restore"/> on each one
/// so that the restored run reproduces the state hash of the run that wrote the checkpoint. Implementations must rebuild
/// their internal indexes from the supplied entries alone and adopt the supplied version verbatim rather than recounting
/// insertions, because the version participates in the run state hash. <see cref="MapElitesArchive{TGenome}"/> is the
/// built-in implementation.
/// </para>
/// <para><b>For Beginners:</b> A checkpoint is like a saved game for a long evolution run: if the process is stopped,
/// the run can pick up where it left off instead of starting over. The archive is the part of that saved state which
/// holds the best solution found for every behavior cell, and this interface is how the engine loads a saved archive
/// back into memory. If you write your own archive and want checkpoint/resume to work with it, implement this interface
/// in addition to <see cref="IEvolutionArchive{TGenome}"/>; if you only ever run without checkpoints, the plain archive
/// interface is enough. The engine only ever calls <see cref="Restore"/> on an archive that is still empty, so an
/// implementation can treat a non-empty archive as a programming error.</para>
/// </remarks>
public interface ICheckpointableEvolutionArchive<TGenome> : IEvolutionArchive<TGenome>
{
    /// <summary>Restores entries and the exact archive version into an empty archive.</summary>
    /// <param name="entries">The elites captured at checkpoint time.</param>
    /// <param name="version">
    /// The archive version counter captured at checkpoint time. It must be at least the number of entries, because
    /// each restored elite counts as one insertion.
    /// </param>
    void Restore(IReadOnlyList<EvolutionArchiveEntry<TGenome>> entries, long version);
}
