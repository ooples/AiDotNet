using AiDotNet.Evolution.Programs.ArtifactStore;

namespace AiDotNet.Interfaces;

/// <summary>Stores and retrieves the evidence an evaluation produced, keyed by the genome that produced it.</summary>
/// <remarks>
/// <para>
/// Artifacts are the difference between "this program scored 0.2" and "this program scored 0.2 and here is the
/// exception it threw". The reference OpenEvolve database keeps them beside the program record and splits them by
/// size (<c>openevolve/database.py</c> <c>store_artifacts</c>): content at or below
/// <c>artifact_size_threshold</c> is serialized into the record, and anything larger is written to a per-program
/// directory. This interface is the same contract expressed as a store a task or an observer owns, so artifacts do
/// not have to travel through the evolution engine's own state to be durable.
/// </para>
/// <para>
/// Implementations must be safe for concurrent use, must never execute or interpret artifact content, and must
/// bound what they accept, because artifact content is produced by code a model wrote. Retrieval is by genome
/// identifier, which is the canonical hash of the program's normalized source, so a checkpointed run can go back to
/// the evidence for any candidate it recorded. <see cref="PurgeAsync"/> is the retention hook: a long run
/// accumulates artifacts indefinitely otherwise.
/// </para>
/// <para><b>For Beginners:</b> This is a filing cabinet for whatever your candidate programs printed or wrote while
/// being scored. File it with <see cref="StoreAsync"/> under the program's identifier, look it up later with
/// <see cref="GetAsync"/>, see what is on file without reading it with <see cref="ListAsync"/>, and throw out old
/// paperwork with <see cref="PurgeAsync"/>. Without it, everything a program printed is lost the moment the run
/// ends, which is exactly when you want to read it.</para>
/// </remarks>
public interface IProgramArtifactStore
{
    /// <summary>Stores artifacts for one genome, merging with anything already stored under the same names.</summary>
    /// <param name="genomeId">The canonical genome identifier the artifacts belong to.</param>
    /// <param name="artifacts">The artifacts to store; an empty sequence is a no-op.</param>
    /// <param name="cancellationToken">Cancels the operation.</param>
    /// <returns>Descriptors for the artifacts as they were actually stored, including any truncation.</returns>
    Task<IReadOnlyList<ProgramArtifactDescriptor>> StoreAsync(
        string genomeId,
        IEnumerable<ProgramArtifact> artifacts,
        CancellationToken cancellationToken = default);

    /// <summary>Retrieves every artifact stored for one genome.</summary>
    /// <param name="genomeId">The canonical genome identifier.</param>
    /// <param name="cancellationToken">Cancels the operation.</param>
    /// <returns>The artifacts in ordinal name order; an empty list when the genome has none.</returns>
    Task<IReadOnlyList<ProgramArtifact>> GetAsync(string genomeId, CancellationToken cancellationToken = default);

    /// <summary>Retrieves one named artifact for one genome.</summary>
    /// <param name="genomeId">The canonical genome identifier.</param>
    /// <param name="name">The artifact's label.</param>
    /// <param name="cancellationToken">Cancels the operation.</param>
    /// <returns>The artifact, or <c>null</c> when the genome has no artifact with that name.</returns>
    Task<ProgramArtifact?> GetAsync(string genomeId, string name, CancellationToken cancellationToken = default);

    /// <summary>Lists what is stored for one genome without reading the content.</summary>
    /// <param name="genomeId">The canonical genome identifier.</param>
    /// <param name="cancellationToken">Cancels the operation.</param>
    /// <returns>Descriptors in ordinal name order; an empty list when the genome has none.</returns>
    Task<IReadOnlyList<ProgramArtifactDescriptor>> ListAsync(string genomeId, CancellationToken cancellationToken = default);

    /// <summary>Removes every artifact stored for one genome.</summary>
    /// <param name="genomeId">The canonical genome identifier.</param>
    /// <param name="cancellationToken">Cancels the operation.</param>
    /// <returns><c>true</c> when something was removed.</returns>
    Task<bool> RemoveAsync(string genomeId, CancellationToken cancellationToken = default);

    /// <summary>Applies the retention policy, removing artifacts that are too old or beyond the retained count.</summary>
    /// <param name="utcNow">The instant to measure age against, supplied so the policy can be tested deterministically.</param>
    /// <param name="cancellationToken">Cancels the operation.</param>
    /// <returns>How many genomes had their artifacts removed.</returns>
    Task<int> PurgeAsync(DateTimeOffset utcNow, CancellationToken cancellationToken = default);
}
