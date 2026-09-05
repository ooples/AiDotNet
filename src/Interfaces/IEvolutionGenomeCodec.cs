namespace AiDotNet.Interfaces;

/// <summary>Serializes task genomes for portable checkpoints.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine never inspects genome contents. When a checkpoint store is configured it asks the codec to turn
/// every seed genome and every archived elite into a string, and on resume it asks the codec to rebuild those
/// genomes and to re-serialize the supplied seeds so they can be compared with the checkpointed seed sequence.
/// <see cref="Id"/> and <see cref="VersionHash"/> are folded into the engine's compatibility hash, so changing
/// either one deliberately invalidates older checkpoints instead of silently loading payloads that a newer codec
/// would interpret differently.
/// </para>
/// <para><b>For Beginners:</b> A genome is the task-specific object being evolved: a parameter vector, a program
/// text, a network layout. To pause a run and pick it up later, the engine must write those objects to storage,
/// but it only sees them as an opaque <typeparamref name="TGenome"/> and cannot know how. A codec is the small
/// adapter you supply that does know: <see cref="Serialize"/> writes one genome as text (JSON is the natural
/// choice) and <see cref="Deserialize"/> reads it back. Think of it as the save-file format for your genome type.
/// You need one only when you pass a checkpoint store to the engine or enable <c>Resume</c>; runs without
/// persistence never call it.</para>
/// <para>
/// Contract: <c>Deserialize(Serialize(g))</c> must produce a genome that the task canonicalizes to the same
/// identity as <c>g</c>; <see cref="Serialize"/> must be deterministic for identical genomes because payloads are
/// compared with ordinal string equality; and neither method may return <c>null</c>, which the engine reports as a
/// checkpoint fault. Payloads should be culture-independent and free of machine-specific paths so checkpoints
/// remain portable between hosts.
/// </para>
/// </remarks>
public interface IEvolutionGenomeCodec<TGenome>
{
    /// <summary>Gets a stable codec identifier.</summary>
    /// <remarks>Part of the checkpoint compatibility hash; keep it constant for the lifetime of a payload format.</remarks>
    string Id { get; }

    /// <summary>Gets a codec version hash.</summary>
    /// <remarks>
    /// Change this whenever the payload format changes incompatibly so older checkpoints are refused rather than misread.
    /// </remarks>
    string VersionHash { get; }

    /// <summary>Serializes an immutable genome snapshot.</summary>
    /// <param name="genome">The genome to encode; never <c>null</c>.</param>
    /// <returns>A non-null, deterministic, culture-independent text payload.</returns>
    string Serialize(TGenome genome);

    /// <summary>Deserializes an immutable genome snapshot.</summary>
    /// <param name="payload">Text previously produced by <see cref="Serialize"/> under the same <see cref="VersionHash"/>.</param>
    /// <returns>The reconstructed genome; never <c>null</c>.</returns>
    /// <exception cref="System.IO.InvalidDataException">The payload cannot be decoded.</exception>
    TGenome Deserialize(string payload);
}
