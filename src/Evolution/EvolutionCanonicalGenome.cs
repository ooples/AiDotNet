using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Associates an immutable task genome with its stable canonical identity.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// <c>IEvolutionTask&lt;TGenome&gt;.CanonicalizeAsync</c> produces this pair after validating and snapshotting a
/// proposed genome. The identity must depend only on what the genome means, not on how it happens to be
/// represented: two genomes that differ only in dictionary ordering, whitespace, or floating-point formatting must
/// map to the same <see cref="Id"/>. The engine uses the identity for duplicate detection, the evaluation cache,
/// deterministic archive tie-breaking, lineage records, and checkpoint payloads. The ID is trimmed on construction
/// and must not be blank; the genome must not be <c>null</c>.
/// </para>
/// <para><b>For Beginners:</b> Think of the ID as a fingerprint for a candidate solution. The configuration
/// <c>{ learningRate = 0.01, depth = 5 }</c> and <c>{ depth = 5, learningRate = 0.01 }</c> describe the same
/// model, so they should share one fingerprint; that is how the engine notices it has already scored a proposal and
/// avoids paying for the same training run twice. When you implement a task, build the ID from the normalized,
/// sorted content of the genome (a hash of its canonical text is typical) rather than from object references or
/// creation time. Treat the wrapped genome as read-only, because archives and checkpoints hold on to it.</para>
/// </remarks>
public sealed class EvolutionCanonicalGenome<TGenome>
{
    /// <summary>Initializes a canonical genome.</summary>
    /// <param name="genome">An immutable genome snapshot.</param>
    /// <param name="id">A stable identity for its semantics rather than its incidental representation.</param>
    /// <exception cref="ArgumentNullException"><paramref name="genome"/> or <paramref name="id"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> is empty or white space.</exception>
    public EvolutionCanonicalGenome(TGenome genome, string id)
    {
        if (genome is null) throw new ArgumentNullException(nameof(genome));
        Guard.NotNullOrWhiteSpace(id);
        Genome = genome;
        Id = id.Trim();
    }

    /// <summary>Gets the immutable task-specific genome.</summary>
    public TGenome Genome { get; }

    /// <summary>Gets the stable canonical genome identity.</summary>
    public string Id { get; }
}
