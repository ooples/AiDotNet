using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Associates an immutable task genome with its stable canonical identity.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionCanonicalGenome<TGenome>
{
    /// <summary>Initializes a canonical genome.</summary>
    /// <param name="genome">An immutable genome snapshot.</param>
    /// <param name="id">A stable identity for its semantics rather than its incidental representation.</param>
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
