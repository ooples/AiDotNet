namespace AiDotNet.Interfaces;

/// <summary>Serializes task genomes for portable checkpoints.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionGenomeCodec<TGenome>
{
    /// <summary>Gets a stable codec identifier.</summary>
    string Id { get; }

    /// <summary>Gets a codec version hash.</summary>
    string VersionHash { get; }

    /// <summary>Serializes an immutable genome snapshot.</summary>
    string Serialize(TGenome genome);

    /// <summary>Deserializes an immutable genome snapshot.</summary>
    TGenome Deserialize(string payload);
}
