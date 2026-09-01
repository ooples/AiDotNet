namespace AiDotNet.Evolution;

/// <summary>One immutable elite transfer between islands.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionMigration<TGenome>
{
    /// <summary>Initializes a migration transfer.</summary>
    public EvolutionMigration(int sourceIsland, int destinationIsland, EvolutionArchiveEntry<TGenome> entry)
    {
        if (sourceIsland < 0) throw new ArgumentOutOfRangeException(nameof(sourceIsland));
        if (destinationIsland < 0) throw new ArgumentOutOfRangeException(nameof(destinationIsland));
        if (sourceIsland == destinationIsland) throw new ArgumentException("Migration requires distinct islands.", nameof(destinationIsland));
        SourceIsland = sourceIsland;
        DestinationIsland = destinationIsland;
        Entry = entry ?? throw new ArgumentNullException(nameof(entry));
    }

    /// <summary>Gets the source island.</summary>
    public int SourceIsland { get; }
    /// <summary>Gets the destination island.</summary>
    public int DestinationIsland { get; }
    /// <summary>Gets the immutable elite to copy.</summary>
    public EvolutionArchiveEntry<TGenome> Entry { get; }
}
