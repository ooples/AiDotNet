namespace AiDotNet.Evolution;

/// <summary>One immutable elite transfer between islands.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A migration is produced by an <see cref="AiDotNet.Interfaces.IMigrationPolicy{TGenome}"/> and consumed by
/// the evolution engine. It names a source island, a different destination island, and the exact archive
/// <see cref="Entry"/> to copy. The constructor checks that both indices are non-negative and distinct; the
/// engine additionally verifies that the indices are within the configured island count and that the entry is
/// genuinely present in the source island before offering the entry's candidate and evaluation to the
/// destination archive through its normal insertion rules. The entry is shared rather than cloned, and because
/// it is immutable a migration never alters the source archive.
/// </para>
/// <para><b>For Beginners:</b> When an evolutionary search runs several independent populations (islands), it
/// periodically lets good solutions travel between them so that progress made on one island can seed the
/// others. This small class is the boarding pass for one such trip: which island the elite leaves, which island
/// it arrives at, and which elite is travelling. You only create these yourself when implementing a custom
/// migration policy; otherwise the built-in ring policy creates them for you. The destination island decides
/// on arrival whether the newcomer is good enough to occupy its cell, so a migration is an offer rather than a
/// guarantee of insertion.</para>
/// </remarks>
public sealed class EvolutionMigration<TGenome>
{
    /// <summary>Initializes a migration transfer.</summary>
    /// <param name="sourceIsland">The zero-based index of the island the elite is copied from.</param>
    /// <param name="destinationIsland">The zero-based index of the island the elite is offered to; must differ from <paramref name="sourceIsland"/>.</param>
    /// <param name="entry">The immutable elite to copy.</param>
    /// <exception cref="ArgumentOutOfRangeException">An island index is negative.</exception>
    /// <exception cref="ArgumentException">The source and destination islands are the same.</exception>
    /// <exception cref="ArgumentNullException"><paramref name="entry"/> is <c>null</c>.</exception>
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
