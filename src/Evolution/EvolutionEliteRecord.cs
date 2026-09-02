using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One entry of the engine's cross-island global elite index.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// An <see cref="EvolutionArchiveEntry{TGenome}"/> knows its cell, candidate, and evaluation but not which island it
/// was produced on, because each island owns a separate archive. The global elite index spans every island, so each
/// of its records pairs the entry with the <see cref="Island"/> that produced it. Records are immutable and are
/// copied verbatim into checkpoints, so a resumed run reconstructs the same index in the same order.
/// </para>
/// <para><b>For Beginners:</b> The engine keeps a small leaderboard of the strongest candidates found anywhere in
/// the run, across all islands. This class is one row of that leaderboard: it holds the candidate, its score and
/// behaviour cell, and the island number it came from. You get the whole list back from
/// <see cref="EvolutionRunResult{TGenome}.GlobalElites"/> after a run, which is the quickest way to see the overall
/// top performers without walking every island archive yourself.</para>
/// </remarks>
public sealed class EvolutionEliteRecord<TGenome>
{
    /// <summary>Initializes an elite record.</summary>
    /// <param name="island">The zero-based island the entry was produced on.</param>
    /// <param name="entry">The archive entry, including its cell, candidate, and completed evaluation.</param>
    /// <exception cref="ArgumentNullException"><paramref name="entry"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="island"/> is negative.</exception>
    public EvolutionEliteRecord(int island, EvolutionArchiveEntry<TGenome> entry)
    {
        Guard.NotNull(entry);
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        Island = island;
        Entry = entry;
    }

    /// <summary>Gets the zero-based island the entry was produced on.</summary>
    public int Island { get; }

    /// <summary>Gets the archive entry.</summary>
    public EvolutionArchiveEntry<TGenome> Entry { get; }
}
