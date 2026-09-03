using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One candidate recovered from a checkpoint, with the island and cell it occupied.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// A checkpoint exists so a run can continue, but it is also the only complete record of what a finished run
/// actually held: every elite, its genome in full, its score and descriptors, and the lineage that produced it.
/// This is one of those, in a form a reader can use without an engine, a task, or the operator that made it.
/// </para>
/// <para><b>For Beginners:</b> After a search finishes you often want to look at what it found rather than to carry
/// on searching. This is one saved candidate: the thing itself, which island and which cell of the map it sat in,
/// and how it scored.</para>
/// </remarks>
public sealed class EvolutionCheckpointEntry<TGenome>
{
    /// <summary>Initializes an entry.</summary>
    /// <param name="island">The island the entry belonged to.</param>
    /// <param name="source">Where in the checkpoint the entry was found.</param>
    /// <param name="entry">The candidate, its cell, and its evaluation.</param>
    /// <exception cref="ArgumentNullException"><paramref name="entry"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="island"/> is negative.</exception>
    public EvolutionCheckpointEntry(int island, EvolutionCheckpointEntrySource source,
        EvolutionArchiveEntry<TGenome> entry)
    {
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        Guard.NotNull(entry);
        Island = island;
        Source = source;
        Entry = entry;
    }

    /// <summary>Gets the island the entry belonged to.</summary>
    public int Island { get; }

    /// <summary>Gets where in the checkpoint the entry was found.</summary>
    public EvolutionCheckpointEntrySource Source { get; }

    /// <summary>Gets the candidate, its cell, and its evaluation.</summary>
    public EvolutionArchiveEntry<TGenome> Entry { get; }

    /// <summary>Gets the canonical identity of the candidate.</summary>
    public string GenomeId => Entry.Evaluation.GenomeId;

    /// <summary>Gets the candidate itself.</summary>
    public TGenome Genome => Entry.Candidate.CanonicalGenome.Genome;
}
