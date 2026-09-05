using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Everything a checkpoint holds about the candidates a run found, read back without an engine.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// Resuming is not the only reason to open a checkpoint. It is also the complete record of a finished run: every
/// elite, every retained runner-up, each one's genome in full, its score and descriptors, and the lineage that
/// produced it. This type is that record, and <see cref="TryGetAncestry"/> walks the lineage links to answer the
/// question a score alone cannot: where did this candidate come from?
/// </para>
/// <para>
/// An ancestry chain can end early, and saying so is the point. A checkpoint keeps only what a bounded archive and
/// a bounded history still hold, so a parent that lost its cell long ago is simply not there. The reference
/// implementation's checkpoint trace extraction has the same limit; reporting the chain as partial is what stops a
/// reader from mistaking a truncated ancestry for a candidate that came from nowhere.
/// </para>
/// <para><b>For Beginners:</b> Open a saved run and read what it found. Each entry is one candidate with its score
/// and its position on the map, and <see cref="TryGetAncestry"/> gives you the line of candidates it descended
/// from, oldest first, as far back as the save still records.</para>
/// </remarks>
public sealed class EvolutionCheckpointContents<TGenome>
{
    private readonly ReadOnlyCollection<EvolutionCheckpointEntry<TGenome>> _entries;
    private readonly Dictionary<string, EvolutionCheckpointEntry<TGenome>> _byGenomeId;

    /// <summary>Initializes the contents of one checkpoint.</summary>
    /// <param name="runId">The run the checkpoint belongs to.</param>
    /// <param name="sequence">The checkpoint's monotonic sequence number.</param>
    /// <param name="compatibilityHash">The configuration identity a resume must match.</param>
    /// <param name="entries">Every recovered candidate.</param>
    /// <exception cref="ArgumentNullException">An argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="runId"/> is blank or an entry is <c>null</c>.</exception>
    public EvolutionCheckpointContents(string runId, long sequence, string compatibilityHash,
        IReadOnlyList<EvolutionCheckpointEntry<TGenome>> entries)
    {
        Guard.NotNullOrWhiteSpace(runId);
        Guard.NotNull(compatibilityHash);
        Guard.NotNull(entries);
        EvolutionCheckpointEntry<TGenome>[] copied = entries.ToArray();
        if (copied.Any(entry => entry is null))
            throw new ArgumentException("Checkpoint entries cannot contain nulls.", nameof(entries));

        RunId = runId;
        Sequence = sequence;
        CompatibilityHash = compatibilityHash;
        _entries = Array.AsReadOnly(copied);

        // One candidate can be saved in several places at once, so the index keeps the most authoritative: a current
        // elite outranks a global-index copy, which outranks a runner-up kept in the history.
        _byGenomeId = new Dictionary<string, EvolutionCheckpointEntry<TGenome>>(StringComparer.Ordinal);
        foreach (EvolutionCheckpointEntry<TGenome> entry in copied.OrderBy(entry => (int)entry.Source))
            if (!_byGenomeId.ContainsKey(entry.GenomeId)) _byGenomeId[entry.GenomeId] = entry;
    }

    /// <summary>Gets the run the checkpoint belongs to.</summary>
    public string RunId { get; }

    /// <summary>Gets the checkpoint's monotonic sequence number.</summary>
    public long Sequence { get; }

    /// <summary>Gets the configuration identity a resume must match.</summary>
    public string CompatibilityHash { get; }

    /// <summary>Gets every recovered candidate, elites first.</summary>
    public IReadOnlyList<EvolutionCheckpointEntry<TGenome>> Entries => _entries;

    /// <summary>Gets the distinct candidates by canonical identity, each from its most authoritative source.</summary>
    public IReadOnlyCollection<EvolutionCheckpointEntry<TGenome>> DistinctCandidates => _byGenomeId.Values;

    /// <summary>Finds one candidate by canonical identity.</summary>
    /// <param name="genomeId">The identity to look for.</param>
    /// <returns>The candidate, or <c>null</c> when the checkpoint does not hold it.</returns>
    /// <exception cref="ArgumentException"><paramref name="genomeId"/> is blank.</exception>
    public EvolutionCheckpointEntry<TGenome>? Find(string genomeId)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        return _byGenomeId.TryGetValue(genomeId, out EvolutionCheckpointEntry<TGenome>? entry) ? entry : null;
    }

    /// <summary>Rebuilds the chain of ancestors that led to one candidate, oldest first.</summary>
    /// <param name="genomeId">The candidate to trace.</param>
    /// <param name="ancestry">The chain, oldest first, ending with the candidate itself.</param>
    /// <returns>
    /// <c>true</c> when the chain reaches a candidate with no parent, meaning the whole ancestry is present;
    /// <c>false</c> when it stops at a parent the checkpoint no longer holds.
    /// </returns>
    /// <remarks>
    /// The first parent is followed, which is the one the engine records as the candidate a variation was applied
    /// to; other parents, where an operator uses them, appear on each step's own lineage. A cycle cannot extend the
    /// chain, because a candidate already visited ends the walk.
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="genomeId"/> is blank.</exception>
    public bool TryGetAncestry(string genomeId, out IReadOnlyList<EvolutionCheckpointEntry<TGenome>> ancestry)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        var chain = new List<EvolutionCheckpointEntry<TGenome>>();
        var visited = new HashSet<string>(StringComparer.Ordinal);
        bool complete = false;

        string? current = genomeId;
        while (current is not null && visited.Add(current))
        {
            EvolutionCheckpointEntry<TGenome>? entry = Find(current);
            if (entry is null) break;
            chain.Add(entry);

            IReadOnlyList<string> parents = entry.Entry.Evaluation.Lineage.ParentIds;
            if (parents.Count == 0)
            {
                complete = true;
                break;
            }
            current = parents[0];
        }

        chain.Reverse();
        ancestry = Array.AsReadOnly(chain.ToArray());
        return complete;
    }
}
