using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Selects a uniformly random occupied cell and distinct uniformly sampled inspirations.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// This is the engine's default <see cref="ISelectionPolicy{TGenome}"/> and the selection rule of the original
/// MAP-Elites algorithm (Mouret and Clune, 2015, "Illuminating search spaces by mapping elites", arXiv:1504.04909):
/// every occupied cell is equally likely to be chosen as the parent, regardless of its quality. Inspirations are drawn
/// without replacement from the remaining entries using a partial Fisher-Yates shuffle, so at most
/// <c>inspirationCount</c> distinct elites are returned and the parent is never among them. The policy is stateless and
/// consumes only the supplied <see cref="StableRandom"/> stream, which keeps proposals reproducible and checkpointable.
/// </para>
/// <para><b>For Beginners:</b> A selection policy decides which existing solutions get to be parents for the next
/// round of mutations. This one is the simplest fair choice: it picks any filled cell of the archive with equal
/// probability, like drawing a name from a hat, and then draws a few more distinct names to serve as inspirations that
/// the variation operator may borrow ideas from. Uniform selection is a strong default for quality-diversity search
/// because it keeps exploring every region of the behavior map instead of piling effort onto the current best cell.
/// Choose a different policy, such as a curiosity-driven one, when some cells are clearly more productive than others
/// and you want the search to concentrate there.</para>
/// <para>
/// Each call costs O(N) to enumerate the archive's entries plus O(k) swaps for k inspirations, where N is the number of
/// occupied cells.
/// </para>
/// </remarks>
public sealed class UniformEvolutionSelectionPolicy<TGenome> : ISelectionPolicy<TGenome>
{
    /// <inheritdoc/>
    public string Id => "uniform-elites";

    /// <inheritdoc/>
    public string VersionHash => "uniform-elites-v1";

    /// <inheritdoc/>
    public EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount)
    {
        Guard.NotNull(archive);
        Guard.NotNull(random);
        if (inspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(inspirationCount));
        EvolutionArchiveEntry<TGenome>? parent = archive.Sample(random);
        if (parent is null) return null;

        EvolutionArchiveEntry<TGenome>[] candidates = archive.Entries
            .Where(entry => entry.Evaluation.GenomeId != parent.Evaluation.GenomeId)
            .ToArray();
        int take = Math.Min(inspirationCount, candidates.Length);
        var inspirations = new List<EvolutionArchiveEntry<TGenome>>(take);
        for (int i = 0; i < take; i++)
        {
            int selected = random.NextInt(i, candidates.Length);
            EvolutionArchiveEntry<TGenome> temporary = candidates[i];
            candidates[i] = candidates[selected];
            candidates[selected] = temporary;
            inspirations.Add(candidates[i]);
        }
        return new EvolutionSelection<TGenome>(parent, inspirations.AsReadOnly());
    }
}
