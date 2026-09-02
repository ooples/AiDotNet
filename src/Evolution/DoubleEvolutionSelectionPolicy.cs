using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>
/// Separates uniform parent selection from quality-ranked inspiration selection (double selection).
/// </summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The parent is sampled uniformly from the archive's occupied cells through
/// <see cref="IEvolutionArchive{TGenome}.Sample"/>, preserving the exploration pressure of MAP-Elites
/// (Mouret and Clune, 2015). The inspirations are the <c>inspirationCount</c> best elites by quality in the
/// archive's optimization direction, with ties broken by ordinal genome identifier and the parent's genome
/// excluded. Given the same archive contents and the same random stream the result is identical, so the policy is
/// checkpoint-safe. Ranking costs O(n log n) in the number of occupied cells per call, which is negligible next to
/// a typical evaluation but grows with archive size. This policy keeps no state; adaptive alternatives implement
/// <see cref="IOutcomeAwareEvolutionSelectionPolicy{TGenome}"/>.
/// </para>
/// <para><b>For Beginners:</b> When evolution creates a new candidate it usually starts from one existing solution
/// (the parent) and may also look at a few other strong solutions for ideas (the inspirations). This policy picks
/// the parent at random from every occupied archive cell, so odd corners of the search space still get explored,
/// but it always chooses the top-scoring elites as inspirations, so the variation operator is steered toward what
/// currently works best. It is the natural choice when your variation operator can combine information from
/// several sources, such as a prompt-driven code mutator that is shown one program to edit plus several
/// high-scoring programs to learn from. If your operator ignores inspirations,
/// <see cref="UniformEvolutionSelectionPolicy{TGenome}"/> performs the same parent sampling with less work.</para>
/// </remarks>
public sealed class DoubleEvolutionSelectionPolicy<TGenome> : ISelectionPolicy<TGenome>
{
    /// <inheritdoc/>
    public string Id => "double-selection";

    /// <inheritdoc/>
    public string VersionHash => "double-selection-v1";

    /// <inheritdoc/>
    public EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount)
    {
        Guard.NotNull(archive);
        Guard.NotNull(random);
        if (inspirationCount < 0) throw new ArgumentOutOfRangeException(nameof(inspirationCount));
        EvolutionArchiveEntry<TGenome>? parent = archive.Sample(random);
        if (parent is null) return null;
        IOrderedEnumerable<EvolutionArchiveEntry<TGenome>> ranked = archive.Direction == EvolutionOptimizationDirection.Maximize
            ? archive.Entries.OrderByDescending(entry => entry.Evaluation.Quality)
            : archive.Entries.OrderBy(entry => entry.Evaluation.Quality);
        EvolutionArchiveEntry<TGenome>[] inspirations = ranked
            .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
            .Where(entry => entry.Evaluation.GenomeId != parent.Evaluation.GenomeId)
            .Take(inspirationCount)
            .ToArray();
        return new EvolutionSelection<TGenome>(parent, Array.AsReadOnly(inspirations));
    }
}
