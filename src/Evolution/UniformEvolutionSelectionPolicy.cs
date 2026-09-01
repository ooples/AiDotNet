using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Selects a uniformly random occupied cell and distinct uniformly sampled inspirations.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
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
