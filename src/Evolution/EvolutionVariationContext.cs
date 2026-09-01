namespace AiDotNet.Evolution;

/// <summary>Inputs available to one variation proposal.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionVariationContext<TGenome>
{
    /// <summary>Initializes a variation context.</summary>
    public EvolutionVariationContext(
        EvolutionArchiveEntry<TGenome> parent,
        IReadOnlyList<EvolutionArchiveEntry<TGenome>> inspirations,
        StableRandom random,
        long generation,
        int island)
    {
        Parent = parent ?? throw new ArgumentNullException(nameof(parent));
        Inspirations = inspirations ?? throw new ArgumentNullException(nameof(inspirations));
        Random = random ?? throw new ArgumentNullException(nameof(random));
        if (generation < 0) throw new ArgumentOutOfRangeException(nameof(generation));
        if (island < 0) throw new ArgumentOutOfRangeException(nameof(island));
        Generation = generation;
        Island = island;
    }

    /// <summary>Gets the selected parent.</summary>
    public EvolutionArchiveEntry<TGenome> Parent { get; }
    /// <summary>Gets selected inspiration elites.</summary>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Inspirations { get; }
    /// <summary>Gets a proposal-local stable random stream.</summary>
    public StableRandom Random { get; }
    /// <summary>Gets the logical generation.</summary>
    public long Generation { get; }
    /// <summary>Gets the target island.</summary>
    public int Island { get; }
}
