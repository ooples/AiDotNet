namespace AiDotNet.Evolution;

/// <summary>A selected parent and its inspiration elites.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public sealed class EvolutionSelection<TGenome>
{
    /// <summary>Initializes a selection.</summary>
    public EvolutionSelection(EvolutionArchiveEntry<TGenome> parent, IReadOnlyList<EvolutionArchiveEntry<TGenome>> inspirations)
    {
        Parent = parent ?? throw new ArgumentNullException(nameof(parent));
        Inspirations = inspirations ?? throw new ArgumentNullException(nameof(inspirations));
    }

    /// <summary>Gets the selected parent.</summary>
    public EvolutionArchiveEntry<TGenome> Parent { get; }
    /// <summary>Gets selected inspirations.</summary>
    public IReadOnlyList<EvolutionArchiveEntry<TGenome>> Inspirations { get; }
}
