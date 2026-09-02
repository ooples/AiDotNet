namespace AiDotNet.Evolution;

/// <summary>A selected parent and its inspiration elites.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// An <see cref="AiDotNet.Interfaces.ISelectionPolicy{TGenome}"/> returns one instance per proposal. The engine
/// records the <see cref="Parent"/> and <see cref="Inspirations"/> identifiers in the candidate's lineage and hands
/// both to the variation operator through <see cref="EvolutionVariationContext{TGenome}"/>. The parent is the
/// archive entry the operator mutates or recombines; the inspirations are additional elites, usually from other
/// cells, that the operator may consult but is not required to use. Both are immutable archive entries, so a
/// selection can be logged or retained without copying genomes.
/// </para>
/// <para><b>For Beginners:</b> Before the engine creates a new candidate it needs a starting point, and this object
/// is that starting point. The parent is the main ingredient: the existing solution the new candidate is derived
/// from. The inspirations are a few other good solutions offered as extra context, the way a chef adjusting one
/// recipe might glance at two or three others for ideas. The number requested is
/// <see cref="AiDotNet.Configuration.EvolutionEngineOptions.InspirationCount"/>, but the list can be shorter, even
/// empty, while the archive is still small. You normally do not construct this type yourself; you receive its
/// contents inside your variation operator, or return one from a custom selection policy.</para>
/// <para>
/// The parent/inspiration split mirrors the prompt-sampling scheme of LLM-driven evolutionary code search such as
/// AlphaEvolve (Novikov et al., 2025), where one program is edited and several others are shown as inspiration; here
/// it applies to any genome type.
/// </para>
/// </remarks>
public sealed class EvolutionSelection<TGenome>
{
    /// <summary>Initializes a selection.</summary>
    /// <param name="parent">The archive entry to derive the next candidate from.</param>
    /// <param name="inspirations">
    /// Zero or more additional elites offered to the variation operator; may be empty but not <c>null</c>.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="parent"/> or <paramref name="inspirations"/> is <c>null</c>.
    /// </exception>
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
