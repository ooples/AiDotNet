namespace AiDotNet.Evolution;

/// <summary>Inputs available to one variation proposal.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine builds one context per variation proposal and passes it to
/// <c>IVariationOperator&lt;TGenome&gt;.ProposeAsync</c>. <see cref="Random"/> is a proposal-local
/// <see cref="StableRandom"/> stream derived from the run seed and the proposal's evaluation ID, so an operator that
/// draws all of its randomness from it is reproducible and resumable. <see cref="Parent"/> and
/// <see cref="Inspirations"/> are immutable archive entries; the operator must produce a new genome rather than mutate
/// the parent's genome in place.
/// </para>
/// <para><b>For Beginners:</b> When the engine asks your variation operator to create a new candidate, it hands over
/// this bundle of everything the operator is allowed to look at: the parent elite to start from, a few other elites it
/// may borrow ideas from (crossover-style), a random-number stream for any coin flips, and the generation and island
/// numbers for bookkeeping. Think of it as the ingredients laid out on a workbench: the operator reads them and builds a
/// new genome without changing any of them. For example, a mutation operator might copy the parent's genome, flip a few
/// entries using <see cref="Random"/>, and return the copy. Always use the supplied <see cref="Random"/> rather than a
/// global random source; otherwise the run cannot be reproduced or resumed.</para>
/// </remarks>
public sealed class EvolutionVariationContext<TGenome>
{
    /// <summary>Initializes a variation context.</summary>
    /// <param name="parent">The selected parent elite.</param>
    /// <param name="inspirations">Additional selected elites that variation may draw from; may be empty.</param>
    /// <param name="random">The proposal-local deterministic random stream.</param>
    /// <param name="generation">The non-negative logical generation of the proposal.</param>
    /// <param name="island">The non-negative zero-based target island index.</param>
    /// <exception cref="ArgumentNullException"><paramref name="parent"/>, <paramref name="inspirations"/>, or <paramref name="random"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="generation"/> or <paramref name="island"/> is negative.</exception>
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
