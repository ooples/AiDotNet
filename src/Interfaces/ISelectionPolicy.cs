using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Selects parents and inspirations from an archive.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The engine calls <see cref="Select"/> once per variation proposal with a proposal-local
/// <see cref="StableRandom"/> stream derived from the run seed and the evaluation identifier, so an
/// implementation that draws randomness only from that stream is reproducible and safe to resume from a
/// checkpoint. <see cref="Id"/> and <see cref="VersionHash"/> are folded into the engine's compatibility
/// hash; change <see cref="VersionHash"/> whenever the selection behavior changes so that stale checkpoints
/// are rejected instead of being resumed with different semantics. Policies that need to learn from
/// evaluation outcomes additionally implement <see cref="IOutcomeAwareEvolutionSelectionPolicy{TGenome}"/>.
/// </para>
/// <para><b>For Beginners:</b> Evolution improves solutions by copying good ones and changing them a little.
/// A selection policy decides which existing solution (the "parent") gets copied next and which other strong
/// solutions (the "inspirations") the variation operator may borrow ideas from. In a MAP-Elites archive
/// (Mouret and Clune, 2015) the parent is usually drawn uniformly from the occupied cells so that unusual but
/// promising regions keep getting attention, rather than always picking the single best score. Use the
/// built-in <see cref="UniformEvolutionSelectionPolicy{TGenome}"/> or
/// <see cref="DoubleEvolutionSelectionPolicy{TGenome}"/> unless you have a specific reason to bias which
/// elites are recombined.</para>
/// </remarks>
public interface ISelectionPolicy<TGenome>
{
    /// <summary>Gets a stable policy identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Selects a parent and optional inspiration set.</summary>
    /// <param name="archive">The archive to draw elites from.</param>
    /// <param name="random">A proposal-local stable random stream owned by the caller.</param>
    /// <param name="inspirationCount">The maximum number of inspiration elites to return; zero or greater.</param>
    /// <returns>The selection, or <c>null</c> when the archive holds no elite to select.</returns>
    EvolutionSelection<TGenome>? Select(IEvolutionArchive<TGenome> archive, StableRandom random, int inspirationCount);
}
