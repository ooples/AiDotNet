using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Proposes mutation, crossover, or another variation without evaluator knowledge.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The evolution engine selects a parent elite (and optionally several inspiration elites) from an archive,
/// forks a proposal-local <see cref="StableRandom"/> stream, and packages them into an
/// <see cref="EvolutionVariationContext{TGenome}"/>. The operator turns that context into one new genome.
/// It never evaluates, validates, or canonicalizes genomes; those responsibilities belong to
/// <see cref="IEvolutionTask{TGenome}"/>, which is what lets one operator be reused across unrelated tasks.
/// </para>
/// <para>
/// Determinism contract: every random decision must come from <c>context.Random</c>, never from
/// <c>System.Random</c>, <c>Guid.NewGuid()</c>, the clock, or thread scheduling. <see cref="Id"/> is recorded
/// in each candidate's lineage, and <see cref="VersionHash"/> is folded into the run compatibility hash, so a
/// resumed checkpoint refuses to continue with an operator whose behavior has changed.
/// </para>
/// <para><b>For Beginners:</b> A variation operator is the "make a new guess from an old one" step of an
/// evolutionary search. The engine hands it one existing good solution (the parent), possibly a few other good
/// solutions for inspiration, and a private random-number stream; the operator returns a fresh candidate for the
/// engine to evaluate. For example, a mutation operator for hyperparameter search might copy the parent's settings
/// and resample only the learning rate, while a crossover operator might take the model family from the parent and
/// the tree depth from an inspiration. Use the supplied random stream for every coin flip so that re-running a
/// search with the same seed reproduces exactly the same proposals.</para>
/// </remarks>
public interface IVariationOperator<TGenome>
{
    /// <summary>Gets a stable operator identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Proposes a new genome.</summary>
    /// <param name="context">The parent, inspirations, and proposal-local random stream for this proposal.</param>
    /// <param name="cancellationToken">A token that cancels the proposal.</param>
    /// <returns>A new genome derived from the context; the engine canonicalizes and evaluates it.</returns>
    ValueTask<TGenome> ProposeAsync(EvolutionVariationContext<TGenome> context, CancellationToken cancellationToken = default);
}
