using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Proposes mutation, crossover, or another variation without evaluator knowledge.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IVariationOperator<TGenome>
{
    /// <summary>Gets a stable operator identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Proposes a new genome.</summary>
    ValueTask<TGenome> ProposeAsync(EvolutionVariationContext<TGenome> context, CancellationToken cancellationToken = default);
}
