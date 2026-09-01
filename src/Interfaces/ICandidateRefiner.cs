using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Optionally improves a proposed genome while returning a new immutable snapshot.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface ICandidateRefiner<TGenome>
{
    /// <summary>Gets a stable refiner identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash for checkpoint compatibility.</summary>
    string VersionHash { get; }

    /// <summary>Returns a refined genome without modifying the input object.</summary>
    ValueTask<TGenome> RefineAsync(TGenome genome, EvolutionRefinementContext context, CancellationToken cancellationToken = default);
}
