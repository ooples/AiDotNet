using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Defines validation, canonical identity, and evaluation for a domain-specific genome.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
public interface IEvolutionTask<TGenome>
{
    /// <summary>Gets a stable task identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash that changes whenever task semantics change.</summary>
    string VersionHash { get; }

    /// <summary>Gets a version hash that changes whenever evaluator semantics or data change.</summary>
    string EvaluatorVersionHash { get; }

    /// <summary>Validates, snapshots, and canonicalizes a proposed genome.</summary>
    ValueTask<EvolutionCanonicalGenome<TGenome>> CanonicalizeAsync(TGenome genome, CancellationToken cancellationToken = default);

    /// <summary>Evaluates one canonical candidate.</summary>
    ValueTask<EvolutionTaskResult> EvaluateAsync(
        EvolutionCandidate<TGenome> candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default);
}
