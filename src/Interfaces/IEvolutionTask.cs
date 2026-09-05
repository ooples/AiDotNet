using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Defines validation, canonical identity, and evaluation for a domain-specific genome.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// The task is the one component the <see cref="EvolutionEngine{TGenome}"/> knows nothing about in advance: it
/// decides what a valid genome looks like, which genomes count as the same candidate, and how good a candidate
/// is. <see cref="CanonicalizeAsync"/> runs before every evaluation so that two proposals with the same meaning
/// share one <see cref="EvolutionCanonicalGenome{TGenome}.Id"/>; the engine uses that identity for
/// deduplication, evaluation caching, and checkpoint compatibility. <see cref="EvaluateAsync"/> then turns one
/// canonical candidate into an <see cref="EvolutionTaskResult"/> carrying a scalar quality and the behavior
/// descriptors that place it in a quality-diversity archive.
/// </para>
/// <para>
/// <see cref="Id"/>, <see cref="VersionHash"/>, and <see cref="EvaluatorVersionHash"/> are folded into the run
/// compatibility hash, so changing any of them prevents a checkpoint produced under different semantics from
/// being resumed. Implementations should be deterministic for a given <see cref="EvolutionEvaluationContext"/>:
/// draw any randomness from <see cref="EvolutionEvaluationContext.CreateRandom"/> rather than from ambient state,
/// and report recoverable problems through <see cref="EvolutionTaskResult.Failed"/> rather than throwing, so the
/// engine can record the diagnostic and apply its retry and failure policies.
/// </para>
/// <para><b>For Beginners:</b> Think of the engine as a contest organizer and the task as the judge who knows the
/// rules of one particular game. The judge checks that an entry is legal and gives it a canonical name, so the
/// same entry submitted twice is recognized, and then scores it. For example, a task that evolves hyperparameter
/// settings would canonicalize by sorting the parameter names, use a hash of those settings as the identity, and
/// evaluate by training a model and returning its validation score plus a couple of descriptors such as model
/// family and size. Implement this interface once per problem you want to evolve; the engine, archives, and
/// checkpoints work unchanged for any task.</para>
/// </remarks>
public interface IEvolutionTask<TGenome>
{
    /// <summary>Gets a stable task identifier.</summary>
    string Id { get; }

    /// <summary>Gets a version hash that changes whenever task semantics change.</summary>
    string VersionHash { get; }

    /// <summary>Gets a version hash that changes whenever evaluator semantics or data change.</summary>
    string EvaluatorVersionHash { get; }

    /// <summary>Validates, snapshots, and canonicalizes a proposed genome.</summary>
    /// <param name="genome">The proposed genome.</param>
    /// <param name="cancellationToken">A token that cancels the operation.</param>
    /// <returns>An immutable snapshot of the genome paired with its stable canonical identity.</returns>
    ValueTask<EvolutionCanonicalGenome<TGenome>> CanonicalizeAsync(TGenome genome, CancellationToken cancellationToken = default);

    /// <summary>Evaluates one canonical candidate.</summary>
    /// <param name="candidate">The candidate to evaluate.</param>
    /// <param name="context">Deterministic per-evaluation context, including the seed stream and attempt count.</param>
    /// <param name="cancellationToken">A token that cancels the operation.</param>
    /// <returns>The terminal result, including quality and descriptors when the evaluation completed.</returns>
    ValueTask<EvolutionTaskResult> EvaluateAsync(
        EvolutionCandidate<TGenome> candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default);
}
