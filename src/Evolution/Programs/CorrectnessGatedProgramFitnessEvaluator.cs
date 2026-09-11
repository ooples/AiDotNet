using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Runs explicit correctness checks before an expensive program fitness evaluation.</summary>
/// <remarks>
/// The correctness evaluator must report a maximization score: the fraction of required public checks passed,
/// with 1 meaning all passed. A completed score below 1 or any positive constraint violation rejects the program.
/// Use deterministic tests or trusted reference comparisons, not an LLM judge, for this gate. Both evaluators
/// remain responsible for sandboxing, timeouts and truthful cost reporting in the same units. This wrapper
/// neither executes code itself nor provides an isolation boundary. Final held-out validation belongs outside
/// the search: its cases and diagnostics must not be fed into proposal generation.
/// </remarks>
internal sealed class CorrectnessGatedProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private readonly IProgramFitnessEvaluator _correctness;
    private readonly IProgramFitnessEvaluator _fitness;

    /// <summary>Creates a fail-closed correctness gate with separately versioned checking and scoring stages.</summary>
    /// <param name="correctness">Checks returning a maximization pass fraction in [0, 1].</param>
    /// <param name="fitness">The performance or quality evaluator, called only after every check passes.</param>
    public CorrectnessGatedProgramFitnessEvaluator(IProgramFitnessEvaluator correctness, IProgramFitnessEvaluator fitness)
    {
        Guard.NotNull(correctness);
        Guard.NotNull(fitness);
        Guard.NotNullOrWhiteSpace(correctness.Id);
        Guard.NotNullOrWhiteSpace(correctness.VersionHash);
        Guard.NotNullOrWhiteSpace(fitness.Id);
        Guard.NotNullOrWhiteSpace(fitness.VersionHash);
        _correctness = correctness;
        _fitness = fitness;
        VersionHash = EvolutionHash.Combine(new[]
        {
            "correctness-gated-program-v2-complete-cost-receipts", correctness.Id, correctness.VersionHash, fitness.Id, fitness.VersionHash
        });
    }

    /// <inheritdoc/>
    public string Id => "correctness-gated-program";
    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// Returned costs include both completed stages. A rejected check retains its own diagnostics and artifacts;
    /// after a successful check, fitness metadata is authoritative and correctness metadata is not merged.
    /// Evaluators should return failure results with costs rather than throw when work has already been spent.
    /// Positive fitness-stage constraint violations also reject a completed result before archive insertion.
    /// </remarks>
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(context);
        cancellationToken.ThrowIfCancellationRequested();
        EvolutionTaskResult? validation = await _correctness.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        if (validation is null) throw new InvalidOperationException("Correctness evaluation returned no result or resource receipt.");
        if (validation.Status != EvolutionEvaluationStatus.Completed) return validation;
        if (validation.Direction != EvolutionOptimizationDirection.Maximize || validation.Quality != 1 ||
            validation.ConstraintViolations.Any(value => value > 0))
            return Copy(validation, EvolutionEvaluationStatus.Rejected, validation.CostUnits);
        cancellationToken.ThrowIfCancellationRequested();
        EvolutionTaskResult? fitness = await _fitness.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        if (fitness is null)
            throw new InvalidOperationException("Fitness evaluation returned no result or resource receipt.");
        EvolutionEvaluationStatus status = fitness.Status == EvolutionEvaluationStatus.Completed &&
            fitness.ConstraintViolations.Any(value => value > 0) ? EvolutionEvaluationStatus.Rejected : fitness.Status;
        return Copy(fitness, status, validation.CostUnits + fitness.CostUnits);
    }

    private static EvolutionTaskResult Copy(EvolutionTaskResult result, EvolutionEvaluationStatus status, double cost) => new(
        status, status == EvolutionEvaluationStatus.Completed ? result.Quality : null, result.Direction,
        result.Descriptors, result.Objectives, result.ConstraintViolations, cost, result.Diagnostics, result.Metrics, result.Artifacts);
}
