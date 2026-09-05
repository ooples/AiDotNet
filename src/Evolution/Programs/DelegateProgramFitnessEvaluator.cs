using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Adapts a caller-supplied function into an <see cref="IProgramFitnessEvaluator"/>.</summary>
/// <remarks>
/// <para>
/// Most first evaluators are a handful of lines: compile the candidate, run one benchmark, return a number. This
/// adapter removes the ceremony of declaring a class for that, while still requiring the two pieces of identity
/// the engine needs — a stable <see cref="Id"/> and a <see cref="VersionHash"/> that changes when the scoring rule
/// changes — so a checkpoint cannot be resumed against a silently different evaluator.
/// </para>
/// <para>
/// The delegate is invoked exactly once per evaluation attempt and its result is returned unchanged. Exceptions
/// escaping the delegate are not caught here; the engine converts them into a redacted diagnostic and applies its
/// failure policy, so a delegate that wants a retryable failure should return
/// <see cref="EvolutionTaskResult.Failed"/> instead of throwing.
/// </para>
/// <para><b>For Beginners:</b> This is the quickest way to plug your own scoring rule into program evolution: hand
/// it a small function that takes a candidate program and returns a score, and you have a working evaluator. Give
/// it a name and a version string, and remember to change the version whenever you change the way you score, so
/// old saved runs are not resumed with the new rules by accident.</para>
/// </remarks>
public sealed class DelegateProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private readonly Func<ProgramGenome, EvolutionEvaluationContext, CancellationToken, ValueTask<EvolutionTaskResult>> _evaluate;

    /// <summary>Initializes an evaluator backed by an asynchronous function.</summary>
    /// <param name="evaluate">The function that scores one candidate.</param>
    /// <param name="id">A stable evaluator identifier.</param>
    /// <param name="versionHash">A version hash that changes whenever the scoring rule changes.</param>
    /// <exception cref="ArgumentNullException"><paramref name="evaluate"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> or <paramref name="versionHash"/> is empty or white space.</exception>
    public DelegateProgramFitnessEvaluator(
        Func<ProgramGenome, EvolutionEvaluationContext, CancellationToken, ValueTask<EvolutionTaskResult>> evaluate,
        string id = "delegate-program-evaluator",
        string versionHash = "delegate-program-evaluator-v1")
    {
        Guard.NotNull(evaluate);
        Guard.NotNullOrWhiteSpace(id);
        Guard.NotNullOrWhiteSpace(versionHash);
        _evaluate = evaluate;
        Id = id.Trim();
        VersionHash = versionHash.Trim();
    }

    /// <summary>Initializes an evaluator backed by a synchronous scoring function.</summary>
    /// <param name="score">The function that returns a finite quality for one candidate.</param>
    /// <param name="id">A stable evaluator identifier.</param>
    /// <param name="versionHash">A version hash that changes whenever the scoring rule changes.</param>
    /// <param name="direction">Whether larger or smaller scores are better.</param>
    /// <exception cref="ArgumentNullException"><paramref name="score"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> or <paramref name="versionHash"/> is empty or white space.</exception>
    public DelegateProgramFitnessEvaluator(
        Func<ProgramGenome, double> score,
        string id = "delegate-program-evaluator",
        string versionHash = "delegate-program-evaluator-v1",
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize)
        : this(BuildAdapter(score, direction), id, versionHash)
    {
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <inheritdoc/>
    public ValueTask<EvolutionTaskResult> EvaluateAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(context);
        return _evaluate(candidate, context, cancellationToken);
    }

    private static Func<ProgramGenome, EvolutionEvaluationContext, CancellationToken, ValueTask<EvolutionTaskResult>> BuildAdapter(
        Func<ProgramGenome, double> score,
        EvolutionOptimizationDirection direction)
    {
        Guard.NotNull(score);
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction))
            throw new ArgumentOutOfRangeException(nameof(direction));

        return (genome, _, cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            double value = score(genome);
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Failed(
                    "program_score_not_finite", "The scoring function returned a value that is not finite."));
            }

            return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(
                value, new Dictionary<string, double>(StringComparer.Ordinal), direction));
        };
    }
}
