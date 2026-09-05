using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Adds ordered, increasingly expensive evaluation stages to an evolution task.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// This interface extends <see cref="IEvolutionTask{TGenome}"/> rather than replacing it, so every existing task
/// continues to compile and run unchanged and a task opts into cascade evaluation simply by implementing the two extra
/// members. The engine uses the cascade path only when the task implements this interface <b>and</b>
/// <c>EvolutionEngineOptions.Cascade.Enabled</c> is set; otherwise <see cref="IEvolutionTask{TGenome}.EvaluateAsync"/>
/// is called exactly as before. That is a deliberate contrast with OpenEvolve, which decides whether a cascade exists
/// by searching the evaluator's <i>source text</i> for the substring <c>evaluate_stage1</c> and then silently mutating
/// the caller's configuration object (api.py:148-153), and which hard-codes exactly three stage functions
/// (evaluator.py:360-548).
/// </para>
/// <para>
/// Stages run in index order from <c>0</c> to <see cref="StageCount"/> minus one. After each non-final stage the engine
/// compares the stage's <see cref="EvolutionTaskResult.Quality"/> against the matching entry of
/// <c>EvolutionCascadeOptions.Thresholds</c> in the result's optimization direction; a stage that does not clear its
/// threshold ends the evaluation with <see cref="AiDotNet.Enums.EvolutionEvaluationStatus.Skipped"/> and a diagnostic
/// naming the stage, and - unless the options say otherwise - does not consume the run's expensive-evaluation budget.
/// Descriptors, reporting metrics, artifacts, and diagnostics from every executed stage are merged, with later stages
/// overriding earlier ones on a name collision, and each stage's cost units are recorded separately in
/// <see cref="EvolutionEvaluationCost.StageCostUnits"/>.
/// </para>
/// <para><b>For Beginners:</b> Some candidates are obviously bad and you should find that out cheaply. A cascade is a
/// series of increasingly thorough tests: stage 0 might be a five-second smoke test, stage 1 a one-minute unit-test
/// run, and stage 2 the full benchmark that takes an hour. A candidate only reaches the expensive stage if it cleared
/// the cheap ones, so most of your budget is spent on candidates that deserve it. Implement <see cref="StageCount"/> to
/// say how many tests you have, and <see cref="EvaluateStageAsync"/> to run test number <c>stage</c>. Return a
/// completed result with a quality from each stage; the engine handles the thresholds, the merging, and the
/// bookkeeping.</para>
/// </remarks>
public interface ICascadeEvolutionTask<TGenome> : IEvolutionTask<TGenome>
{
    /// <summary>Gets the number of ordered evaluation stages; must be at least one and must not change during a run.</summary>
    int StageCount { get; }

    /// <summary>Evaluates one cascade stage of a canonical candidate.</summary>
    /// <param name="stage">The zero-based stage index, always less than <see cref="StageCount"/>.</param>
    /// <param name="candidate">The candidate to evaluate.</param>
    /// <param name="context">Deterministic per-evaluation context, including the seed stream and attempt count.</param>
    /// <param name="cancellationToken">A token that cancels the operation.</param>
    /// <returns>
    /// The stage's terminal result. A completed result must carry a finite quality so the engine can apply the stage
    /// threshold; any other status ends the cascade at this stage.
    /// </returns>
    ValueTask<EvolutionTaskResult> EvaluateStageAsync(
        int stage,
        EvolutionCandidate<TGenome> candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default);
}
