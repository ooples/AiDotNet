using System.Collections.ObjectModel;

namespace AiDotNet.Evolution;

/// <summary>Immutable evaluation cost, stage, and elapsed-time metadata.</summary>
/// <remarks>
/// <para>
/// Every <see cref="EvolutionEvaluation"/> carries one of these records so that budgets, reports, and checkpoints can
/// account for what an evaluation consumed. <see cref="Elapsed"/> is wall-clock evaluator time measured by the engine
/// across all attempts, <see cref="AttemptCount"/> is the number of attempts the canonical candidate needed (greater
/// than one only when the failure policy retried it, and zero when the result was served from the evaluation cache),
/// and <see cref="CostUnits"/> is whatever nonnegative, task-defined quantity the
/// <see cref="AiDotNet.Interfaces.IEvolutionTask{TGenome}"/> reported through
/// <see cref="EvolutionTaskResult.CostUnits"/>, such as training epochs, tokens, or GPU-seconds. A cache hit reports
/// zero cost units because no evaluator work was performed.
/// </para>
/// <para>
/// Elapsed time is excluded from the deterministic run state hash because it varies between machines, whereas attempt
/// counts and cost units are reproducible for a given seed and are included. The constructor rejects negative
/// durations, negative attempt counts, and non-finite or negative cost units.
/// </para>
/// <para>
/// When staged (cascade) evaluation is enabled, <see cref="StageCostUnits"/> breaks the total down by stage for the
/// most recent attempt, and <see cref="RejectedStage"/> names the stage whose threshold the candidate failed to clear,
/// or is <c>null</c> when no stage rejected it. That is the accounting OpenEvolve cannot report: a cascade rejection
/// there collapses to an ordinary metrics dictionary whose fallback score is <c>0.0</c>, with no record of which stage
/// stopped it or what the earlier stages cost (evaluator.py:422-430,668-707).
/// </para>
/// <para><b>For Beginners:</b> This is the receipt attached to each evaluated candidate: how long it took, how many
/// tries were needed, and how much of your own budget unit it used. Suppose your task trains a small model per
/// candidate and reports the number of epochs as the cost; after a run you can sum <see cref="CostUnits"/> across the
/// archive to see the total training spent, or compare <see cref="Elapsed"/> between candidates to find configurations
/// that are unexpectedly slow. You do not usually create this class yourself; the engine builds it from the task's
/// result and its own timer.</para>
/// </remarks>
public sealed class EvolutionEvaluationCost
{
    /// <summary>Initializes cost metadata.</summary>
    /// <param name="elapsed">Nonnegative wall-clock evaluator time.</param>
    /// <param name="attemptCount">The nonnegative number of evaluation attempts for the canonical candidate.</param>
    /// <param name="costUnits">Finite, nonnegative task-defined resource units.</param>
    /// <param name="stageCostUnits">
    /// Optional per-stage resource units for the most recent cascade attempt; every value must be finite and
    /// nonnegative.
    /// </param>
    /// <param name="rejectedStage">
    /// The zero-based cascade stage whose threshold the candidate failed to clear, or <c>null</c> when no stage
    /// rejected it.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="elapsed"/> or <paramref name="attemptCount"/> is negative, <paramref name="costUnits"/> or a
    /// stage cost is negative or not finite, or <paramref name="rejectedStage"/> is negative.
    /// </exception>
    public EvolutionEvaluationCost(TimeSpan elapsed, int attemptCount, double costUnits,
        IReadOnlyList<double>? stageCostUnits = null, int? rejectedStage = null)
    {
        if (elapsed < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(elapsed));
        if (attemptCount < 0) throw new ArgumentOutOfRangeException(nameof(attemptCount));
        if (!EvolutionDescriptorDefinition.IsFinite(costUnits) || costUnits < 0) throw new ArgumentOutOfRangeException(nameof(costUnits));
        if (rejectedStage.HasValue && rejectedStage.Value < 0) throw new ArgumentOutOfRangeException(nameof(rejectedStage));
        double[] stageCosts = stageCostUnits?.ToArray() ?? Array.Empty<double>();
        foreach (double stageCost in stageCosts)
            if (!EvolutionDescriptorDefinition.IsFinite(stageCost) || stageCost < 0)
                throw new ArgumentOutOfRangeException(nameof(stageCostUnits));
        Elapsed = elapsed;
        AttemptCount = attemptCount;
        CostUnits = costUnits;
        StageCostUnits = new ReadOnlyCollection<double>(stageCosts);
        RejectedStage = rejectedStage;
    }

    /// <summary>Gets wall-clock evaluator time.</summary>
    public TimeSpan Elapsed { get; }

    /// <summary>Gets the one-based attempt count for this canonical candidate.</summary>
    public int AttemptCount { get; }

    /// <summary>Gets task-defined resource units.</summary>
    public double CostUnits { get; }

    /// <summary>Gets per-stage resource units for the most recent cascade attempt; empty without staged evaluation.</summary>
    public IReadOnlyList<double> StageCostUnits { get; }

    /// <summary>Gets the cascade stage that rejected the candidate, or <c>null</c> when none did.</summary>
    public int? RejectedStage { get; }
}
