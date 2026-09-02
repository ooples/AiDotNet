using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>Configures staged (cascade) evaluation for tasks that implement <c>ICascadeEvolutionTask&lt;TGenome&gt;</c>.</summary>
/// <remarks>
/// <para>
/// Cascade evaluation is off by default, so an engine behaves exactly as before until <see cref="Enabled"/> is set.
/// When it is set, the task must implement <c>ICascadeEvolutionTask&lt;TGenome&gt;</c> and the configuration is
/// validated against that task's stage count at construction time: <see cref="Thresholds"/> must contain exactly one
/// fewer entry than there are stages, every threshold must be finite, the thresholds must be monotonically
/// non-decreasing when the archive maximizes and non-increasing when it minimizes, and <see cref="StageTimeouts"/> must
/// either be empty or carry one positive duration per stage. Every one of those checks throws at configure time.
/// OpenEvolve validates none of them: it warns and silently degrades to direct evaluation when the stage functions are
/// missing (evaluator.py:101-130), ignores the third of its three default thresholds entirely, and applies one global
/// timeout to each stage so a three-stage cascade can take three times the configured limit (evaluator.py:398-501).
/// </para>
/// <para>
/// <see cref="ChargeRejectedStagesToBudget"/> controls the accounting that matters most for an expensive evaluator.
/// Left at its default of <c>false</c>, a candidate that a pre-final stage rejects costs zero of
/// <c>MaxEvaluationAttempts</c>, so the budget measures full evaluations rather than screening calls; the work is still
/// visible through <see cref="EvolutionEvaluationCost.AttemptCount"/> and
/// <see cref="EvolutionEvaluationCost.StageCostUnits"/>. Set it to <c>true</c> when every stage is expensive enough
/// that screening should be capped too.
/// </para>
/// <para><b>For Beginners:</b> These settings control a "cheap test first" evaluation pipeline. <see cref="Thresholds"/>
/// is the score a candidate must reach at each cheap stage to earn the next, more expensive one - with three stages you
/// supply two thresholds. <see cref="StageTimeouts"/> lets you give the quick smoke test a few seconds while allowing
/// the full benchmark an hour. And because a candidate thrown out by the smoke test never ran the expensive benchmark,
/// by default it does not count against the number of evaluations you budgeted, which is usually what you meant when
/// you said "run 100 evaluations".</para>
/// </remarks>
public sealed class EvolutionCascadeOptions
{
    private static readonly double[] NoThresholds = Array.Empty<double>();
    private static readonly TimeSpan[] NoStageTimeouts = Array.Empty<TimeSpan>();

    /// <summary>Gets or sets whether staged evaluation runs; <c>false</c> keeps the single-call evaluator path.</summary>
    public bool Enabled { get; set; }

    /// <summary>Gets or sets the per-stage gates, one fewer than the task's stage count.</summary>
    public IReadOnlyList<double> Thresholds { get; set; } = NoThresholds;

    /// <summary>Gets or sets one cooperative timeout per stage; empty applies <c>EvaluationTimeout</c> to each stage.</summary>
    public IReadOnlyList<TimeSpan> StageTimeouts { get; set; } = NoStageTimeouts;

    /// <summary>Gets or sets whether a stage rejection consumes one of the run's evaluation attempts.</summary>
    public bool ChargeRejectedStagesToBudget { get; set; }

    /// <summary>Validates every value that does not depend on the task and returns an independent copy.</summary>
    /// <returns>A defensive copy that later mutation of this instance cannot affect.</returns>
    /// <exception cref="ArgumentNullException">A collection property was set to <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A threshold is not finite, or a stage timeout is out of range.</exception>
    internal EvolutionCascadeOptions SnapshotAndValidate()
    {
        Guard.NotNull(Thresholds);
        Guard.NotNull(StageTimeouts);
        double[] thresholds = Thresholds.ToArray();
        foreach (double threshold in thresholds)
            if (!EvolutionDescriptorDefinition.IsFinite(threshold))
                throw new ArgumentOutOfRangeException(nameof(Thresholds), "Cascade thresholds must be finite.");
        TimeSpan[] timeouts = StageTimeouts.ToArray();
        foreach (TimeSpan timeout in timeouts)
            if (timeout <= TimeSpan.Zero || timeout.TotalMilliseconds > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(StageTimeouts),
                    "Cascade stage timeouts must be positive and within the cross-target cancellation timer range.");

        return new EvolutionCascadeOptions
        {
            Enabled = Enabled,
            Thresholds = thresholds,
            StageTimeouts = timeouts,
            ChargeRejectedStagesToBudget = ChargeRejectedStagesToBudget
        };
    }

    /// <summary>Validates the stage-dependent invariants once the task's stage count and direction are known.</summary>
    /// <param name="stageCount">The task's stage count.</param>
    /// <param name="direction">The archive optimization direction the thresholds must be monotone in.</param>
    /// <exception cref="ArgumentException">
    /// <paramref name="stageCount"/> is not positive, the threshold count does not equal <paramref name="stageCount"/>
    /// minus one, the thresholds are not monotone in <paramref name="direction"/>, or the stage-timeout count is
    /// neither zero nor <paramref name="stageCount"/>.
    /// </exception>
    internal void ValidateAgainstStages(int stageCount, EvolutionOptimizationDirection direction)
    {
        if (stageCount < 1)
            throw new ArgumentException("A cascade task must declare at least one stage.", nameof(stageCount));
        if (Thresholds.Count != stageCount - 1)
            throw new ArgumentException(
                $"Cascade evaluation with {stageCount} stages requires exactly {stageCount - 1} thresholds.",
                nameof(stageCount));
        for (int i = 1; i < Thresholds.Count; i++)
        {
            bool monotone = direction == EvolutionOptimizationDirection.Maximize
                ? Thresholds[i] >= Thresholds[i - 1]
                : Thresholds[i] <= Thresholds[i - 1];
            if (!monotone)
                throw new ArgumentException(
                    "Cascade thresholds must become stricter with each stage in the archive's optimization direction.",
                    nameof(stageCount));
        }
        if (StageTimeouts.Count != 0 && StageTimeouts.Count != stageCount)
            throw new ArgumentException(
                "Cascade stage timeouts must be empty or supply exactly one duration per stage.", nameof(stageCount));
    }

    /// <summary>Returns a stable, culture-independent representation suitable for compatibility hashes.</summary>
    /// <returns>The canonical text form of every value that changes cascade behaviour.</returns>
    internal string ToCanonicalString() => string.Join("|", new[]
    {
        Enabled ? "cascade" : "no-cascade",
        ChargeRejectedStagesToBudget ? "charge-rejected" : "refund-rejected",
        string.Join(",", Thresholds.Select(value => value.ToString("R", CultureInfo.InvariantCulture))),
        string.Join(",", StageTimeouts.Select(value => value.Ticks.ToString(CultureInfo.InvariantCulture)))
    });
}
