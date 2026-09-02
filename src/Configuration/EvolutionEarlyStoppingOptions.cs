using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;

namespace AiDotNet.Configuration;

/// <summary>Stops an evolution run once its chosen progress metric has plateaued.</summary>
/// <remarks>
/// <para>
/// Early stopping is off by default: <see cref="PatienceEvaluations"/> is zero, so a run ends only on its budgets, its
/// time limit, a fail-fast candidate, a reached <c>TargetQuality</c>, or cancellation. When patience is positive the
/// engine evaluates <see cref="Metric"/> after every committed batch, normalised so that larger is always better even
/// under <see cref="EvolutionOptimizationDirection.Minimize"/>. A batch whose metric gains at least
/// <see cref="MinimumImprovement"/> resets the counter; otherwise that batch's committed evaluations are added to it,
/// and the run stops with <see cref="EvolutionStopReason.EarlyStopped"/> once the counter reaches
/// <see cref="PatienceEvaluations"/>. The counter and the running best are both checkpointed and folded into the run
/// state hash, so a resumed run continues counting from exactly where it stopped.
/// </para>
/// <para>
/// Two differences from OpenEvolve matter. It offers an "event" mode in which a negative patience makes the stop
/// condition an exact floating-point equality against the convergence threshold (process_parallel.py:792-801), which
/// almost never fires for a computed score; this engine has no equality mode and always uses a tolerance comparison.
/// And its controller writes the final checkpoint only when the final iteration index happens to be a multiple of the
/// checkpoint interval (controller.py:508-513), so an early stop between intervals silently discards the run's last
/// state; this engine always writes a final checkpoint before returning, whatever the stop reason.
/// </para>
/// <para><b>For Beginners:</b> A long search often finds its best answer early and then spends hours confirming it.
/// Early stopping ends that waiting. <see cref="PatienceEvaluations"/> is how many candidates you are willing to see go
/// by with no real progress before giving up, and <see cref="MinimumImprovement"/> defines "real": a gain smaller than
/// this does not count, which prevents floating-point noise from resetting the clock forever. Start with a patience of
/// a few hundred evaluations. If your run stops too eagerly, raise the patience rather than lowering the minimum
/// improvement.</para>
/// </remarks>
public sealed class EvolutionEarlyStoppingOptions
{
    /// <summary>Gets or sets how many committed evaluations without improvement end the run; zero disables early stopping.</summary>
    public long PatienceEvaluations { get; set; }

    /// <summary>Gets or sets the smallest metric gain that counts as progress.</summary>
    public double MinimumImprovement { get; set; } = 1e-3;

    /// <summary>Gets or sets which progress metric the plateau is measured on.</summary>
    /// <remarks>Ignored when <see cref="MetricName"/> names an evaluator metric instead.</remarks>
    public EvolutionEarlyStoppingMetric Metric { get; set; } = EvolutionEarlyStoppingMetric.BestQuality;

    /// <summary>Gets or sets an evaluator metric to watch by name, or <c>null</c> to use <see cref="Metric"/>.</summary>
    /// <remarks>
    /// <para>
    /// The three built-in choices describe the search itself. A run often plateaus on something only the evaluator
    /// can see, such as a validation score that stops improving while overall quality still drifts, and the
    /// reference implementation watches any metric key for exactly that reason. Naming one here watches the best
    /// value of that metric across every island, normalised so larger is better under either optimization
    /// direction, and a run whose evaluations never report it simply never stops early.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when the number you actually care about is one your own scoring code
    /// reports, rather than the archive's overall best.</para>
    /// </remarks>
    public string? MetricName { get; set; }

    /// <summary>Validates every value and returns an independent copy.</summary>
    /// <returns>A defensive copy that later mutation of this instance cannot affect.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <see cref="PatienceEvaluations"/> is negative, <see cref="MinimumImprovement"/> is not finite or is negative, or
    /// <see cref="Metric"/> is undefined.
    /// </exception>
    internal EvolutionEarlyStoppingOptions SnapshotAndValidate()
    {
        if (PatienceEvaluations < 0) throw new ArgumentOutOfRangeException(nameof(PatienceEvaluations));
        if (!EvolutionDescriptorDefinition.IsFinite(MinimumImprovement) || MinimumImprovement < 0)
            throw new ArgumentOutOfRangeException(nameof(MinimumImprovement),
                "The minimum improvement must be finite and non-negative.");
        if (!Enum.IsDefined(typeof(EvolutionEarlyStoppingMetric), Metric))
            throw new ArgumentOutOfRangeException(nameof(Metric));
        if (MetricName is not null && string.IsNullOrWhiteSpace(MetricName))
            throw new ArgumentException(
                "The watched metric name cannot be empty; leave it null to use the built-in metric.",
                nameof(MetricName));

        return new EvolutionEarlyStoppingOptions
        {
            PatienceEvaluations = PatienceEvaluations,
            MinimumImprovement = MinimumImprovement,
            Metric = Metric,
            MetricName = MetricName?.Trim()
        };
    }

    /// <summary>Returns a stable, culture-independent representation suitable for compatibility hashes.</summary>
    /// <returns>The canonical text form of every value that changes early-stopping behaviour.</returns>
    internal string ToCanonicalString() => string.Join("|", new[]
    {
        PatienceEvaluations.ToString(CultureInfo.InvariantCulture),
        MinimumImprovement.ToString("R", CultureInfo.InvariantCulture),
        ((int)Metric).ToString(CultureInfo.InvariantCulture),
        MetricName ?? "built-in"
    });
}
