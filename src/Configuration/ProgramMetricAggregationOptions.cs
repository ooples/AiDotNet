namespace AiDotNet.Configuration;

/// <summary>Declares how a metrics dictionary is collapsed into the single quality an archive ranks on.</summary>
/// <remarks>
/// <para>
/// The defaults reproduce the reference OpenEvolve rule
/// (<c>openevolve/utils/metrics_utils.py</c> <c>get_fitness_score</c>): a metric literally named
/// <c>combined_score</c> wins outright, and otherwise the numeric metrics that are not archive feature dimensions
/// are averaged. <see cref="ExcludedFeatureDimensions"/> is the counterpart of that function's
/// <c>feature_dimensions</c> argument, which exists so a behaviour coordinate such as program length does not also
/// pull on fitness.
/// </para>
/// <para>
/// The remaining settings enable rules upstream has no counterpart for. <see cref="Weights"/> turns the implicit,
/// unvalidated mean into an explicit weighted mean whose weights are checked before a run starts, and
/// <see cref="ReferencePoint"/> with <see cref="AugmentationCoefficient"/> configures a Chebyshev achievement
/// scalarization that improves whichever objective is currently furthest from target.
/// <see cref="RequireAllWeightedMetrics"/> decides whether a weight naming a metric the evaluator did not report
/// is a configuration error or merely something to report.
/// </para>
/// <para><b>For Beginners:</b> Your evaluation code returns several numbers; this object says how they become one
/// score. Leave it alone and the behaviour matches the reference implementation. Set
/// <see cref="Strategy"/> to <c>Weighted</c> and fill in <see cref="Weights"/> when you know that, say, accuracy
/// matters four times as much as speed. Add the names of your MAP-Elites coordinates to
/// <see cref="ExcludedFeatureDimensions"/> so those coordinates describe the program without also scoring it.</para>
/// </remarks>
public sealed class ProgramMetricAggregationOptions
{
    /// <summary>The metric name the reference implementation prefers over every other, <c>combined_score</c>.</summary>
    public const string DefaultCombinedScoreKey = "combined_score";

    /// <summary>The largest number of issues one aggregation reports.</summary>
    public const int MaxReportedIssues = 256;

    /// <summary>Gets or sets the rule used to collapse the metrics. Defaults to the reference rule.</summary>
    public ProgramMetricAggregationStrategy Strategy { get; set; } = ProgramMetricAggregationStrategy.CombinedScoreOrMean;

    /// <summary>Gets or sets the metric name that short-circuits aggregation. Defaults to <c>combined_score</c>.</summary>
    /// <remarks>Used only by <see cref="ProgramMetricAggregationStrategy.CombinedScoreOrMean"/>.</remarks>
    public string CombinedScoreKey { get; set; } = DefaultCombinedScoreKey;

    /// <summary>Gets the metric names excluded from averaging because they are archive feature dimensions.</summary>
    /// <remarks>
    /// Matches the <c>feature_dimensions</c> argument of the reference implementation. When excluding every metric
    /// would leave nothing to average, the aggregation falls back to averaging all metrics, as upstream does.
    /// </remarks>
    public ICollection<string> ExcludedFeatureDimensions { get; } = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>Gets the per-metric weights used by <see cref="ProgramMetricAggregationStrategy.Weighted"/> and <see cref="ProgramMetricAggregationStrategy.Tchebycheff"/>.</summary>
    /// <remarks>Every weight must be finite and non-negative, and at least one must be positive.</remarks>
    public IDictionary<string, double> Weights { get; } = new Dictionary<string, double>(StringComparer.Ordinal);

    /// <summary>Gets the per-metric target values used by <see cref="ProgramMetricAggregationStrategy.Tchebycheff"/>.</summary>
    /// <remarks>Every weighted metric must have a finite reference value; the shortfall is measured against it.</remarks>
    public IDictionary<string, double> ReferencePoint { get; } = new Dictionary<string, double>(StringComparer.Ordinal);

    /// <summary>Gets or sets the augmentation coefficient of the Chebyshev scalarization. Defaults to 0.</summary>
    /// <remarks>
    /// Zero gives the classic weighted Chebyshev function. A small positive value, conventionally between 0.0001
    /// and 0.05, adds that multiple of the summed weighted shortfalls, which excludes weakly efficient solutions.
    /// </remarks>
    public double AugmentationCoefficient { get; set; }

    /// <summary>Gets or sets whether a declared weight naming an unreported metric is an error. Defaults to <c>true</c>.</summary>
    /// <remarks>
    /// When <c>false</c>, the missing metric is reported as an issue and the remaining weights are renormalized, so
    /// a partially reported evaluation still scores instead of throwing.
    /// </remarks>
    public bool RequireAllWeightedMetrics { get; set; } = true;

    /// <summary>Gets or sets whether text metrics that parse as finite numbers are accepted. Defaults to <c>true</c>.</summary>
    /// <remarks>
    /// The reference implementation reaches the same outcome for <c>combined_score</c> through Python's
    /// <c>float()</c> conversion. Setting this to <c>false</c> makes any text metric an issue instead.
    /// </remarks>
    public bool AllowTextMetricConversion { get; set; } = true;

    /// <summary>Creates an independent copy so a running aggregation is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same settings.</returns>
    public ProgramMetricAggregationOptions Clone()
    {
        var copy = new ProgramMetricAggregationOptions
        {
            Strategy = Strategy,
            CombinedScoreKey = CombinedScoreKey,
            AugmentationCoefficient = AugmentationCoefficient,
            RequireAllWeightedMetrics = RequireAllWeightedMetrics,
            AllowTextMetricConversion = AllowTextMetricConversion
        };
        foreach (string name in ExcludedFeatureDimensions) copy.ExcludedFeatureDimensions.Add(name);
        foreach (KeyValuePair<string, double> pair in Weights) copy.Weights[pair.Key] = pair.Value;
        foreach (KeyValuePair<string, double> pair in ReferencePoint) copy.ReferencePoint[pair.Key] = pair.Value;
        return copy;
    }

    /// <summary>Rejects a configuration that could not produce a defensible score.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="Strategy"/> is undefined, or a numeric setting is not finite or is negative.</exception>
    /// <exception cref="InvalidOperationException">
    /// A required setting is missing for the chosen strategy: an empty combined-score key, no positive weight, or a
    /// weighted metric with no reference value under the Chebyshev strategy.
    /// </exception>
    /// <exception cref="ArgumentException">A metric name is empty or white space.</exception>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ProgramMetricAggregationStrategy), Strategy))
            throw new ArgumentOutOfRangeException(nameof(Strategy));
        if (!IsFinite(AugmentationCoefficient) || AugmentationCoefficient < 0.0)
            throw new ArgumentOutOfRangeException(nameof(AugmentationCoefficient), AugmentationCoefficient,
                "The augmentation coefficient must be a finite, non-negative number.");

        foreach (string name in ExcludedFeatureDimensions)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("An excluded feature dimension name cannot be empty.", nameof(ExcludedFeatureDimensions));
        }

        foreach (KeyValuePair<string, double> pair in Weights)
        {
            if (string.IsNullOrWhiteSpace(pair.Key))
                throw new ArgumentException("A weighted metric name cannot be empty.", nameof(Weights));
            if (!IsFinite(pair.Value) || pair.Value < 0.0)
                throw new ArgumentOutOfRangeException(nameof(Weights), pair.Value,
                    $"The weight for '{pair.Key}' must be a finite, non-negative number.");
        }

        foreach (KeyValuePair<string, double> pair in ReferencePoint)
        {
            if (string.IsNullOrWhiteSpace(pair.Key))
                throw new ArgumentException("A reference-point metric name cannot be empty.", nameof(ReferencePoint));
            if (!IsFinite(pair.Value))
                throw new ArgumentOutOfRangeException(nameof(ReferencePoint), pair.Value,
                    $"The reference value for '{pair.Key}' must be a finite number.");
        }

        if (Strategy == ProgramMetricAggregationStrategy.CombinedScoreOrMean &&
            string.IsNullOrWhiteSpace(CombinedScoreKey))
        {
            throw new InvalidOperationException("The combined-score strategy requires a non-empty CombinedScoreKey.");
        }

        if (Strategy != ProgramMetricAggregationStrategy.Weighted &&
            Strategy != ProgramMetricAggregationStrategy.Tchebycheff)
        {
            return;
        }

        if (Weights.Count == 0)
            throw new InvalidOperationException($"The {Strategy} strategy requires at least one declared weight.");
        if (!Weights.Values.Any(weight => weight > 0.0))
            throw new InvalidOperationException($"The {Strategy} strategy requires at least one positive weight.");

        if (Strategy != ProgramMetricAggregationStrategy.Tchebycheff) return;
        foreach (KeyValuePair<string, double> pair in Weights)
        {
            if (pair.Value > 0.0 && !ReferencePoint.ContainsKey(pair.Key))
            {
                throw new InvalidOperationException(
                    $"The Tchebycheff strategy requires a reference value for the weighted metric '{pair.Key}'.");
            }
        }
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}
