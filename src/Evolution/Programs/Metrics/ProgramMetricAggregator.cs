using AiDotNet.Configuration;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Metrics;

/// <summary>Turns a dictionary of evaluator metrics into the single scalar quality an archive ranks on.</summary>
/// <remarks>
/// <para>
/// The default strategy reproduces the reference OpenEvolve rule
/// (<c>openevolve/utils/metrics_utils.py</c> <c>get_fitness_score</c>, lines 72-118) exactly: an empty dictionary
/// scores zero; a metric literally named <c>combined_score</c> is converted and returned outright; otherwise the
/// numeric metrics that are not archive feature dimensions are averaged; and when excluding the feature dimensions
/// leaves nothing, every numeric metric is averaged instead. Booleans never count as scores - Python treats
/// <c>True</c> as <c>1</c>, so averaging a timeout flag in would give a crashed program a mid-range fitness - and
/// <c>NaN</c> values are dropped, while infinities are not.
/// </para>
/// <para>
/// Two things are deliberately different. First, every value the rule discards is returned as a
/// <see cref="ProgramMetricIssue"/> instead of vanishing, so an evaluator that reports its accuracy as the string
/// <c>"0.9"</c> is distinguishable from one whose program genuinely scored zero. Second, text is converted only
/// when it parses to a finite invariant-culture number, where Python's <c>float()</c> would also accept the
/// literals <c>nan</c> and <c>inf</c>; that keeps the outcome identical on every target framework. Beyond the
/// reference rule, <see cref="ProgramMetricAggregationStrategy.Weighted"/> and
/// <see cref="ProgramMetricAggregationStrategy.Tchebycheff"/> offer aggregations whose parameters are validated
/// before a run begins rather than assumed.
/// </para>
/// <para><b>For Beginners:</b> Your evaluation code measures several things about a candidate program, but the
/// search can only rank on one number. This class does that conversion and, just as importantly, tells you what it
/// did. Construct one with default options and call <see cref="Aggregate(IReadOnlyDictionary{string, ProgramMetricValue})"/>;
/// read <see cref="ProgramMetricAggregationResult.Value"/> for the score, check
/// <see cref="ProgramMetricAggregationResult.HasFiniteValue"/> before reporting it as a completed evaluation, and
/// read <see cref="ProgramMetricAggregationResult.Issues"/> when the score is not what you expected.</para>
/// </remarks>
public sealed class ProgramMetricAggregator
{
    /// <summary>The metric name reported on an issue that concerns the aggregation itself rather than one metric.</summary>
    public const string AggregateMetricName = "(aggregate)";

    private readonly ProgramMetricAggregationOptions _options;

    /// <summary>Initializes an aggregator.</summary>
    /// <param name="options">The aggregation settings; <c>null</c> uses the reference rule with its defaults.</param>
    /// <exception cref="ArgumentOutOfRangeException">A numeric setting is not finite or is negative.</exception>
    /// <exception cref="InvalidOperationException">A required setting is missing for the chosen strategy.</exception>
    /// <exception cref="ArgumentException">A metric name in the options is empty or white space.</exception>
    public ProgramMetricAggregator(ProgramMetricAggregationOptions? options = null)
    {
        ProgramMetricAggregationOptions effective = options is null
            ? new ProgramMetricAggregationOptions()
            : options.Clone();
        effective.Validate();
        _options = effective;
    }

    /// <summary>Gets an independent copy of the settings this aggregator was validated with.</summary>
    /// <returns>A copy that a caller may mutate without affecting this instance.</returns>
    public ProgramMetricAggregationOptions GetOptions() => _options.Clone();

    /// <summary>Gets whether larger or smaller aggregated values are better under the configured strategy.</summary>
    /// <remarks>
    /// Every strategy maximizes except <see cref="ProgramMetricAggregationStrategy.Tchebycheff"/>, which measures a
    /// shortfall from a target and is therefore minimized. Report this value as the direction of the evaluation so
    /// the archive compares candidates the right way round.
    /// </remarks>
    public EvolutionOptimizationDirection PreferredDirection =>
        _options.Strategy == ProgramMetricAggregationStrategy.Tchebycheff
            ? EvolutionOptimizationDirection.Minimize
            : EvolutionOptimizationDirection.Maximize;

    /// <summary>Collapses a mixed-type metrics dictionary into one scalar quality.</summary>
    /// <param name="metrics">The reported metrics; an empty dictionary aggregates to zero.</param>
    /// <returns>The scalar quality together with the metrics used and the metrics set aside.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="metrics"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="metrics"/> contains a null value or an empty name.</exception>
    /// <exception cref="InvalidOperationException">
    /// A weighted strategy is configured with <see cref="ProgramMetricAggregationOptions.RequireAllWeightedMetrics"/>
    /// and a declared weight names a metric that was not reported as a usable number.
    /// </exception>
    public ProgramMetricAggregationResult Aggregate(IReadOnlyDictionary<string, ProgramMetricValue> metrics)
    {
        Guard.NotNull(metrics);
        var ordered = new SortedDictionary<string, ProgramMetricValue>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, ProgramMetricValue> pair in metrics)
        {
            if (string.IsNullOrWhiteSpace(pair.Key))
                throw new ArgumentException("A metric name cannot be empty or white space.", nameof(metrics));
            if (pair.Value is null)
                throw new ArgumentException($"The metric '{pair.Key}' has a null value.", nameof(metrics));
            ordered[pair.Key.Trim()] = pair.Value;
        }

        switch (_options.Strategy)
        {
            case ProgramMetricAggregationStrategy.CombinedScoreOrMean:
                return AggregateCombinedScoreOrMean(ordered);
            case ProgramMetricAggregationStrategy.Mean:
                return AggregateMean(ordered, ProgramMetricAggregationStrategy.Mean, new List<ProgramMetricIssue>());
            case ProgramMetricAggregationStrategy.Weighted:
                return AggregateWeighted(ordered);
            default:
                return AggregateTchebycheff(ordered);
        }
    }

    /// <summary>Collapses an all-numeric metrics dictionary into one scalar quality.</summary>
    /// <param name="metrics">The reported numeric metrics.</param>
    /// <returns>The scalar quality together with the metrics used and the metrics set aside.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="metrics"/> is <c>null</c>.</exception>
    public ProgramMetricAggregationResult Aggregate(IReadOnlyDictionary<string, double> metrics)
    {
        Guard.NotNull(metrics);
        var wrapped = new Dictionary<string, ProgramMetricValue>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, double> pair in metrics) wrapped[pair.Key] = ProgramMetricValue.Number(pair.Value);
        return Aggregate(wrapped);
    }

    /// <summary>Applies the reference OpenEvolve fitness rule with no other configuration.</summary>
    /// <param name="metrics">The reported metrics.</param>
    /// <param name="featureDimensions">Archive feature dimensions to exclude from the average, or <c>null</c> for none.</param>
    /// <returns>The scalar quality the reference implementation would have produced, plus the discards it hid.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="metrics"/> is <c>null</c>.</exception>
    public static ProgramMetricAggregationResult UpstreamFitnessScore(
        IReadOnlyDictionary<string, ProgramMetricValue> metrics,
        IEnumerable<string>? featureDimensions = null)
    {
        var options = new ProgramMetricAggregationOptions();
        if (featureDimensions is not null)
        {
            foreach (string dimension in featureDimensions)
            {
                if (!string.IsNullOrWhiteSpace(dimension)) options.ExcludedFeatureDimensions.Add(dimension.Trim());
            }
        }

        return new ProgramMetricAggregator(options).Aggregate(metrics);
    }

    private ProgramMetricAggregationResult AggregateCombinedScoreOrMean(SortedDictionary<string, ProgramMetricValue> metrics)
    {
        const ProgramMetricAggregationStrategy Strategy = ProgramMetricAggregationStrategy.CombinedScoreOrMean;
        var issues = new List<ProgramMetricIssue>();
        string key = _options.CombinedScoreKey.Trim();
        if (metrics.TryGetValue(key, out ProgramMetricValue? preferred) && preferred is not null)
        {
            switch (preferred.Kind)
            {
                case ProgramMetricValueKind.Number:
                {
                    preferred.TryGetNumber(allowTextConversion: false, out double number);
                    AddValueIssue(issues, key, number);
                    return Complete(number, Strategy, usedCombinedScore: true, new[] { key }, issues);
                }

                case ProgramMetricValueKind.Flag:
                {
                    Add(issues, key, ProgramMetricIssueReason.BooleanFlag,
                        "The combined-score metric is a flag; a true flag scores one and a false flag scores zero.");
                    return Complete(preferred.FlagValue ? 1.0 : 0.0, Strategy, usedCombinedScore: true, new[] { key }, issues);
                }

                default:
                {
                    if (_options.AllowTextMetricConversion &&
                        preferred.TryGetNumber(allowTextConversion: true, out double converted))
                    {
                        return Complete(converted, Strategy, usedCombinedScore: true, new[] { key }, issues);
                    }

                    Add(issues, key, ProgramMetricIssueReason.NonNumericText,
                        "The combined-score metric is text that is not a finite number; the mean was used instead.");
                    break;
                }
            }
        }

        return AggregateMean(metrics, Strategy, issues);
    }

    private ProgramMetricAggregationResult AggregateMean(
        SortedDictionary<string, ProgramMetricValue> metrics,
        ProgramMetricAggregationStrategy strategy,
        List<ProgramMetricIssue> issues)
    {
        var numeric = new List<KeyValuePair<string, double>>();
        var excludedNumeric = new List<KeyValuePair<string, double>>();
        foreach (KeyValuePair<string, ProgramMetricValue> pair in metrics)
        {
            if (pair.Value.Kind == ProgramMetricValueKind.Flag)
            {
                Add(issues, pair.Key, ProgramMetricIssueReason.BooleanFlag,
                    "A flag is not a score and was left out of the average.");
                continue;
            }

            if (pair.Value.Kind == ProgramMetricValueKind.Text)
            {
                Add(issues, pair.Key, ProgramMetricIssueReason.NonNumericText,
                    "Text metrics never contribute to an average, even when they parse as numbers.");
                continue;
            }

            pair.Value.TryGetNumber(allowTextConversion: false, out double number);
            if (double.IsNaN(number))
            {
                Add(issues, pair.Key, ProgramMetricIssueReason.NotANumber,
                    "The value is not a number and was left out of the average.");
                continue;
            }

            if (_options.ExcludedFeatureDimensions.Contains(pair.Key))
            {
                excludedNumeric.Add(new KeyValuePair<string, double>(pair.Key, number));
                continue;
            }

            numeric.Add(new KeyValuePair<string, double>(pair.Key, number));
        }

        List<KeyValuePair<string, double>> contributing;
        if (numeric.Count > 0)
        {
            contributing = numeric;
            foreach (KeyValuePair<string, double> pair in excludedNumeric)
            {
                Add(issues, pair.Key, ProgramMetricIssueReason.ExcludedFeatureDimension,
                    "The metric is an archive feature dimension and was excluded from the average.");
            }
        }
        else if (excludedNumeric.Count > 0)
        {
            contributing = excludedNumeric;
        }
        else
        {
            Add(issues, AggregateMetricName, ProgramMetricIssueReason.NoNumericValues,
                "No usable numeric metric was reported, so the aggregation returned zero.");
            return Complete(0.0, strategy, usedCombinedScore: false, Array.Empty<string>(), issues);
        }

        double total = 0.0;
        foreach (KeyValuePair<string, double> pair in contributing)
        {
            AddValueIssue(issues, pair.Key, pair.Value);
            total += pair.Value;
        }

        return Complete(total / contributing.Count, strategy, usedCombinedScore: false,
            contributing.Select(pair => pair.Key), issues);
    }

    private ProgramMetricAggregationResult AggregateWeighted(SortedDictionary<string, ProgramMetricValue> metrics)
    {
        const ProgramMetricAggregationStrategy Strategy = ProgramMetricAggregationStrategy.Weighted;
        var issues = new List<ProgramMetricIssue>();
        var contributing = new List<string>();
        double weightTotal = 0.0;
        double weightedTotal = 0.0;

        foreach (KeyValuePair<string, double> weight in OrderedWeights())
        {
            if (!TryReadWeighted(metrics, weight.Key, issues, out double value)) continue;
            AddValueIssue(issues, weight.Key, value);
            weightTotal += weight.Value;
            weightedTotal += weight.Value * value;
            contributing.Add(weight.Key);
        }

        ReportUnweighted(metrics, issues);
        if (weightTotal <= 0.0)
        {
            Add(issues, AggregateMetricName, ProgramMetricIssueReason.NoNumericValues,
                "No weighted metric was reported as a usable number, so the aggregation returned zero.");
            return Complete(0.0, Strategy, usedCombinedScore: false, Array.Empty<string>(), issues);
        }

        return Complete(weightedTotal / weightTotal, Strategy, usedCombinedScore: false, contributing, issues);
    }

    private ProgramMetricAggregationResult AggregateTchebycheff(SortedDictionary<string, ProgramMetricValue> metrics)
    {
        const ProgramMetricAggregationStrategy Strategy = ProgramMetricAggregationStrategy.Tchebycheff;
        var issues = new List<ProgramMetricIssue>();
        var contributing = new List<string>();
        double worst = 0.0;
        double sum = 0.0;
        bool any = false;

        foreach (KeyValuePair<string, double> weight in OrderedWeights())
        {
            if (!TryReadWeighted(metrics, weight.Key, issues, out double value)) continue;
            AddValueIssue(issues, weight.Key, value);
            double reference = _options.ReferencePoint[weight.Key];
            double shortfall = weight.Value * Math.Abs(reference - value);
            if (!any || shortfall > worst) worst = shortfall;
            sum += shortfall;
            any = true;
            contributing.Add(weight.Key);
        }

        ReportUnweighted(metrics, issues);
        if (!any)
        {
            Add(issues, AggregateMetricName, ProgramMetricIssueReason.NoNumericValues,
                "No weighted metric was reported as a usable number, so the aggregation returned zero.");
            return Complete(0.0, Strategy, usedCombinedScore: false, Array.Empty<string>(), issues);
        }

        return Complete(worst + (_options.AugmentationCoefficient * sum), Strategy, usedCombinedScore: false,
            contributing, issues);
    }

    private IEnumerable<KeyValuePair<string, double>> OrderedWeights() =>
        _options.Weights.Where(pair => pair.Value > 0.0).OrderBy(pair => pair.Key, StringComparer.Ordinal);

    private bool TryReadWeighted(
        SortedDictionary<string, ProgramMetricValue> metrics,
        string name,
        List<ProgramMetricIssue> issues,
        out double value)
    {
        value = 0.0;
        if (!metrics.TryGetValue(name, out ProgramMetricValue? metric) || metric is null)
        {
            Reject(issues, name, ProgramMetricIssueReason.MissingMetric,
                "The configuration declares a weight for a metric the evaluation did not report.");
            return false;
        }

        if (metric.Kind == ProgramMetricValueKind.Flag)
        {
            Reject(issues, name, ProgramMetricIssueReason.BooleanFlag,
                "A flag cannot be weighted because a flag is not a score.");
            return false;
        }

        if (!metric.TryGetNumber(_options.AllowTextMetricConversion, out double number))
        {
            Reject(issues, name, ProgramMetricIssueReason.NonNumericText,
                "The metric is text that is not a finite number, so it cannot be weighted.");
            return false;
        }

        if (double.IsNaN(number))
        {
            Reject(issues, name, ProgramMetricIssueReason.NotANumber,
                "The metric is not a number, so it cannot be weighted.");
            return false;
        }

        value = number;
        return true;
    }

    private void ReportUnweighted(SortedDictionary<string, ProgramMetricValue> metrics, List<ProgramMetricIssue> issues)
    {
        foreach (KeyValuePair<string, ProgramMetricValue> pair in metrics)
        {
            if (pair.Value.Kind != ProgramMetricValueKind.Number) continue;
            if (_options.ExcludedFeatureDimensions.Contains(pair.Key)) continue;
            if (_options.Weights.TryGetValue(pair.Key, out double weight) && weight > 0.0) continue;
            Add(issues, pair.Key, ProgramMetricIssueReason.NoWeightDeclared,
                "The evaluation reported this numeric metric but the configuration declares no weight for it.");
        }
    }

    private void Reject(List<ProgramMetricIssue> issues, string name, ProgramMetricIssueReason reason, string description)
    {
        if (_options.RequireAllWeightedMetrics)
        {
            throw new InvalidOperationException(
                $"The {_options.Strategy} strategy requires the metric '{name}': {description}");
        }

        Add(issues, name, reason, description);
    }

    private static void AddValueIssue(List<ProgramMetricIssue> issues, string name, double value)
    {
        if (double.IsNaN(value))
        {
            Add(issues, name, ProgramMetricIssueReason.NotANumber,
                "The metric is not a number, so the aggregated quality cannot be compared.");
        }
        else if (double.IsInfinity(value))
        {
            Add(issues, name, ProgramMetricIssueReason.NotFinite,
                "The metric is infinite, so the aggregated quality cannot be compared.");
        }
    }

    private static void Add(List<ProgramMetricIssue> issues, string name, ProgramMetricIssueReason reason, string description)
    {
        if (issues.Count >= ProgramMetricAggregationOptions.MaxReportedIssues) return;
        issues.Add(new ProgramMetricIssue(name, reason, description));
    }

    private static ProgramMetricAggregationResult Complete(
        double value,
        ProgramMetricAggregationStrategy strategy,
        bool usedCombinedScore,
        IEnumerable<string> contributing,
        List<ProgramMetricIssue> issues) =>
        new(value, strategy, usedCombinedScore, contributing, issues);
}
