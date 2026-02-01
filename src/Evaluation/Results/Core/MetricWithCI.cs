using AiDotNet.Evaluation.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Results.Core;

/// <summary>
/// Represents a metric value with optional confidence interval and metadata.
/// </summary>
/// <remarks>
/// <para>
/// This is the fundamental building block for all evaluation metrics. It contains the point
/// estimate, confidence interval, and metadata about how the metric was computed.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you measure something, you get a single number (the point estimate).
/// But that number has uncertainty. The confidence interval tells you the range where the true
/// value likely falls. For example: "Accuracy = 0.85 [0.82, 0.88]" means the accuracy is about
/// 85%, but could reasonably be anywhere from 82% to 88%.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for the metric value.</typeparam>
public class MetricWithCI<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The point estimate (main value) of the metric.
    /// </summary>
    public T Value { get; init; }

    /// <summary>
    /// Lower bound of the confidence interval. Null if CI was not computed.
    /// </summary>
    public T? LowerBound { get; init; }

    /// <summary>
    /// Upper bound of the confidence interval. Null if CI was not computed.
    /// </summary>
    public T? UpperBound { get; init; }

    /// <summary>
    /// Confidence level (e.g., 0.95 for 95% CI). Null if CI was not computed.
    /// </summary>
    public double? ConfidenceLevel { get; init; }

    /// <summary>
    /// Method used to compute the confidence interval.
    /// </summary>
    public ConfidenceIntervalMethod? CIMethod { get; init; }

    /// <summary>
    /// Standard error of the metric. Null if not computed.
    /// </summary>
    public T? StandardError { get; init; }

    /// <summary>
    /// Standard deviation of the metric (e.g., across CV folds). Null if not computed.
    /// </summary>
    public T? StandardDeviation { get; set; }

    /// <summary>
    /// Number of samples used to compute the metric.
    /// </summary>
    public int? SampleCount { get; init; }

    /// <summary>
    /// Whether higher values indicate better performance for this metric.
    /// </summary>
    public MetricDirection Direction { get; init; } = MetricDirection.HigherIsBetter;

    /// <summary>
    /// Name of the metric (e.g., "Accuracy", "MSE").
    /// </summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>
    /// Category of the metric (e.g., "Classification", "Regression").
    /// </summary>
    public string? Category { get; init; }

    /// <summary>
    /// Human-readable description of the metric.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Whether this metric value is valid (not NaN or infinite).
    /// </summary>
    public bool IsValid
    {
        get
        {
            var valueDouble = NumOps.ToDouble(Value);
            return !double.IsNaN(valueDouble) && !double.IsInfinity(valueDouble);
        }
    }

    /// <summary>
    /// Whether a confidence interval is available.
    /// </summary>
    public bool HasConfidenceInterval => LowerBound != null && UpperBound != null;

    /// <summary>
    /// Width of the confidence interval (upper - lower). Null if CI not available.
    /// </summary>
    public T? IntervalWidth
    {
        get
        {
            if (!HasConfidenceInterval) return default;
            return NumOps.Subtract(UpperBound!, LowerBound!);
        }
    }

    /// <summary>
    /// Half-width of the confidence interval (margin of error). Null if CI not available.
    /// </summary>
    public T? MarginOfError
    {
        get
        {
            if (!HasConfidenceInterval) return default;
            var width = NumOps.Subtract(UpperBound!, LowerBound!);
            return NumOps.Divide(width, NumOps.FromDouble(2.0));
        }
    }

    /// <summary>
    /// Initializes a new metric with just a point estimate.
    /// </summary>
    /// <param name="value">The metric value.</param>
    /// <param name="name">Name of the metric.</param>
    /// <param name="direction">Whether higher is better.</param>
    public MetricWithCI(T value, string name = "", MetricDirection direction = MetricDirection.HigherIsBetter)
    {
        Value = value;
        Name = name;
        Direction = direction;
    }

    /// <summary>
    /// Initializes a new metric with a confidence interval.
    /// </summary>
    public MetricWithCI(
        T value,
        T lowerBound,
        T upperBound,
        double confidenceLevel,
        ConfidenceIntervalMethod ciMethod,
        string name = "",
        MetricDirection direction = MetricDirection.HigherIsBetter)
    {
        Value = value;
        LowerBound = lowerBound;
        UpperBound = upperBound;
        ConfidenceLevel = confidenceLevel;
        CIMethod = ciMethod;
        Name = name;
        Direction = direction;
    }

    /// <summary>
    /// Default parameterless constructor for serialization.
    /// </summary>
    public MetricWithCI()
    {
        Value = NumOps.Zero;
    }

    /// <summary>
    /// Formats the metric for display.
    /// </summary>
    /// <param name="decimalPlaces">Number of decimal places.</param>
    /// <param name="includeCI">Whether to include confidence interval.</param>
    /// <returns>Formatted string representation.</returns>
    public string Format(int decimalPlaces = 4, bool includeCI = true)
    {
        var valueStr = NumOps.ToDouble(Value).ToString($"F{decimalPlaces}");

        if (!includeCI || !HasConfidenceInterval)
        {
            return valueStr;
        }

        var lowerStr = NumOps.ToDouble(LowerBound!).ToString($"F{decimalPlaces}");
        var upperStr = NumOps.ToDouble(UpperBound!).ToString($"F{decimalPlaces}");

        return $"{valueStr} [{lowerStr}, {upperStr}]";
    }

    /// <summary>
    /// Formats the metric as a percentage.
    /// </summary>
    public string FormatAsPercentage(int decimalPlaces = 2, bool includeCI = true)
    {
        var valueStr = (NumOps.ToDouble(Value) * 100).ToString($"F{decimalPlaces}") + "%";

        if (!includeCI || !HasConfidenceInterval)
        {
            return valueStr;
        }

        var lowerStr = (NumOps.ToDouble(LowerBound!) * 100).ToString($"F{decimalPlaces}") + "%";
        var upperStr = (NumOps.ToDouble(UpperBound!) * 100).ToString($"F{decimalPlaces}") + "%";

        return $"{valueStr} [{lowerStr}, {upperStr}]";
    }

    /// <summary>
    /// Returns a string representation of the metric.
    /// </summary>
    public override string ToString()
    {
        return string.IsNullOrEmpty(Name) ? Format() : $"{Name}: {Format()}";
    }

    /// <summary>
    /// Checks if this metric is significantly better than another at the given significance level.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Two metrics are significantly different if their confidence
    /// intervals don't overlap. This is a simple approximation - formal statistical tests are
    /// more accurate.</para>
    /// </remarks>
    public bool IsSignificantlyBetterThan(MetricWithCI<T> other)
    {
        if (!HasConfidenceInterval || !other.HasConfidenceInterval)
        {
            return false;
        }

        // Simple non-overlap test (conservative)
        if (Direction == MetricDirection.HigherIsBetter)
        {
            // This metric's lower bound > other's upper bound
            return NumOps.Compare(LowerBound!, other.UpperBound!) > 0;
        }
        else
        {
            // This metric's upper bound < other's lower bound
            return NumOps.Compare(UpperBound!, other.LowerBound!) < 0;
        }
    }
}
