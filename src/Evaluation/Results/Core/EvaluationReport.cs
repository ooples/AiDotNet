using AiDotNet.Evaluation.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Results.Core;

/// <summary>
/// Comprehensive evaluation report containing all computed metrics and analysis results.
/// </summary>
/// <remarks>
/// <para>
/// This is the main result object returned by model evaluation. It contains:
/// <list type="bullet">
/// <item>All computed metrics with confidence intervals</item>
/// <item>Dataset statistics</item>
/// <item>Diagnostic results (residual analysis, calibration, etc.)</item>
/// <item>Warnings and recommendations</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> After evaluating your model, this report tells you everything:
/// how accurate it is, where it struggles, what might be wrong, and how to improve it.
/// Start with the ExecutiveSummary for a quick overview, then dive into specific sections
/// as needed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for metric values.</typeparam>
public class EvaluationReport<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Unique identifier for this evaluation run.
    /// </summary>
    public Guid EvaluationId { get; init; } = Guid.NewGuid();

    /// <summary>
    /// Timestamp when evaluation was performed.
    /// </summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// Model name or identifier being evaluated.
    /// </summary>
    public string? ModelName { get; init; }

    /// <summary>
    /// Model version, if available.
    /// </summary>
    public string? ModelVersion { get; init; }

    /// <summary>
    /// Dataset name or identifier used for evaluation.
    /// </summary>
    public string? DatasetName { get; init; }

    /// <summary>
    /// Task type (Classification, Regression, etc.).
    /// </summary>
    public string TaskType { get; init; } = "Unknown";

    /// <summary>
    /// Statistics about the evaluation dataset.
    /// </summary>
    public DatasetStatistics<T>? DatasetStatistics { get; init; }

    /// <summary>
    /// Primary (most important) metric for this task.
    /// </summary>
    public MetricWithCI<T>? PrimaryMetric { get; init; }

    /// <summary>
    /// Name of the primary metric.
    /// </summary>
    public string? PrimaryMetricName { get; init; }

    /// <summary>
    /// All computed metrics organized by category.
    /// </summary>
    public MetricCollection<T> Metrics { get; init; } = new();

    /// <summary>
    /// Classification-specific results (confusion matrix, ROC, etc.). Null for regression.
    /// </summary>
    public ClassificationResults<T>? ClassificationResults { get; init; }

    /// <summary>
    /// Regression-specific results (residual analysis, etc.). Null for classification.
    /// </summary>
    public RegressionResults<T>? RegressionResults { get; init; }

    /// <summary>
    /// Calibration analysis results. Null if not computed.
    /// </summary>
    public object? CalibrationResults { get; init; }

    /// <summary>
    /// Fairness analysis results. Null if not computed.
    /// </summary>
    public object? FairnessResults { get; init; }

    /// <summary>
    /// Robustness analysis results. Null if not computed.
    /// </summary>
    public object? RobustnessResults { get; init; }

    /// <summary>
    /// Uncertainty analysis results. Null if not computed.
    /// </summary>
    public object? UncertaintyResults { get; init; }

    /// <summary>
    /// Total time taken for evaluation.
    /// </summary>
    public TimeSpan? EvaluationDuration { get; init; }

    /// <summary>
    /// Time breakdown by component.
    /// </summary>
    public Dictionary<string, TimeSpan>? ComponentTimings { get; init; }

    /// <summary>
    /// Warnings generated during evaluation.
    /// </summary>
    public IReadOnlyList<EvaluationWarning> Warnings { get; init; } = Array.Empty<EvaluationWarning>();

    /// <summary>
    /// Recommendations for improvement.
    /// </summary>
    public IReadOnlyList<string> Recommendations { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Metadata about the evaluation configuration.
    /// </summary>
    public Dictionary<string, object>? Metadata { get; init; }

    /// <summary>
    /// Generates an executive summary of the evaluation results.
    /// </summary>
    public string GetExecutiveSummary()
    {
        var lines = new List<string>
        {
            "=== Model Evaluation Summary ===",
            string.Empty,
            $"Model: {ModelName ?? "Unknown"}",
            $"Task: {TaskType}",
            $"Evaluated: {Timestamp:yyyy-MM-dd HH:mm:ss UTC}"
        };

        if (DatasetStatistics != null)
        {
            lines.Add($"Samples: {DatasetStatistics.TotalSamples:N0}");
        }

        lines.Add(string.Empty);

        // Primary metric
        if (PrimaryMetric != null)
        {
            lines.Add($"Primary Metric ({PrimaryMetricName ?? "Unknown"}): {PrimaryMetric.Format()}");
        }

        // Key metrics summary
        lines.Add(string.Empty);
        lines.Add("Key Metrics:");

        var keyMetricNames = GetKeyMetricNames();
        foreach (var name in keyMetricNames)
        {
            var metric = Metrics[name];
            if (metric != null)
            {
                lines.Add($"  {name}: {metric.Format()}");
            }
        }

        // Warnings
        if (Warnings.Count > 0)
        {
            lines.Add(string.Empty);
            lines.Add($"⚠ {Warnings.Count} Warning(s):");
            foreach (var warning in Warnings.Take(3))
            {
                lines.Add($"  - {warning.Message}");
            }

            if (Warnings.Count > 3)
            {
                lines.Add($"  ... and {Warnings.Count - 3} more");
            }
        }

        // Recommendations
        if (Recommendations.Count > 0)
        {
            lines.Add(string.Empty);
            lines.Add("Recommendations:");
            foreach (var rec in Recommendations.Take(3))
            {
                lines.Add($"  • {rec}");
            }
        }

        return string.Join(Environment.NewLine, lines);
    }

    /// <summary>
    /// Gets a specific metric by name.
    /// </summary>
    public MetricWithCI<T>? GetMetric(string name) => Metrics[name];

    /// <summary>
    /// Gets all metrics as a dictionary.
    /// </summary>
    public Dictionary<string, T> GetMetricsAsDictionary() => Metrics.ToDictionary();

    /// <summary>
    /// Checks if a specific metric was computed.
    /// </summary>
    public bool HasMetric(string name) => Metrics.Contains(name);

    /// <summary>
    /// Gets the overall quality assessment (Poor, Fair, Good, Excellent).
    /// </summary>
    public string GetQualityAssessment()
    {
        if (PrimaryMetric == null) return "Unknown";

        var value = NumOps.ToDouble(PrimaryMetric.Value);
        var direction = PrimaryMetric.Direction;

        // Normalize value to 0-1 scale (assuming metrics are typically 0-1 or need conversion)
        double normalizedValue = direction == MetricDirection.LowerIsBetter
            ? 1.0 - Math.Min(1.0, value)
            : Math.Min(1.0, value);

        return normalizedValue switch
        {
            >= 0.95 => "Excellent",
            >= 0.85 => "Good",
            >= 0.70 => "Fair",
            _ => "Needs Improvement"
        };
    }

    private IEnumerable<string> GetKeyMetricNames()
    {
        // Return key metrics based on task type
        if (TaskType.Contains("Classification", StringComparison.OrdinalIgnoreCase))
        {
            return new[] { "Accuracy", "Precision", "Recall", "F1Score", "AUCROC" };
        }
        else if (TaskType.Contains("Regression", StringComparison.OrdinalIgnoreCase))
        {
            return new[] { "R2Score", "RMSE", "MAE", "MAPE" };
        }
        else
        {
            return Metrics.Names.Take(5);
        }
    }
}

/// <summary>
/// Warning generated during evaluation.
/// </summary>
public class EvaluationWarning
{
    /// <summary>
    /// Warning severity level.
    /// </summary>
    public WarningSeverity Severity { get; init; }

    /// <summary>
    /// Warning category.
    /// </summary>
    public string Category { get; init; } = string.Empty;

    /// <summary>
    /// Warning message.
    /// </summary>
    public string Message { get; init; } = string.Empty;

    /// <summary>
    /// Suggested action to address the warning.
    /// </summary>
    public string? SuggestedAction { get; init; }

    /// <summary>
    /// Related metric or data element.
    /// </summary>
    public string? RelatedElement { get; init; }
}

/// <summary>
/// Severity levels for evaluation warnings.
/// </summary>
public enum WarningSeverity
{
    /// <summary>Informational only.</summary>
    Info = 0,

    /// <summary>Minor issue that may affect results.</summary>
    Low = 1,

    /// <summary>Moderate issue that likely affects results.</summary>
    Medium = 2,

    /// <summary>Serious issue that significantly affects results.</summary>
    High = 3,

    /// <summary>Critical issue that invalidates results.</summary>
    Critical = 4
}

/// <summary>
/// Classification-specific results container.
/// </summary>
public class ClassificationResults<T>
{
    /// <summary>
    /// Confusion matrix.
    /// </summary>
    public object? ConfusionMatrix { get; init; }

    /// <summary>
    /// ROC curve data.
    /// </summary>
    public object? ROCCurve { get; init; }

    /// <summary>
    /// Precision-Recall curve data.
    /// </summary>
    public object? PRCurve { get; init; }

    /// <summary>
    /// Per-class metrics (keyed by class label string representation).
    /// </summary>
    public Dictionary<string, MetricCollection<T>>? PerClassMetrics { get; init; }

    /// <summary>
    /// Optimal threshold analysis.
    /// </summary>
    public object? ThresholdAnalysis { get; init; }

    /// <summary>
    /// Selected classification threshold.
    /// </summary>
    public T? SelectedThreshold { get; init; }

    /// <summary>
    /// Threshold selection method used.
    /// </summary>
    public ThresholdSelectionMethod? ThresholdMethod { get; init; }
}

/// <summary>
/// Regression-specific results container.
/// </summary>
public class RegressionResults<T>
{
    /// <summary>
    /// Residual analysis results.
    /// </summary>
    public object? ResidualAnalysis { get; init; }

    /// <summary>
    /// Influence analysis results.
    /// </summary>
    public object? InfluenceAnalysis { get; init; }

    /// <summary>
    /// Q-Q plot data for normality assessment.
    /// </summary>
    public object? QQPlotData { get; init; }

    /// <summary>
    /// Predicted vs actual values for plotting.
    /// </summary>
    public (T[] Predicted, T[] Actual)? PredictedVsActual { get; init; }

    /// <summary>
    /// Residual statistics.
    /// </summary>
    public ResidualStatistics<T>? ResidualStats { get; init; }

    /// <summary>
    /// Results of residual diagnostic tests.
    /// </summary>
    public Dictionary<string, object>? DiagnosticTests { get; init; }
}

/// <summary>
/// Statistics about regression residuals.
/// </summary>
public class ResidualStatistics<T>
{
    /// <summary>Mean of residuals (should be close to 0).</summary>
    public T? Mean { get; init; }

    /// <summary>Standard deviation of residuals.</summary>
    public T? StdDev { get; init; }

    /// <summary>Minimum residual.</summary>
    public T? Min { get; init; }

    /// <summary>Maximum residual.</summary>
    public T? Max { get; init; }

    /// <summary>Median residual.</summary>
    public T? Median { get; init; }

    /// <summary>Skewness of residuals (0 for normal).</summary>
    public double? Skewness { get; init; }

    /// <summary>Kurtosis of residuals (3 for normal).</summary>
    public double? Kurtosis { get; init; }

    /// <summary>Durbin-Watson statistic (2 for no autocorrelation).</summary>
    public double? DurbinWatson { get; init; }

    /// <summary>Number of outlier residuals.</summary>
    public int OutlierCount { get; init; }
}
