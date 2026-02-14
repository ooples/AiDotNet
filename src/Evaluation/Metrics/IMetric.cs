using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;

namespace AiDotNet.Evaluation.Metrics;

/// <summary>
/// Base interface for all evaluation metrics.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("Metric")]
public interface IMetric<T>
{
    /// <summary>
    /// Gets the unique name of the metric.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the category of the metric (Classification, Regression, etc.).
    /// </summary>
    string Category { get; }

    /// <summary>
    /// Gets a human-readable description of the metric.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Gets whether higher values indicate better performance.
    /// </summary>
    MetricDirection Direction { get; }

    /// <summary>
    /// Gets the minimum possible value for this metric, if bounded.
    /// </summary>
    T? MinValue { get; }

    /// <summary>
    /// Gets the maximum possible value for this metric, if bounded.
    /// </summary>
    T? MaxValue { get; }

    /// <summary>
    /// Gets whether this metric requires probability predictions (not just class labels).
    /// </summary>
    bool RequiresProbabilities { get; }

    /// <summary>
    /// Gets whether this metric is suitable for multi-class classification.
    /// </summary>
    bool SupportsMultiClass { get; }
}

/// <summary>
/// Interface for classification metrics.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ClassificationMetric")]
public interface IClassificationMetric<T> : IMetric<T>
{
    /// <summary>
    /// Computes the metric from predicted and actual class labels.
    /// </summary>
    /// <param name="predictions">Predicted class labels.</param>
    /// <param name="actuals">Actual class labels.</param>
    /// <returns>The computed metric value.</returns>
    T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals);

    /// <summary>
    /// Computes the metric with full result including confidence interval.
    /// </summary>
    MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null);
}

/// <summary>
/// Interface for classification metrics that use probabilities.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IProbabilisticClassificationMetric<T> : IMetric<T>
{
    /// <summary>
    /// Computes the metric from predicted probabilities and actual labels.
    /// </summary>
    /// <param name="probabilities">Predicted probabilities (samples × classes for multi-class, or samples for binary).</param>
    /// <param name="actuals">Actual class labels.</param>
    /// <returns>The computed metric value.</returns>
    T Compute(ReadOnlySpan<T> probabilities, ReadOnlySpan<T> actuals, int numClasses = 2);

    /// <summary>
    /// Computes the metric with confidence interval.
    /// </summary>
    MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> probabilities,
        ReadOnlySpan<T> actuals,
        int numClasses = 2,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null);
}

/// <summary>
/// Interface for regression metrics.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IRegressionMetric<T> : IMetric<T>
{
    /// <summary>
    /// Computes the metric from predicted and actual values.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="actuals">Actual values.</param>
    /// <returns>The computed metric value.</returns>
    T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals);

    /// <summary>
    /// Computes the metric with confidence interval.
    /// </summary>
    MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null);
}

/// <summary>
/// Interface for ranking/recommendation metrics.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IRankingMetric<T> : IMetric<T>
{
    /// <summary>
    /// Computes the metric from predicted scores and relevance labels.
    /// </summary>
    /// <param name="scores">Predicted relevance scores (higher = more relevant).</param>
    /// <param name="relevance">Actual relevance labels (binary or graded).</param>
    /// <param name="k">Top-k cutoff. Null for no cutoff.</param>
    /// <returns>The computed metric value.</returns>
    T Compute(ReadOnlySpan<T> scores, ReadOnlySpan<T> relevance, int? k = null);
}

/// <summary>
/// Interface for time series forecasting metrics.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface ITimeSeriesMetric<T> : IMetric<T>
{
    /// <summary>
    /// Computes the metric from predicted and actual time series values.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="actuals">Actual values.</param>
    /// <param name="seasonalPeriod">Seasonal period for metrics like MASE. Null if not seasonal.</param>
    /// <returns>The computed metric value.</returns>
    T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, int? seasonalPeriod = null);
}
