using AiDotNet.Evaluation.CrossValidation;
using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Options;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Engine for executing cross-validation with various strategies and aggregating results.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This engine automates the cross-validation process:
/// <list type="bullet">
/// <item>Splits your data according to the chosen strategy</item>
/// <item>Trains your model on each training fold</item>
/// <item>Evaluates on each validation fold</item>
/// <item>Aggregates results with confidence intervals</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CrossValidationEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Options.CrossValidationOptions? _options;
    private readonly MetricComputationEngine<T> _metricEngine;

    /// <summary>
    /// Initializes the cross-validation engine.
    /// </summary>
    /// <param name="options">Cross-validation options. If null, uses defaults.</param>
    public CrossValidationEngine(Options.CrossValidationOptions? options = null)
    {
        _options = options;
        _metricEngine = new MetricComputationEngine<T>();
    }

    /// <summary>
    /// Performs cross-validation using the specified strategy and model training function.
    /// </summary>
    /// <typeparam name="TModel">The model type.</typeparam>
    /// <param name="strategy">The cross-validation strategy to use.</param>
    /// <param name="features">The feature data (samples × features).</param>
    /// <param name="targets">The target values.</param>
    /// <param name="trainFunc">Function that trains a model on training data and returns it.</param>
    /// <param name="predictFunc">Function that generates predictions from a trained model.</param>
    /// <param name="isClassification">Whether this is a classification task.</param>
    /// <returns>Cross-validation results with aggregated metrics.</returns>
    public CrossValidationResult<T> Execute<TModel>(
        ICrossValidationStrategy<T> strategy,
        T[,] features,
        T[] targets,
        Func<T[,], T[], TModel> trainFunc,
        Func<TModel, T[,], T[]> predictFunc,
        bool isClassification = true)
    {
        int numSamples = features.GetLength(0);
        int numFeatures = features.GetLength(1);

        if (numSamples != targets.Length)
            throw new ArgumentException("Features and targets must have the same number of samples.");

        var foldResults = new List<FoldResult<T>>();
        int foldIndex = 0;

        foreach (var (trainIndices, valIndices) in strategy.Split(numSamples, targets))
        {
            // Extract train and validation data
            var trainFeatures = ExtractRows(features, trainIndices);
            var trainTargets = ExtractElements(targets, trainIndices);
            var valFeatures = ExtractRows(features, valIndices);
            var valTargets = ExtractElements(targets, valIndices);

            // Train model
            var model = trainFunc(trainFeatures, trainTargets);

            // Generate predictions
            var predictions = predictFunc(model, valFeatures);

            // Compute metrics
            var metrics = isClassification
                ? _metricEngine.ComputeClassificationMetrics(predictions, valTargets)
                : _metricEngine.ComputeRegressionMetrics(predictions, valTargets);

            foldResults.Add(new FoldResult<T>
            {
                FoldIndex = foldIndex,
                TrainSize = trainIndices.Length,
                ValidationSize = valIndices.Length,
                Metrics = metrics,
                Predictions = predictions,
                Actuals = valTargets
            });

            foldIndex++;
        }

        return AggregateResults(strategy, foldResults);
    }

    /// <summary>
    /// Performs cross-validation for time series data.
    /// </summary>
    public CrossValidationResult<T> ExecuteTimeSeries<TModel>(
        ICrossValidationStrategy<T> strategy,
        T[] series,
        int lookback,
        Func<T[], TModel> trainFunc,
        Func<TModel, T[], T[]> predictFunc,
        int horizon = 1)
    {
        int numSamples = series.Length - lookback - horizon + 1;
        if (numSamples < 2)
            throw new ArgumentException("Not enough data points for the specified lookback and horizon.");

        var foldResults = new List<FoldResult<T>>();
        int foldIndex = 0;

        foreach (var (trainIndices, valIndices) in strategy.Split(numSamples))
        {
            // For time series, indices refer to starting positions of windows
            var trainSeries = new T[trainIndices.Max() + lookback + horizon];
            Array.Copy(series, trainSeries, Math.Min(series.Length, trainSeries.Length));

            // Train model
            var model = trainFunc(trainSeries);

            // Generate predictions for validation indices
            var predictions = new T[valIndices.Length];
            var actuals = new T[valIndices.Length];

            for (int i = 0; i < valIndices.Length; i++)
            {
                int startIdx = valIndices[i];
                var window = new T[lookback];
                Array.Copy(series, startIdx, window, 0, lookback);

                var pred = predictFunc(model, window);
                predictions[i] = pred[0]; // First prediction in horizon
                actuals[i] = series[startIdx + lookback]; // Actual value at horizon
            }

            // Compute time series metrics
            var metrics = _metricEngine.ComputeRegressionMetrics(predictions, actuals);

            foldResults.Add(new FoldResult<T>
            {
                FoldIndex = foldIndex,
                TrainSize = trainIndices.Length,
                ValidationSize = valIndices.Length,
                Metrics = metrics,
                Predictions = predictions,
                Actuals = actuals
            });

            foldIndex++;
        }

        return AggregateResults(strategy, foldResults);
    }

    private CrossValidationResult<T> AggregateResults(
        ICrossValidationStrategy<T> strategy,
        List<FoldResult<T>> foldResults)
    {
        var aggregatedMetrics = new MetricCollection<T>();

        // Get all metric names from all folds
        var allMetricNames = foldResults
            .SelectMany(f => f.Metrics.Names)
            .Distinct()
            .ToList();

        foreach (var metricName in allMetricNames)
        {
            var values = new List<double>();

            foreach (var fold in foldResults)
            {
                var metric = fold.Metrics[metricName];
                if (metric != null)
                    values.Add(NumOps.ToDouble(metric.Value));
            }

            if (values.Count == 0) continue;

            // Calculate mean and std
            double mean = values.Average();
            double variance = values.Count > 1
                ? values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1)
                : 0;
            double std = Math.Sqrt(variance);

            // Get direction from first available metric
            var sampleMetric = foldResults.First().Metrics[metricName];
            var direction = sampleMetric?.Direction ?? MetricDirection.HigherIsBetter;

            // Calculate 95% CI using t-distribution approximation
            double tValue = 1.96; // Approximate for large n
            if (values.Count < 30)
            {
                // Simple approximation for small samples
                tValue = 2.0 + (4.0 / values.Count);
            }
            double margin = tValue * std / Math.Sqrt(values.Count);

            var aggregated = new MetricWithCI<T>(
                NumOps.FromDouble(mean),
                NumOps.FromDouble(mean - margin),
                NumOps.FromDouble(mean + margin),
                0.95,
                ConfidenceIntervalMethod.NormalApproximation,
                metricName,
                direction)
            {
                Category = sampleMetric?.Category ?? "Unknown",
                Description = sampleMetric?.Description,
                StandardDeviation = NumOps.FromDouble(std)
            };

            aggregatedMetrics.Add(aggregated);
        }

        return new CrossValidationResult<T>
        {
            StrategyName = strategy.Name,
            NumFolds = foldResults.Count,
            FoldResults = foldResults,
            AggregatedMetrics = aggregatedMetrics
        };
    }

    private static T[,] ExtractRows(T[,] matrix, int[] indices)
    {
        int numCols = matrix.GetLength(1);
        var result = new T[indices.Length, numCols];

        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = matrix[indices[i], j];
            }
        }

        return result;
    }

    private static T[] ExtractElements(T[] array, int[] indices)
    {
        var result = new T[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            result[i] = array[indices[i]];
        }
        return result;
    }
}

/// <summary>
/// Results from a single cross-validation fold.
/// </summary>
public class FoldResult<T>
{
    /// <summary>
    /// The fold index (0-based).
    /// </summary>
    public int FoldIndex { get; init; }

    /// <summary>
    /// Number of training samples in this fold.
    /// </summary>
    public int TrainSize { get; init; }

    /// <summary>
    /// Number of validation samples in this fold.
    /// </summary>
    public int ValidationSize { get; init; }

    /// <summary>
    /// Metrics computed on this fold's validation set.
    /// </summary>
    public MetricCollection<T> Metrics { get; init; } = new();

    /// <summary>
    /// Predictions on the validation set.
    /// </summary>
    public T[] Predictions { get; init; } = Array.Empty<T>();

    /// <summary>
    /// Actual values from the validation set.
    /// </summary>
    public T[] Actuals { get; init; } = Array.Empty<T>();
}

/// <summary>
/// Aggregated results from cross-validation across all folds.
/// </summary>
public class CrossValidationResult<T>
{
    /// <summary>
    /// Name of the cross-validation strategy used.
    /// </summary>
    public string StrategyName { get; init; } = "";

    /// <summary>
    /// Number of folds executed.
    /// </summary>
    public int NumFolds { get; init; }

    /// <summary>
    /// Results from each individual fold.
    /// </summary>
    public List<FoldResult<T>> FoldResults { get; init; } = new();

    /// <summary>
    /// Metrics aggregated across all folds with mean, std, and confidence intervals.
    /// </summary>
    public MetricCollection<T> AggregatedMetrics { get; init; } = new();

    /// <summary>
    /// Gets the mean value for a specific metric.
    /// </summary>
    public T GetMeanMetric(string metricName)
    {
        var metric = AggregatedMetrics[metricName];
        return metric != null ? metric.Value : default!;
    }

    /// <summary>
    /// Gets the standard deviation for a specific metric across folds.
    /// </summary>
    public T GetStdMetric(string metricName)
    {
        var metric = AggregatedMetrics[metricName];
        if (metric == null || metric.StandardDeviation == null)
            return default!;
        return metric.StandardDeviation;
    }
}
