using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics;
using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Metrics.Regression;
using AiDotNet.Evaluation.Options;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Core engine for computing evaluation metrics across all task types.
/// </summary>
/// <remarks>
/// <para>
/// This engine provides a unified interface for computing classification, regression,
/// and time series metrics with confidence intervals and proper handling of edge cases.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the heart of model evaluation. Give it predictions and actuals,
/// and it computes all relevant metrics automatically based on task type.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MetricComputationEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly EvaluationOptions<T> _options;

    // Classification metrics registry
    private readonly Dictionary<string, IClassificationMetric<T>> _classificationMetrics;
    private readonly Dictionary<string, IProbabilisticClassificationMetric<T>> _probabilisticMetrics;
    private readonly Dictionary<string, IRegressionMetric<T>> _regressionMetrics;

    /// <summary>
    /// Initializes the metric computation engine with default or custom options.
    /// </summary>
    public MetricComputationEngine(EvaluationOptions<T>? options = null)
    {
        _options = options ?? new EvaluationOptions<T>();
        _classificationMetrics = new Dictionary<string, IClassificationMetric<T>>(StringComparer.OrdinalIgnoreCase);
        _probabilisticMetrics = new Dictionary<string, IProbabilisticClassificationMetric<T>>(StringComparer.OrdinalIgnoreCase);
        _regressionMetrics = new Dictionary<string, IRegressionMetric<T>>(StringComparer.OrdinalIgnoreCase);

        RegisterDefaultMetrics();
    }

    private void RegisterDefaultMetrics()
    {
        // Classification metrics
        RegisterClassificationMetric(new AccuracyMetric<T>());
        RegisterClassificationMetric(new BalancedAccuracyMetric<T>());
        RegisterClassificationMetric(new PrecisionMetric<T>());
        RegisterClassificationMetric(new RecallMetric<T>());
        RegisterClassificationMetric(new F1ScoreMetric<T>());
        RegisterClassificationMetric(new SpecificityMetric<T>());
        RegisterClassificationMetric(new MatthewsCorrelationCoefficientMetric<T>());
        RegisterClassificationMetric(new CohensKappaMetric<T>());

        // Multi-class averaging variants
        RegisterClassificationMetric(new PrecisionMetric<T>(averaging: AveragingMethod.Macro), "Precision_Macro");
        RegisterClassificationMetric(new PrecisionMetric<T>(averaging: AveragingMethod.Weighted), "Precision_Weighted");
        RegisterClassificationMetric(new RecallMetric<T>(averaging: AveragingMethod.Macro), "Recall_Macro");
        RegisterClassificationMetric(new RecallMetric<T>(averaging: AveragingMethod.Weighted), "Recall_Weighted");
        RegisterClassificationMetric(new F1ScoreMetric<T>(averaging: AveragingMethod.Macro), "F1Score_Macro");
        RegisterClassificationMetric(new F1ScoreMetric<T>(averaging: AveragingMethod.Weighted), "F1Score_Weighted");

        // F-beta variants
        RegisterClassificationMetric(new FBetaScoreMetric<T>(beta: 0.5), "F0.5Score");
        RegisterClassificationMetric(new FBetaScoreMetric<T>(beta: 2.0), "F2Score");

        // Additional classification metrics
        RegisterClassificationMetric(new JaccardScoreMetric<T>());
        RegisterClassificationMetric(new HammingLossMetric<T>());
        RegisterClassificationMetric(new HingeLossMetric<T>());
        RegisterClassificationMetric(new ZeroOneLossMetric<T>());
        RegisterClassificationMetric(new NPVMetric<T>());
        RegisterClassificationMetric(new FalsePositiveRateMetric<T>());
        RegisterProbabilisticMetric(new GiniCoefficientMetric<T>());
        RegisterClassificationMetric(new TrueNegativeRateMetric<T>());
        RegisterClassificationMetric(new FalseNegativeRateMetric<T>());
        RegisterClassificationMetric(new InformednessMetric<T>());
        RegisterClassificationMetric(new MarkednessMetric<T>());
        RegisterClassificationMetric(new AUCPRMetric<T>());
        RegisterClassificationMetric(new FowlkesMallowsMetric<T>());
        RegisterClassificationMetric(new ThreatScoreMetric<T>());
        RegisterClassificationMetric(new OptimizedPrecisionMetric<T>());
        RegisterClassificationMetric(new DiagnosticOddsRatioMetric<T>());
        RegisterClassificationMetric(new PositiveLikelihoodRatioMetric<T>());
        RegisterClassificationMetric(new NegativeLikelihoodRatioMetric<T>());
        RegisterClassificationMetric(new PrevalenceThresholdMetric<T>());
        RegisterClassificationMetric(new BalancedErrorRateMetric<T>());

        // Probabilistic metrics
        RegisterProbabilisticMetric(new LogLossMetric<T>());
        RegisterProbabilisticMetric(new AUCROCMetric<T>());
        RegisterProbabilisticMetric(new BrierScoreMetric<T>());

        // Regression metrics
        RegisterRegressionMetric(new MAEMetric<T>());
        RegisterRegressionMetric(new MSEMetric<T>());
        RegisterRegressionMetric(new RMSEMetric<T>());
        RegisterRegressionMetric(new R2ScoreMetric<T>());
        RegisterRegressionMetric(new MAPEMetric<T>());
        RegisterRegressionMetric(new ExplainedVarianceMetric<T>());
        RegisterRegressionMetric(new AdjustedR2Metric<T>());
        RegisterRegressionMetric(new HuberLossMetric<T>());
        RegisterRegressionMetric(new RMSLEMetric<T>());
        RegisterRegressionMetric(new MaxErrorMetric<T>());
        RegisterRegressionMetric(new MedianAbsoluteErrorMetric<T>());
        RegisterRegressionMetric(new MeanBiasErrorMetric<T>());
        RegisterRegressionMetric(new SymmetricMAPEMetric<T>());
        RegisterRegressionMetric(new MeanSquaredLogErrorMetric<T>());
        RegisterRegressionMetric(new TweedieLossMetric<T>());
        RegisterRegressionMetric(new QuantileLossMetric<T>());
        RegisterRegressionMetric(new LogCoshLossMetric<T>());
        RegisterRegressionMetric(new PoissonDevianceMetric<T>());
        RegisterRegressionMetric(new NormalizedMSEMetric<T>());
        RegisterRegressionMetric(new RelativeSquaredErrorMetric<T>());
        RegisterRegressionMetric(new RelativeAbsoluteErrorMetric<T>());
    }

    /// <summary>
    /// Registers a classification metric.
    /// </summary>
    public void RegisterClassificationMetric(IClassificationMetric<T> metric, string? name = null)
    {
        _classificationMetrics[name ?? metric.Name] = metric;
    }

    /// <summary>
    /// Registers a probabilistic classification metric.
    /// </summary>
    public void RegisterProbabilisticMetric(IProbabilisticClassificationMetric<T> metric, string? name = null)
    {
        _probabilisticMetrics[name ?? metric.Name] = metric;
    }

    /// <summary>
    /// Registers a regression metric.
    /// </summary>
    public void RegisterRegressionMetric(IRegressionMetric<T> metric, string? name = null)
    {
        _regressionMetrics[name ?? metric.Name] = metric;
    }

    /// <summary>
    /// Computes all classification metrics.
    /// </summary>
    public MetricCollection<T> ComputeClassificationMetrics(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ReadOnlySpan<T> probabilities = default,
        int numClasses = 2)
    {
        var collection = new MetricCollection<T>();
        bool computeCI = _options.ComputeConfidenceIntervals ?? true;
        double confLevel = _options.ConfidenceLevel ?? 0.95;
        int bootstrapSamples = _options.BootstrapSamples ?? 1000;
        var ciMethod = _options.ConfidenceIntervalMethod ?? ConfidenceIntervalMethod.PercentileBootstrap;

        // Compute label-based metrics
        foreach (var kvp in _classificationMetrics)
        {
            try
            {
                MetricWithCI<T> result;
                if (computeCI)
                {
                    result = kvp.Value.ComputeWithCI(predictions, actuals, ciMethod, confLevel, bootstrapSamples, _options.RandomSeed);
                }
                else
                {
                    var value = kvp.Value.Compute(predictions, actuals);
                    result = new MetricWithCI<T>(value, kvp.Key, kvp.Value.Direction)
                    {
                        Category = kvp.Value.Category,
                        Description = kvp.Value.Description
                    };
                }
                collection.Add(result);
            }
            catch (Exception)
            {
                // Skip metrics that fail (e.g., due to data issues)
            }
        }

        // Compute probability-based metrics if probabilities provided
        if (!probabilities.IsEmpty)
        {
            foreach (var kvp in _probabilisticMetrics)
            {
                try
                {
                    MetricWithCI<T> result;
                    if (computeCI)
                    {
                        result = kvp.Value.ComputeWithCI(probabilities, actuals, numClasses, ciMethod, confLevel, bootstrapSamples, _options.RandomSeed);
                    }
                    else
                    {
                        var value = kvp.Value.Compute(probabilities, actuals, numClasses);
                        result = new MetricWithCI<T>(value, kvp.Key, kvp.Value.Direction)
                        {
                            Category = kvp.Value.Category,
                            Description = kvp.Value.Description
                        };
                    }
                    collection.Add(result);
                }
                catch (Exception)
                {
                    // Skip metrics that fail
                }
            }
        }

        return collection;
    }

    /// <summary>
    /// Computes all regression metrics.
    /// </summary>
    public MetricCollection<T> ComputeRegressionMetrics(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals)
    {
        var collection = new MetricCollection<T>();
        bool computeCI = _options.ComputeConfidenceIntervals ?? true;
        double confLevel = _options.ConfidenceLevel ?? 0.95;
        int bootstrapSamples = _options.BootstrapSamples ?? 1000;
        var ciMethod = _options.ConfidenceIntervalMethod ?? ConfidenceIntervalMethod.PercentileBootstrap;

        foreach (var kvp in _regressionMetrics)
        {
            try
            {
                MetricWithCI<T> result;
                if (computeCI)
                {
                    result = kvp.Value.ComputeWithCI(predictions, actuals, ciMethod, confLevel, bootstrapSamples, _options.RandomSeed);
                }
                else
                {
                    var value = kvp.Value.Compute(predictions, actuals);
                    result = new MetricWithCI<T>(value, kvp.Key, kvp.Value.Direction)
                    {
                        Category = kvp.Value.Category,
                        Description = kvp.Value.Description
                    };
                }
                collection.Add(result);
            }
            catch (Exception)
            {
                // Skip metrics that fail
            }
        }

        return collection;
    }

    /// <summary>
    /// Computes a specific metric by name.
    /// </summary>
    public MetricWithCI<T>? ComputeMetric(
        string metricName,
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ReadOnlySpan<T> probabilities = default,
        int numClasses = 2)
    {
        bool computeCI = _options.ComputeConfidenceIntervals ?? true;
        double confLevel = _options.ConfidenceLevel ?? 0.95;
        int bootstrapSamples = _options.BootstrapSamples ?? 1000;
        var ciMethod = _options.ConfidenceIntervalMethod ?? ConfidenceIntervalMethod.PercentileBootstrap;

        if (_classificationMetrics.TryGetValue(metricName, out var classMetric))
        {
            return computeCI
                ? classMetric.ComputeWithCI(predictions, actuals, ciMethod, confLevel, bootstrapSamples, _options.RandomSeed)
                : new MetricWithCI<T>(classMetric.Compute(predictions, actuals), metricName, classMetric.Direction);
        }

        if (!probabilities.IsEmpty && _probabilisticMetrics.TryGetValue(metricName, out var probMetric))
        {
            return computeCI
                ? probMetric.ComputeWithCI(probabilities, actuals, numClasses, ciMethod, confLevel, bootstrapSamples, _options.RandomSeed)
                : new MetricWithCI<T>(probMetric.Compute(probabilities, actuals, numClasses), metricName, probMetric.Direction);
        }

        if (_regressionMetrics.TryGetValue(metricName, out var regMetric))
        {
            return computeCI
                ? regMetric.ComputeWithCI(predictions, actuals, ciMethod, confLevel, bootstrapSamples, _options.RandomSeed)
                : new MetricWithCI<T>(regMetric.Compute(predictions, actuals), metricName, regMetric.Direction);
        }

        return null;
    }

    /// <summary>
    /// Gets all available metric names.
    /// </summary>
    public IReadOnlyList<string> GetAvailableMetricNames(bool classification = true, bool regression = true, bool probabilistic = true)
    {
        var names = new List<string>();
        if (classification) names.AddRange(_classificationMetrics.Keys);
        if (probabilistic) names.AddRange(_probabilisticMetrics.Keys);
        if (regression) names.AddRange(_regressionMetrics.Keys);
        return names;
    }
}
