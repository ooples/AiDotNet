using AiDotNet.Evaluation.Options;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Engine for analyzing model robustness to input perturbations and noise.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Robustness analysis tests how well your model handles:
/// <list type="bullet">
/// <item><b>Noise:</b> Random perturbations to input features</item>
/// <item><b>Missing data:</b> Features replaced with default values</item>
/// <item><b>Outliers:</b> Extreme values in the input</item>
/// <item><b>Distribution shift:</b> Data that differs from training</item>
/// </list>
/// </para>
/// <para><b>Why robustness matters:</b>
/// <list type="bullet">
/// <item>Real-world data is noisy and imperfect</item>
/// <item>Small changes shouldn't cause dramatic prediction changes</item>
/// <item>Models should degrade gracefully, not catastrophically</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RobustnessEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly RobustnessOptions? _options;
    private readonly MetricComputationEngine<T> _metricEngine;

    /// <summary>
    /// Initializes the robustness engine.
    /// </summary>
    public RobustnessEngine(RobustnessOptions? options = null)
    {
        _options = options;
        _metricEngine = new MetricComputationEngine<T>();
    }

    /// <summary>
    /// Analyzes model robustness by testing with various perturbations.
    /// </summary>
    /// <typeparam name="TModel">The model type.</typeparam>
    /// <param name="features">Test feature matrix.</param>
    /// <param name="targets">Test target array.</param>
    /// <param name="predictFunc">Function to generate predictions.</param>
    /// <param name="model">The trained model to test.</param>
    /// <param name="metricName">Primary metric to track.</param>
    /// <param name="isClassification">Whether this is classification.</param>
    /// <param name="higherIsBetter">Whether higher metric values are better. Default true for accuracy-like metrics.</param>
    /// <returns>Robustness analysis results.</returns>
    public RobustnessResult<T> Analyze<TModel>(
        T[,] features,
        T[] targets,
        Func<TModel, T[,], T[]> predictFunc,
        TModel model,
        string metricName = "Accuracy",
        bool isClassification = true,
        bool higherIsBetter = true)
    {
        if (features == null)
            throw new ArgumentNullException(nameof(features));
        if (targets == null)
            throw new ArgumentNullException(nameof(targets));
        if (features.GetLength(0) != targets.Length)
            throw new ArgumentException("Features and targets must have same number of samples.");
        int n = features.GetLength(0);
        int numFeatures = features.GetLength(1);

        var result = new RobustnessResult<T>
        {
            MetricName = metricName,
            NumSamples = n,
            NumFeatures = numFeatures
        };

        // Baseline performance
        var baselinePreds = predictFunc(model, features);
        var baselineMetrics = isClassification
            ? _metricEngine.ComputeClassificationMetrics(baselinePreds, targets)
            : _metricEngine.ComputeRegressionMetrics(baselinePreds, targets);
        var baselineMetric = baselineMetrics[metricName];
        result.BaselineScore = baselineMetric != null ? NumOps.ToDouble(baselineMetric.Value) : 0;

        var random = _options?.RandomSeed.HasValue == true
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Test with different noise levels
        double[] noiseLevels = _options?.NoiseLevels ?? new[] { 0.01, 0.05, 0.1, 0.2, 0.5 };
        foreach (var noiseLevel in noiseLevels)
        {
            var noisyFeatures = AddGaussianNoise(features, noiseLevel, random);
            var noisyPreds = predictFunc(model, noisyFeatures);
            var noisyMetrics = isClassification
                ? _metricEngine.ComputeClassificationMetrics(noisyPreds, targets)
                : _metricEngine.ComputeRegressionMetrics(noisyPreds, targets);
            var noisyMetric = noisyMetrics[metricName];
            double score = noisyMetric != null ? NumOps.ToDouble(noisyMetric.Value) : 0;

            result.NoiseRobustness[noiseLevel] = score;
            // Degradation calculation respects metric direction
            result.NoiseDegradation[noiseLevel] = higherIsBetter
                ? result.BaselineScore - score  // Higher is better: degradation when score decreases
                : score - result.BaselineScore; // Lower is better: degradation when score increases
        }

        // Test feature dropout (missing data)
        double[] dropoutRates = _options?.DropoutRates ?? new[] { 0.1, 0.2, 0.3, 0.5 };
        foreach (var dropoutRate in dropoutRates)
        {
            var droppedFeatures = ApplyFeatureDropout(features, dropoutRate, random);
            var droppedPreds = predictFunc(model, droppedFeatures);
            var droppedMetrics = isClassification
                ? _metricEngine.ComputeClassificationMetrics(droppedPreds, targets)
                : _metricEngine.ComputeRegressionMetrics(droppedPreds, targets);
            var droppedMetric = droppedMetrics[metricName];
            double score = droppedMetric != null ? NumOps.ToDouble(droppedMetric.Value) : 0;

            result.DropoutRobustness[dropoutRate] = score;
        }

        // Per-feature importance via permutation
        result.FeatureImportance = ComputePermutationImportance(
            features, targets, predictFunc, model, metricName, isClassification, result.BaselineScore, random, higherIsBetter);

        // Compute overall robustness score
        result.OverallRobustnessScore = ComputeOverallRobustness(result);

        return result;
    }

    private T[,] AddGaussianNoise(T[,] features, double noiseLevel, Random random)
    {
        int rows = features.GetLength(0);
        int cols = features.GetLength(1);
        var result = new T[rows, cols];

        // Compute feature standard deviations
        var stds = new double[cols];
        for (int j = 0; j < cols; j++)
        {
            double sum = 0, sumSq = 0;
            for (int i = 0; i < rows; i++)
            {
                double val = NumOps.ToDouble(features[i, j]);
                sum += val;
                sumSq += val * val;
            }
            double mean = sum / rows;
            double variance = (sumSq / rows) - (mean * mean);
            stds[j] = Math.Sqrt(Math.Max(0, variance));
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double originalValue = NumOps.ToDouble(features[i, j]);
                // Box-Muller transform for Gaussian noise
                double u1 = 1.0 - random.NextDouble();
                double u2 = random.NextDouble();
                double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                double noisyValue = originalValue + noise * noiseLevel * stds[j];
                result[i, j] = NumOps.FromDouble(noisyValue);
            }
        }

        return result;
    }

    private T[,] ApplyFeatureDropout(T[,] features, double dropoutRate, Random random)
    {
        int rows = features.GetLength(0);
        int cols = features.GetLength(1);
        var result = new T[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (random.NextDouble() < dropoutRate)
                    result[i, j] = NumOps.Zero; // Replace with zero (or could use mean)
                else
                    result[i, j] = features[i, j];
            }
        }

        return result;
    }

    private Dictionary<int, double> ComputePermutationImportance<TModel>(
        T[,] features,
        T[] targets,
        Func<TModel, T[,], T[]> predictFunc,
        TModel model,
        string metricName,
        bool isClassification,
        double baselineScore,
        Random random,
        bool higherIsBetter)
    {
        int numFeatures = features.GetLength(1);
        int n = features.GetLength(0);
        var importance = new Dictionary<int, double>();

        for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++)
        {
            // Permute this feature
            var permutedFeatures = (T[,])features.Clone();
            var permutation = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();

            for (int i = 0; i < n; i++)
            {
                permutedFeatures[i, featureIdx] = features[permutation[i], featureIdx];
            }

            var permutedPreds = predictFunc(model, permutedFeatures);
            var permutedMetrics = isClassification
                ? _metricEngine.ComputeClassificationMetrics(permutedPreds, targets)
                : _metricEngine.ComputeRegressionMetrics(permutedPreds, targets);
            var permutedMetric = permutedMetrics[metricName];
            double score = permutedMetric != null ? NumOps.ToDouble(permutedMetric.Value) : 0;

            // Importance = how much the metric degrades when this feature is shuffled
            // For higher-is-better metrics: importance = baseline - permuted (positive = important)
            // For lower-is-better metrics: importance = permuted - baseline (positive = important)
            importance[featureIdx] = higherIsBetter ? baselineScore - score : score - baselineScore;
        }

        return importance;
    }

    private double ComputeOverallRobustness(RobustnessResult<T> result)
    {
        // Weighted average of robustness across different perturbation types
        double noiseRobustness = 0;
        if (result.NoiseRobustness.Count > 0)
        {
            noiseRobustness = result.NoiseRobustness.Values.Average() / Math.Max(0.001, result.BaselineScore);
        }

        double dropoutRobustness = 0;
        if (result.DropoutRobustness.Count > 0)
        {
            dropoutRobustness = result.DropoutRobustness.Values.Average() / Math.Max(0.001, result.BaselineScore);
        }

        return (noiseRobustness + dropoutRobustness) / 2;
    }
}

/// <summary>
/// Results from robustness analysis.
/// </summary>
public class RobustnessResult<T>
{
    /// <summary>
    /// Name of the metric tracked.
    /// </summary>
    public string MetricName { get; init; } = "";

    /// <summary>
    /// Number of samples tested.
    /// </summary>
    public int NumSamples { get; init; }

    /// <summary>
    /// Number of features in the dataset.
    /// </summary>
    public int NumFeatures { get; init; }

    /// <summary>
    /// Baseline score without perturbations.
    /// </summary>
    public double BaselineScore { get; set; }

    /// <summary>
    /// Performance at each noise level (noise level → score).
    /// </summary>
    public Dictionary<double, double> NoiseRobustness { get; init; } = new();

    /// <summary>
    /// Performance degradation at each noise level.
    /// </summary>
    public Dictionary<double, double> NoiseDegradation { get; init; } = new();

    /// <summary>
    /// Performance at each dropout rate (dropout rate → score).
    /// </summary>
    public Dictionary<double, double> DropoutRobustness { get; init; } = new();

    /// <summary>
    /// Feature importance via permutation (feature index → importance).
    /// </summary>
    public Dictionary<int, double> FeatureImportance { get; set; } = new();

    /// <summary>
    /// Overall robustness score (0 to 1, higher is better).
    /// </summary>
    public double OverallRobustnessScore { get; set; }
}
