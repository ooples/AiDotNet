using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Options;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Engine for generating learning curves: how model performance changes with training set size.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Learning curves show how your model improves as you give it more data:
/// <list type="bullet">
/// <item>X-axis: Number of training samples</item>
/// <item>Y-axis: Performance metric (e.g., accuracy)</item>
/// <item>Typically shows both training and validation scores</item>
/// </list>
/// </para>
/// <para><b>What learning curves tell you:</b>
/// <list type="bullet">
/// <item><b>High bias (underfitting)</b>: Both curves plateau at low performance</item>
/// <item><b>High variance (overfitting)</b>: Large gap between train (high) and val (low)</item>
/// <item><b>Need more data</b>: Validation curve still improving at max training size</item>
/// <item><b>Good fit</b>: Both curves converge at high performance</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LearningCurveEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly LearningCurveOptions? _options;
    private readonly MetricComputationEngine<T> _metricEngine;

    /// <summary>
    /// Initializes the learning curve engine.
    /// </summary>
    public LearningCurveEngine(LearningCurveOptions? options = null)
    {
        _options = options;
        _metricEngine = new MetricComputationEngine<T>();
    }

    /// <summary>
    /// Generates a learning curve by training at various dataset sizes.
    /// </summary>
    /// <typeparam name="TModel">The model type.</typeparam>
    /// <param name="features">Full feature matrix.</param>
    /// <param name="targets">Full target array.</param>
    /// <param name="trainFunc">Function to train a model.</param>
    /// <param name="predictFunc">Function to generate predictions.</param>
    /// <param name="metricName">Primary metric to track (e.g., "Accuracy", "RMSE").</param>
    /// <param name="trainSizes">Training sizes as fractions (0-1) or absolute counts.</param>
    /// <param name="cvFolds">Number of cross-validation folds per training size.</param>
    /// <param name="isClassification">Whether this is classification.</param>
    /// <param name="higherIsBetter">Whether higher metric values are better. Default true for accuracy-like metrics, false for error metrics like RMSE.</param>
    /// <returns>Learning curve results.</returns>
    public LearningCurveResult<T> Generate<TModel>(
        T[,] features,
        T[] targets,
        Func<T[,], T[], TModel> trainFunc,
        Func<TModel, T[,], T[]> predictFunc,
        string metricName = "Accuracy",
        double[]? trainSizes = null,
        int cvFolds = 5,
        bool isClassification = true,
        bool higherIsBetter = true)
    {
        int totalSamples = features.GetLength(0);
        int numFeatures = features.GetLength(1);

        // Default train sizes: 10%, 25%, 50%, 75%, 100%
        trainSizes ??= new[] { 0.1, 0.25, 0.5, 0.75, 1.0 };

        var result = new LearningCurveResult<T>
        {
            MetricName = metricName,
            TotalSamples = totalSamples,
            CVFolds = cvFolds
        };

        var random = _options?.RandomSeed.HasValue == true
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : new Random();

        foreach (var sizeSpec in trainSizes)
        {
            // Convert to absolute size
            int trainSize = sizeSpec <= 1.0
                ? (int)(sizeSpec * totalSamples)
                : (int)sizeSpec;

            trainSize = Math.Max(cvFolds + 1, Math.Min(trainSize, totalSamples));

            var trainScores = new List<double>();
            var valScores = new List<double>();

            // Run CV at this training size
            for (int fold = 0; fold < cvFolds; fold++)
            {
                // Shuffle and split
                var indices = Enumerable.Range(0, totalSamples).OrderBy(_ => random.Next()).ToArray();
                var trainIndices = indices.Take(trainSize).ToArray();
                var valIndices = indices.Skip(trainSize).Take(Math.Min(trainSize, totalSamples - trainSize)).ToArray();

                if (valIndices.Length == 0)
                {
                    // Use part of training for validation if we've used all data
                    valIndices = trainIndices.Skip(trainSize / 2).ToArray();
                    trainIndices = trainIndices.Take(trainSize / 2).ToArray();
                }

                // Extract data
                var trainFeatures = ExtractRows(features, trainIndices);
                var trainTargets = ExtractElements(targets, trainIndices);
                var valFeatures = ExtractRows(features, valIndices);
                var valTargets = ExtractElements(targets, valIndices);

                // Train model
                var model = trainFunc(trainFeatures, trainTargets);

                // Evaluate on training set
                var trainPreds = predictFunc(model, trainFeatures);
                var trainMetrics = isClassification
                    ? _metricEngine.ComputeClassificationMetrics(trainPreds, trainTargets)
                    : _metricEngine.ComputeRegressionMetrics(trainPreds, trainTargets);
                var trainMetric = trainMetrics[metricName];
                if (trainMetric != null)
                    trainScores.Add(NumOps.ToDouble(trainMetric.Value));

                // Evaluate on validation set
                var valPreds = predictFunc(model, valFeatures);
                var valMetrics = isClassification
                    ? _metricEngine.ComputeClassificationMetrics(valPreds, valTargets)
                    : _metricEngine.ComputeRegressionMetrics(valPreds, valTargets);
                var valMetric = valMetrics[metricName];
                if (valMetric != null)
                    valScores.Add(NumOps.ToDouble(valMetric.Value));
            }

            result.TrainingSizes.Add(trainSize);
            result.TrainScoreMeans.Add(trainScores.Average());
            result.TrainScoreStds.Add(StandardDeviation(trainScores));
            result.ValidationScoreMeans.Add(valScores.Average());
            result.ValidationScoreStds.Add(StandardDeviation(valScores));
        }

        result.Diagnosis = DiagnoseLearningCurve(result, higherIsBetter);
        return result;
    }

    private BiasVarianceDiagnosis DiagnoseLearningCurve(LearningCurveResult<T> curve, bool higherIsBetter)
    {
        if (curve.TrainScoreMeans.Count < 2)
            return BiasVarianceDiagnosis.Unknown;

        double finalTrainScore = curve.TrainScoreMeans.Last();
        double finalValScore = curve.ValidationScoreMeans.Last();
        double gap = Math.Abs(finalTrainScore - finalValScore);

        // Heuristics for diagnosis - adjusted based on metric direction
        // For higher-is-better metrics (accuracy): good train score > 0.9, bad val score < 0.7
        // For lower-is-better metrics (error): good train score < 0.1, bad val score > 0.3
        bool goodTrainScore;
        bool badValScore;
        bool valImproving;

        if (higherIsBetter)
        {
            goodTrainScore = finalTrainScore > 0.9;
            badValScore = finalValScore < 0.7;
            valImproving = curve.ValidationScoreMeans.Last() > curve.ValidationScoreMeans[^2];
        }
        else
        {
            // For error metrics, lower is better
            goodTrainScore = finalTrainScore < 0.1;
            badValScore = finalValScore > 0.3;
            valImproving = curve.ValidationScoreMeans.Last() < curve.ValidationScoreMeans[^2];
        }

        bool largeGap = gap > 0.15;
        bool converging = Math.Abs(curve.ValidationScoreMeans.Last() - curve.ValidationScoreMeans[^2]) < 0.02;

        // High variance: training score is good but validation score is bad with large gap
        if (goodTrainScore && badValScore && largeGap)
            return BiasVarianceDiagnosis.HighVariance;

        // High bias: training score is bad (can't even fit training data)
        if (!goodTrainScore && !largeGap)
            return BiasVarianceDiagnosis.HighBias;

        // Needs more data: validation score is still improving
        if (!converging && valImproving)
            return BiasVarianceDiagnosis.NeedsMoreData;

        // Good fit: good training score with small gap to validation
        if (goodTrainScore && !largeGap)
            return BiasVarianceDiagnosis.GoodFit;

        return BiasVarianceDiagnosis.Unknown;
    }

    private static double StandardDeviation(List<double> values)
    {
        if (values.Count <= 1) return 0;
        double mean = values.Average();
        double variance = values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1);
        return Math.Sqrt(variance);
    }

    private static T[,] ExtractRows(T[,] matrix, int[] indices)
    {
        int numCols = matrix.GetLength(1);
        var result = new T[indices.Length, numCols];
        for (int i = 0; i < indices.Length; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = matrix[indices[i], j];
        return result;
    }

    private static T[] ExtractElements(T[] array, int[] indices)
    {
        var result = new T[indices.Length];
        for (int i = 0; i < indices.Length; i++)
            result[i] = array[indices[i]];
        return result;
    }
}

/// <summary>
/// Results from learning curve analysis.
/// </summary>
public class LearningCurveResult<T>
{
    /// <summary>
    /// Name of the metric tracked.
    /// </summary>
    public string MetricName { get; init; } = "";

    /// <summary>
    /// Total number of samples in the dataset.
    /// </summary>
    public int TotalSamples { get; init; }

    /// <summary>
    /// Number of CV folds used at each training size.
    /// </summary>
    public int CVFolds { get; init; }

    /// <summary>
    /// Training sizes evaluated.
    /// </summary>
    public List<int> TrainingSizes { get; init; } = new();

    /// <summary>
    /// Mean training scores at each size.
    /// </summary>
    public List<double> TrainScoreMeans { get; init; } = new();

    /// <summary>
    /// Standard deviation of training scores at each size.
    /// </summary>
    public List<double> TrainScoreStds { get; init; } = new();

    /// <summary>
    /// Mean validation scores at each size.
    /// </summary>
    public List<double> ValidationScoreMeans { get; init; } = new();

    /// <summary>
    /// Standard deviation of validation scores at each size.
    /// </summary>
    public List<double> ValidationScoreStds { get; init; } = new();

    /// <summary>
    /// Diagnosis of bias/variance tradeoff.
    /// </summary>
    public BiasVarianceDiagnosis Diagnosis { get; set; }
}
