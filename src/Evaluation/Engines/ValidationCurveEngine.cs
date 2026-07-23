using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Options;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Engine for generating validation curves: how model performance changes with hyperparameter values.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Validation curves show how a hyperparameter affects performance:
/// <list type="bullet">
/// <item>X-axis: Hyperparameter values (e.g., regularization strength)</item>
/// <item>Y-axis: Performance metric (training and validation scores)</item>
/// <item>Helps identify optimal hyperparameter range</item>
/// </list>
/// </para>
/// <para><b>What validation curves tell you:</b>
/// <list type="bullet">
/// <item><b>Underfitting region</b>: Both train and val scores are low</item>
/// <item><b>Optimal region</b>: Val score peaks, train is slightly higher</item>
/// <item><b>Overfitting region</b>: Train high, val declining</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ValidationCurveEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly ValidationCurveOptions? _options;
    private readonly MetricComputationEngine<T> _metricEngine;

    /// <summary>
    /// Initializes the validation curve engine.
    /// </summary>
    public ValidationCurveEngine(ValidationCurveOptions? options = null)
    {
        _options = options;
        _metricEngine = new MetricComputationEngine<T>();
    }

    /// <summary>
    /// Generates a validation curve for a given hyperparameter.
    /// </summary>
    /// <typeparam name="TModel">The model type.</typeparam>
    /// <param name="features">Feature matrix.</param>
    /// <param name="targets">Target array.</param>
    /// <param name="trainFunc">Function to train model with a specific hyperparameter value.</param>
    /// <param name="predictFunc">Function to generate predictions.</param>
    /// <param name="parameterName">Name of the hyperparameter being varied.</param>
    /// <param name="parameterValues">Values to test for the hyperparameter.</param>
    /// <param name="metricName">Primary metric to track.</param>
    /// <param name="cvFolds">Number of cross-validation folds per parameter value.</param>
    /// <param name="isClassification">Whether this is classification.</param>
    /// <param name="higherIsBetter">Whether higher metric values are better. Default true for accuracy-like metrics.</param>
    /// <returns>Validation curve results.</returns>
    public ValidationCurveResult<T> Generate<TModel>(
        T[,] features,
        T[] targets,
        Func<T[,], T[], double, TModel> trainFunc,
        Func<TModel, T[,], T[]> predictFunc,
        string parameterName,
        double[] parameterValues,
        string metricName = "Accuracy",
        int cvFolds = 5,
        bool isClassification = true,
        bool higherIsBetter = true)
    {
        if (parameterValues == null || parameterValues.Length == 0)
            throw new ArgumentException("Parameter values cannot be null or empty.", nameof(parameterValues));
        if (cvFolds < 2)
            throw new ArgumentException("CV folds must be at least 2.", nameof(cvFolds));

        int totalSamples = features.GetLength(0);
        int numFeatures = features.GetLength(1);

        var result = new ValidationCurveResult<T>
        {
            ParameterName = parameterName,
            MetricName = metricName,
            CVFolds = cvFolds
        };

        var random = _options?.RandomSeed.HasValue == true
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        foreach (var paramValue in parameterValues)
        {
            var trainScores = new List<T>();
            var valScores = new List<T>();

            // Run CV at this parameter value
            for (int fold = 0; fold < cvFolds; fold++)
            {
                // Simple K-fold split
                int foldSize = totalSamples / cvFolds;
                int valStart = fold * foldSize;
                int valEnd = (fold == cvFolds - 1) ? totalSamples : (fold + 1) * foldSize;

                var trainIndices = new List<int>();
                var valIndices = new List<int>();

                for (int i = 0; i < totalSamples; i++)
                {
                    if (i >= valStart && i < valEnd)
                        valIndices.Add(i);
                    else
                        trainIndices.Add(i);
                }

                // Extract data
                var trainFeatures = ExtractRows(features, trainIndices.ToArray());
                var trainTargets = ExtractElements(targets, trainIndices.ToArray());
                var valFeatures = ExtractRows(features, valIndices.ToArray());
                var valTargets = ExtractElements(targets, valIndices.ToArray());

                // Train model with parameter value
                var model = trainFunc(trainFeatures, trainTargets, paramValue);

                // Evaluate on training set
                var trainPreds = predictFunc(model, trainFeatures);
                var trainMetrics = isClassification
                    ? _metricEngine.ComputeClassificationMetrics(trainPreds, trainTargets)
                    : _metricEngine.ComputeRegressionMetrics(trainPreds, trainTargets);
                var trainMetric = trainMetrics[metricName];
                if (trainMetric != null)
                    trainScores.Add(trainMetric.Value);

                // Evaluate on validation set
                var valPreds = predictFunc(model, valFeatures);
                var valMetrics = isClassification
                    ? _metricEngine.ComputeClassificationMetrics(valPreds, valTargets)
                    : _metricEngine.ComputeRegressionMetrics(valPreds, valTargets);
                var valMetric = valMetrics[metricName];
                if (valMetric != null)
                    valScores.Add(valMetric.Value);
            }

            result.ParameterValues.Add(paramValue);
            result.TrainScoreMeans.Add(Mean(trainScores));
            result.TrainScoreStds.Add(StandardDeviation(trainScores));
            result.ValidationScoreMeans.Add(Mean(valScores));
            result.ValidationScoreStds.Add(StandardDeviation(valScores));
        }

        result.OptimalParameterValue = FindOptimalParameter(result, higherIsBetter);
        return result;
    }

    private double FindOptimalParameter(ValidationCurveResult<T> curve, bool higherIsBetter)
    {
        if (curve.ParameterValues.Count == 0) return 0;

        // Find parameter with best validation score (respecting metric direction)
        int bestIdx = 0;
        T bestScore = curve.ValidationScoreMeans[0];

        for (int i = 1; i < curve.ValidationScoreMeans.Count; i++)
        {
            bool isBetter = higherIsBetter
                ? NumOps.GreaterThan(curve.ValidationScoreMeans[i], bestScore)
                : NumOps.LessThan(curve.ValidationScoreMeans[i], bestScore);

            if (isBetter)
            {
                bestScore = curve.ValidationScoreMeans[i];
                bestIdx = i;
            }
        }

        return curve.ParameterValues[bestIdx];
    }

    private static T Mean(List<T> values)
    {
        if (values.Count == 0)
            throw new InvalidOperationException("Cannot compute mean of an empty sequence. No fold scores were recorded for this parameter value.");
        T sum = NumOps.Zero;
        foreach (var v in values)
        {
            sum = NumOps.Add(sum, v);
        }
        return NumOps.Divide(sum, NumOps.FromDouble(values.Count));
    }

    private static T StandardDeviation(List<T> values)
    {
        if (values.Count <= 1) return NumOps.Zero;
        T mean = Mean(values);
        T variance = NumOps.Zero;
        foreach (var v in values)
        {
            T diff = NumOps.Subtract(v, mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(values.Count - 1));
        return NumOps.Sqrt(variance);
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
/// Results from validation curve analysis.
/// </summary>
public class ValidationCurveResult<T>
{
    /// <summary>
    /// Name of the hyperparameter varied.
    /// </summary>
    public string ParameterName { get; init; } = "";

    /// <summary>
    /// Name of the metric tracked.
    /// </summary>
    public string MetricName { get; init; } = "";

    /// <summary>
    /// Number of CV folds used at each parameter value.
    /// </summary>
    public int CVFolds { get; init; }

    /// <summary>
    /// Hyperparameter values tested.
    /// </summary>
    public List<double> ParameterValues { get; init; } = new();

    /// <summary>
    /// Mean training scores at each parameter value.
    /// </summary>
    public List<T> TrainScoreMeans { get; init; } = new();

    /// <summary>
    /// Standard deviation of training scores.
    /// </summary>
    public List<T> TrainScoreStds { get; init; } = new();

    /// <summary>
    /// Mean validation scores at each parameter value.
    /// </summary>
    public List<T> ValidationScoreMeans { get; init; } = new();

    /// <summary>
    /// Standard deviation of validation scores.
    /// </summary>
    public List<T> ValidationScoreStds { get; init; } = new();

    /// <summary>
    /// Optimal parameter value based on validation score.
    /// </summary>
    public double OptimalParameterValue { get; set; }
}
