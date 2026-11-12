using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes Matthews Correlation Coefficient (MCC): a balanced measure for binary and multi-class classification.
/// </summary>
/// <remarks>
/// <para>
/// For binary classification:
/// MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
/// </para>
/// <para>
/// <b>For Beginners:</b> MCC is one of the best single-number metrics for classification, especially
/// with imbalanced classes. It uses all four quadrants of the confusion matrix and produces a high
/// score only when the model does well on both positives and negatives.
/// </para>
/// <para>
/// <b>Interpretation:</b>
/// <list type="bullet">
/// <item>MCC = +1: Perfect prediction</item>
/// <item>MCC = 0: No better than random</item>
/// <item>MCC = -1: Complete disagreement (predictions are inverted)</item>
/// </list>
/// </para>
/// <para>
/// <b>Advantages over other metrics:</b>
/// <list type="bullet">
/// <item>Works well with imbalanced classes (unlike accuracy)</item>
/// <item>Considers all parts of confusion matrix (unlike F1)</item>
/// <item>Is symmetric: swapping positives/negatives gives same |MCC|</item>
/// </list>
/// </para>
/// <para>
/// <b>Research reference:</b> Matthews (1975), Chicco &amp; Jurman (2020) recommend MCC over F1.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MatthewsCorrelationCoefficientMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _positiveLabel;

    /// <inheritdoc/>
    public string Name => "MatthewsCorrelationCoefficient";

    /// <inheritdoc/>
    public string Category => "Classification";

    /// <inheritdoc/>
    public string Description => "Balanced measure using all confusion matrix quadrants, ranging from -1 to +1.";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.HigherIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.FromDouble(-1.0);

    /// <inheritdoc/>
    public T? MaxValue => NumOps.One;

    /// <inheritdoc/>
    public bool RequiresProbabilities => false;

    /// <inheritdoc/>
    public bool SupportsMultiClass => true;

    /// <summary>
    /// Initializes a new MCC metric with default positive label (1).
    /// </summary>
    public MatthewsCorrelationCoefficientMetric()
    {
        _positiveLabel = NumOps.One;
    }

    /// <summary>
    /// Initializes a new MCC metric with an explicit positive label.
    /// </summary>
    /// <param name="positiveLabel">The label considered positive for binary classification.</param>
    public MatthewsCorrelationCoefficientMetric(T positiveLabel)
    {
        _positiveLabel = positiveLabel;
    }

    /// <inheritdoc/>
    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
        {
            throw new ArgumentException("Predictions and actuals must have the same length.");
        }

        if (predictions.Length == 0)
        {
            return NumOps.Zero;
        }

        // Get unique classes from both predictions and actuals
        var predictionTensor = EvaluationTensorOps<T>.ToTensor(predictions);
        var actualTensor = EvaluationTensorOps<T>.ToTensor(actuals);
        var classes = EvaluationTensorOps<T>.UniqueLabels(actualTensor, predictionTensor);

        if (classes.Length == 2)
        {
            return ComputeBinaryMCC(predictions, actuals);
        }
        else
        {
            return ComputeMultiClassMCC(predictionTensor, actualTensor, classes);
        }
    }

    private T ComputeBinaryMCC(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        var (tp, tn, fp, fn) = EvaluationTensorOps<T>.BinaryConfusion(predictions, actuals, _positiveLabel);

        // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        double numerator = (double)tp * tn - (double)fp * fn;

        double d1 = tp + fp;
        double d2 = tp + fn;
        double d3 = tn + fp;
        double d4 = tn + fn;

        double denominator = Math.Sqrt(d1 * d2 * d3 * d4);

        if (denominator == 0)
        {
            return NumOps.Zero;
        }

        return NumOps.FromDouble(numerator / denominator);
    }

    private static T ComputeMultiClassMCC(Tensor<T> predictions, Tensor<T> actuals, Tensor<T> classes)
    {
        // Compute multi-class MCC using Gorodkin's formula
        // MCC = (c*s - sum(p_k * t_k)) / sqrt((s^2 - sum(p_k^2)) * (s^2 - sum(t_k^2)))
        // where c = sum of correct (diagonal), s = total samples
        // p_k = sum of column k (predicted as k), t_k = sum of row k (actual k)

        int n = predictions.Length;
        double c = EvaluationTensorOps<T>.Count(EvaluationTensorOps<T>.PairEqualMask(predictions, actuals));

        double sumPkTk = 0;
        double sumPk2 = 0;
        double sumTk2 = 0;

        foreach (T cls in classes)
        {
            var counts = EvaluationTensorOps<T>.LabelCounts(predictions, actuals, cls);
            double p = counts.PredictedPositives;
            double t = counts.ActualPositives;
            sumPkTk += p * t;
            sumPk2 += p * p;
            sumTk2 += t * t;
        }

        double s = n;
        double numerator = c * s - sumPkTk;
        double denominator = Math.Sqrt((s * s - sumPk2) * (s * s - sumTk2));

        if (denominator == 0)
        {
            return NumOps.Zero;
        }

        return NumOps.FromDouble(numerator / denominator);
    }

    /// <inheritdoc/>
    public MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null)
    {
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = ComputeBootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);

        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T lower, T upper) ComputeBootstrapCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        int bootstrapSamples,
        double confidenceLevel,
        int? randomSeed)
    {
        int n = predictions.Length;
        if (n == 0)
        {
            return (NumOps.FromDouble(-1), NumOps.One);
        }

        var random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        var bootstrapValues = new List<T>(bootstrapSamples);
        var predArray = predictions.ToArray();
        var actualArray = actuals.ToArray();

        for (int b = 0; b < bootstrapSamples; b++)
        {
            var sampledPred = new T[n];
            var sampledActual = new T[n];

            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sampledPred[i] = predArray[idx];
                sampledActual[i] = actualArray[idx];
            }

            bootstrapValues.Add(Compute(sampledPred, sampledActual));
        }

        double alpha = 1 - confidenceLevel;
        int lowerIdx = (int)Math.Floor(alpha / 2 * bootstrapSamples);
        int upperIdx = (int)Math.Ceiling((1 - alpha / 2) * bootstrapSamples) - 1;

        lowerIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, lowerIdx));
        upperIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, upperIdx));

        return EvaluationTensorOps<T>.OrderStatistics(bootstrapValues, lowerIdx, upperIdx);
    }
}
