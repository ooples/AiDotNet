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
/// <b>Research reference:</b> Matthews (1975), Chicco & Jurman (2020) recommend MCC over F1.
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
    /// Initializes a new MCC metric.
    /// </summary>
    /// <param name="positiveLabel">The label considered positive for binary classification.</param>
    public MatthewsCorrelationCoefficientMetric(T? positiveLabel = default)
    {
        _positiveLabel = positiveLabel ?? NumOps.One;
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
        var classes = new HashSet<double>();
        for (int i = 0; i < actuals.Length; i++)
        {
            classes.Add(NumOps.ToDouble(actuals[i]));
            classes.Add(NumOps.ToDouble(predictions[i]));
        }

        if (classes.Count == 2)
        {
            return ComputeBinaryMCC(predictions, actuals);
        }
        else
        {
            return ComputeMultiClassMCC(predictions, actuals, classes);
        }
    }

    private T ComputeBinaryMCC(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        double positiveLabelValue = NumOps.ToDouble(_positiveLabel);

        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            bool predicted = Math.Abs(NumOps.ToDouble(predictions[i]) - positiveLabelValue) < 1e-10;
            bool actual = Math.Abs(NumOps.ToDouble(actuals[i]) - positiveLabelValue) < 1e-10;

            if (predicted && actual) tp++;
            else if (!predicted && !actual) tn++;
            else if (predicted && !actual) fp++;
            else fn++;
        }

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

    private T ComputeMultiClassMCC(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, HashSet<double> classes)
    {
        // Build confusion matrix with class index lookup for efficiency
        var classList = classes.ToList();
        int k = classList.Count;
        var classIndex = new Dictionary<double, int>(k);
        for (int i = 0; i < k; i++)
        {
            classIndex[classList[i]] = i;
        }

        var confusionMatrix = new int[k, k];

        for (int i = 0; i < predictions.Length; i++)
        {
            int predIdx = classIndex[NumOps.ToDouble(predictions[i])];
            int actualIdx = classIndex[NumOps.ToDouble(actuals[i])];
            confusionMatrix[actualIdx, predIdx]++;
        }

        // Compute multi-class MCC using Gorodkin's formula
        // MCC = (c*s - sum(p_k * t_k)) / sqrt((s^2 - sum(p_k^2)) * (s^2 - sum(t_k^2)))
        // where c = sum of correct (diagonal), s = total samples
        // p_k = sum of column k (predicted as k), t_k = sum of row k (actual k)

        int n = predictions.Length;
        double c = 0; // correct predictions (trace)
        var p = new double[k]; // predicted totals per class
        var t = new double[k]; // actual totals per class

        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                if (i == j)
                {
                    c += confusionMatrix[i, j];
                }
                p[j] += confusionMatrix[i, j];
                t[i] += confusionMatrix[i, j];
            }
        }

        double sumPkTk = 0;
        double sumPk2 = 0;
        double sumTk2 = 0;

        for (int i = 0; i < k; i++)
        {
            sumPkTk += p[i] * t[i];
            sumPk2 += p[i] * p[i];
            sumTk2 += t[i] * t[i];
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

        var bootstrapValues = new double[bootstrapSamples];
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

            bootstrapValues[b] = NumOps.ToDouble(Compute(sampledPred, sampledActual));
        }

        Array.Sort(bootstrapValues);

        double alpha = 1 - confidenceLevel;
        int lowerIdx = (int)Math.Floor(alpha / 2 * bootstrapSamples);
        int upperIdx = (int)Math.Ceiling((1 - alpha / 2) * bootstrapSamples) - 1;

        lowerIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, lowerIdx));
        upperIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, upperIdx));

        return (NumOps.FromDouble(bootstrapValues[lowerIdx]), NumOps.FromDouble(bootstrapValues[upperIdx]));
    }
}
