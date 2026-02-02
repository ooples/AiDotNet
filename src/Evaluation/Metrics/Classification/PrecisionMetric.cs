using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes precision (positive predictive value): the proportion of positive predictions that are correct.
/// </summary>
/// <remarks>
/// <para>
/// Precision = TP / (TP + FP) = True Positives / Predicted Positives
/// </para>
/// <para>
/// <b>For Beginners:</b> Precision answers: "When the model predicts positive, how often is it correct?"
/// High precision means few false positives (false alarms). A spam filter with 99% precision
/// means only 1% of emails it flags as spam are actually legitimate.
/// </para>
/// <para>
/// <b>When to prioritize precision:</b>
/// <list type="bullet">
/// <item>When false positives are costly (blocking legitimate transactions, accusing innocent people)</item>
/// <item>When you need high confidence in positive predictions</item>
/// </list>
/// </para>
/// <para>
/// <b>Multi-class:</b> For multi-class problems, use averaging (micro, macro, weighted) to combine
/// per-class precision values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PrecisionMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _positiveLabel;
    private readonly AveragingMethod _averaging;

    /// <inheritdoc/>
    public string Name => _averaging == AveragingMethod.None ? "Precision" : $"Precision_{_averaging}";

    /// <inheritdoc/>
    public string Category => "Classification";

    /// <inheritdoc/>
    public string Description => "Proportion of positive predictions that are correct (TP / (TP + FP)).";

    /// <inheritdoc/>
    public MetricDirection Direction => MetricDirection.HigherIsBetter;

    /// <inheritdoc/>
    public T? MinValue => NumOps.Zero;

    /// <inheritdoc/>
    public T? MaxValue => NumOps.One;

    /// <inheritdoc/>
    public bool RequiresProbabilities => false;

    /// <inheritdoc/>
    public bool SupportsMultiClass => true;

    /// <summary>
    /// Initializes a new precision metric.
    /// </summary>
    /// <param name="positiveLabel">The label considered positive for binary classification.</param>
    /// <param name="averaging">Averaging method for multi-class.</param>
    public PrecisionMetric(T? positiveLabel = default, AveragingMethod averaging = AveragingMethod.Binary)
    {
        _positiveLabel = positiveLabel ?? NumOps.One;
        _averaging = averaging;
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

        if (_averaging == AveragingMethod.Binary || _averaging == AveragingMethod.None)
        {
            return ComputeBinaryPrecision(predictions, actuals, _positiveLabel);
        }

        // Get unique classes from both predictions and actuals
        var classes = GetUniqueClasses(predictions, actuals);

        if (_averaging == AveragingMethod.Micro)
        {
            return ComputeMicroPrecision(predictions, actuals, classes);
        }

        // Macro or Weighted averaging
        double sumPrecision = 0;
        double sumWeights = 0;

        foreach (var cls in classes)
        {
            var precision = NumOps.ToDouble(ComputeBinaryPrecision(predictions, actuals, NumOps.FromDouble(cls)));
            int classCount = CountClass(actuals, cls);

            if (_averaging == AveragingMethod.Macro)
            {
                sumPrecision += precision;
                sumWeights += 1;
            }
            else // Weighted
            {
                sumPrecision += precision * classCount;
                sumWeights += classCount;
            }
        }

        return sumWeights > 0 ? NumOps.FromDouble(sumPrecision / sumWeights) : NumOps.Zero;
    }

    private T ComputeBinaryPrecision(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, T positiveLabel)
    {
        int truePositives = 0;
        int predictedPositives = 0;
        double positiveLabelValue = NumOps.ToDouble(positiveLabel);

        for (int i = 0; i < predictions.Length; i++)
        {
            if (Math.Abs(NumOps.ToDouble(predictions[i]) - positiveLabelValue) < 1e-10)
            {
                predictedPositives++;
                if (Math.Abs(NumOps.ToDouble(actuals[i]) - positiveLabelValue) < 1e-10)
                {
                    truePositives++;
                }
            }
        }

        return predictedPositives > 0 ? NumOps.FromDouble((double)truePositives / predictedPositives) : NumOps.Zero;
    }

    private T ComputeMicroPrecision(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, HashSet<double> classes)
    {
        int totalTruePositives = 0;
        int totalPredictedPositives = 0;

        foreach (var cls in classes)
        {
            for (int i = 0; i < predictions.Length; i++)
            {
                if (Math.Abs(NumOps.ToDouble(predictions[i]) - cls) < 1e-10)
                {
                    totalPredictedPositives++;
                    if (Math.Abs(NumOps.ToDouble(actuals[i]) - cls) < 1e-10)
                    {
                        totalTruePositives++;
                    }
                }
            }
        }

        return totalPredictedPositives > 0 ? NumOps.FromDouble((double)totalTruePositives / totalPredictedPositives) : NumOps.Zero;
    }

    private static HashSet<double> GetUniqueClasses(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        var classes = new HashSet<double>();
        for (int i = 0; i < predictions.Length; i++)
        {
            classes.Add(NumOps.ToDouble(predictions[i]));
        }
        for (int i = 0; i < actuals.Length; i++)
        {
            classes.Add(NumOps.ToDouble(actuals[i]));
        }
        return classes;
    }

    private static int CountClass(ReadOnlySpan<T> values, double cls)
    {
        int count = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (Math.Abs(NumOps.ToDouble(values[i]) - cls) < 1e-10)
            {
                count++;
            }
        }
        return count;
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
            return (NumOps.Zero, NumOps.One);
        }

        var random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : new Random();

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
