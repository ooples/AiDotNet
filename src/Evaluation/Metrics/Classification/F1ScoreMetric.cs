using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes F1 score: the harmonic mean of precision and recall.
/// </summary>
/// <remarks>
/// <para>
/// F1 = 2 * (Precision * Recall) / (Precision + Recall)
/// </para>
/// <para>
/// <b>For Beginners:</b> F1 score balances precision and recall into a single number.
/// It's particularly useful when:
/// <list type="bullet">
/// <item>You care about both false positives and false negatives</item>
/// <item>Classes are imbalanced (where accuracy would be misleading)</item>
/// <item>You need a single number to compare models</item>
/// </list>
/// </para>
/// <para>
/// <b>Interpretation:</b>
/// <list type="bullet">
/// <item>F1 = 1.0: Perfect precision and recall</item>
/// <item>F1 = 0.5: Mediocre balance</item>
/// <item>F1 near 0: Poor performance on at least one of precision or recall</item>
/// </list>
/// </para>
/// <para>
/// <b>Note:</b> F1 is the harmonic mean because it penalizes extreme differences.
/// If precision = 0.95 and recall = 0.1, F1 = 0.18 (not 0.525 arithmetic mean).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class F1ScoreMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _positiveLabel;
    private readonly AveragingMethod _averaging;

    /// <inheritdoc/>
    public string Name => _averaging == AveragingMethod.None ? "F1Score" : $"F1Score_{_averaging}";

    /// <inheritdoc/>
    public string Category => "Classification";

    /// <inheritdoc/>
    public string Description => "Harmonic mean of precision and recall, balancing both metrics.";

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
    /// Initializes a new F1 score metric.
    /// </summary>
    /// <param name="positiveLabel">The label considered positive for binary classification.</param>
    /// <param name="averaging">Averaging method for multi-class.</param>
    public F1ScoreMetric(T? positiveLabel = default, AveragingMethod averaging = AveragingMethod.Binary)
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
            return ComputeBinaryF1(predictions, actuals, _positiveLabel);
        }

        // Get unique classes from both predictions and actuals
        var classes = GetUniqueClasses(predictions, actuals);

        if (_averaging == AveragingMethod.Micro)
        {
            return ComputeMicroF1(predictions, actuals, classes);
        }

        // Macro or Weighted averaging
        double sumF1 = 0;
        double sumWeights = 0;

        foreach (var cls in classes)
        {
            var f1 = NumOps.ToDouble(ComputeBinaryF1(predictions, actuals, NumOps.FromDouble(cls)));
            int classCount = CountClass(actuals, cls);

            if (_averaging == AveragingMethod.Macro)
            {
                sumF1 += f1;
                sumWeights += 1;
            }
            else // Weighted
            {
                sumF1 += f1 * classCount;
                sumWeights += classCount;
            }
        }

        return sumWeights > 0 ? NumOps.FromDouble(sumF1 / sumWeights) : NumOps.Zero;
    }

    private T ComputeBinaryF1(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, T positiveLabel)
    {
        int truePositives = 0;
        int predictedPositives = 0;
        int actualPositives = 0;
        double positiveLabelValue = NumOps.ToDouble(positiveLabel);

        for (int i = 0; i < predictions.Length; i++)
        {
            bool predicted = Math.Abs(NumOps.ToDouble(predictions[i]) - positiveLabelValue) < 1e-10;
            bool actual = Math.Abs(NumOps.ToDouble(actuals[i]) - positiveLabelValue) < 1e-10;

            if (predicted) predictedPositives++;
            if (actual) actualPositives++;
            if (predicted && actual) truePositives++;
        }

        if (predictedPositives == 0 && actualPositives == 0)
        {
            return NumOps.One; // Perfect case: no positives and none predicted
        }

        double precision = predictedPositives > 0 ? (double)truePositives / predictedPositives : 0;
        double recall = actualPositives > 0 ? (double)truePositives / actualPositives : 0;

        if (precision + recall == 0)
        {
            return NumOps.Zero;
        }

        return NumOps.FromDouble(2 * precision * recall / (precision + recall));
    }

    private T ComputeMicroF1(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, HashSet<double> classes)
    {
        int totalTruePositives = 0;
        int totalPredictedPositives = 0;
        int totalActualPositives = 0;

        foreach (var cls in classes)
        {
            for (int i = 0; i < predictions.Length; i++)
            {
                bool predicted = Math.Abs(NumOps.ToDouble(predictions[i]) - cls) < 1e-10;
                bool actual = Math.Abs(NumOps.ToDouble(actuals[i]) - cls) < 1e-10;

                if (predicted) totalPredictedPositives++;
                if (actual) totalActualPositives++;
                if (predicted && actual) totalTruePositives++;
            }
        }

        double precision = totalPredictedPositives > 0 ? (double)totalTruePositives / totalPredictedPositives : 0;
        double recall = totalActualPositives > 0 ? (double)totalTruePositives / totalActualPositives : 0;

        if (precision + recall == 0)
        {
            return NumOps.Zero;
        }

        return NumOps.FromDouble(2 * precision * recall / (precision + recall));
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
