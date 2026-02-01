using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes balanced accuracy: the average recall across all classes.
/// </summary>
/// <remarks>
/// <para>
/// Balanced Accuracy = (Sum of per-class recall) / (Number of classes)
/// For binary: Balanced Accuracy = (Sensitivity + Specificity) / 2
/// </para>
/// <para>
/// <b>For Beginners:</b> Balanced accuracy gives equal weight to each class, regardless of size.
/// This makes it much better than regular accuracy for imbalanced datasets. A model that only
/// predicts the majority class will have ~50% balanced accuracy for binary classification,
/// not the misleadingly high regular accuracy.
/// </para>
/// <para>
/// <b>Example:</b> With 95 negative and 5 positive samples:
/// - A model predicting all negative: Accuracy = 95%, Balanced Accuracy = 50%
/// - A model with 90% sensitivity and 90% specificity: Balanced Accuracy = 90%
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BalancedAccuracyMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "BalancedAccuracy";

    /// <inheritdoc/>
    public string Category => "Classification";

    /// <inheritdoc/>
    public string Description => "Average recall across all classes, giving equal weight to each class.";

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

        // Get unique classes
        var classes = new HashSet<double>();
        for (int i = 0; i < actuals.Length; i++)
        {
            classes.Add(NumOps.ToDouble(actuals[i]));
        }

        if (classes.Count == 0)
        {
            return NumOps.Zero;
        }

        // Compute recall for each class
        double sumRecall = 0;
        int validClasses = 0;

        foreach (var cls in classes)
        {
            int truePositives = 0;
            int actualPositives = 0;

            for (int i = 0; i < actuals.Length; i++)
            {
                if (Math.Abs(NumOps.ToDouble(actuals[i]) - cls) < 1e-10)
                {
                    actualPositives++;
                    if (Math.Abs(NumOps.ToDouble(predictions[i]) - cls) < 1e-10)
                    {
                        truePositives++;
                    }
                }
            }

            if (actualPositives > 0)
            {
                sumRecall += (double)truePositives / actualPositives;
                validClasses++;
            }
        }

        if (validClasses == 0)
        {
            return NumOps.Zero;
        }

        return NumOps.FromDouble(sumRecall / validClasses);
    }

    /// <inheritdoc/>
    public MetricWithCI<T> ComputeWithCI(
        ReadOnlySpan<T> predictions,
        ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.BCaBootstrap,
        double confidenceLevel = 0.95,
        int bootstrapSamples = 1000,
        int? randomSeed = null)
    {
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
