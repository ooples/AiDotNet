using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Classification;

/// <summary>
/// Computes classification accuracy: the proportion of correct predictions.
/// </summary>
/// <remarks>
/// <para>
/// Accuracy = (TP + TN) / (TP + TN + FP + FN) = Correct / Total
/// </para>
/// <para>
/// <b>For Beginners:</b> Accuracy is the simplest classification metric - what percentage
/// of predictions were correct? An accuracy of 0.9 means 90% of predictions were right.
/// </para>
/// <para>
/// <b>Limitations:</b> Accuracy can be misleading for imbalanced datasets. If 95% of samples
/// are class A, a model that always predicts A achieves 95% accuracy but is useless.
/// Use balanced accuracy, F1-score, or other metrics for imbalanced data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AccuracyMetric<T> : IClassificationMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "Accuracy";

    /// <inheritdoc/>
    public string Category => "Classification";

    /// <inheritdoc/>
    public string Description => "Proportion of correct predictions out of total predictions.";

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

        int correct = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            if (NumOps.Compare(predictions[i], actuals[i]) == 0)
            {
                correct++;
            }
        }

        return NumOps.FromDouble((double)correct / predictions.Length);
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

        // For accuracy, we can use Wilson score interval or bootstrap
        if (ciMethod == ConfidenceIntervalMethod.WilsonScore)
        {
            var (lower, upper) = ComputeWilsonScoreCI(predictions, actuals, confidenceLevel);
            return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
        }
        else
        {
            var (lower, upper) = ComputeBootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);
            return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
        }
    }

    private (T lower, T upper) ComputeWilsonScoreCI(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, double confidenceLevel)
    {
        int n = predictions.Length;
        if (n == 0)
        {
            return (NumOps.Zero, NumOps.One);
        }

        int correct = 0;
        for (int i = 0; i < n; i++)
        {
            if (NumOps.Compare(predictions[i], actuals[i]) == 0)
            {
                correct++;
            }
        }

        double p = (double)correct / n;
        double z = GetZScore(confidenceLevel);
        double z2 = z * z;

        double denominator = 1 + z2 / n;
        double center = (p + z2 / (2 * n)) / denominator;
        double margin = z * Math.Sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denominator;

        double lower = Math.Max(0, center - margin);
        double upper = Math.Min(1, center + margin);

        return (NumOps.FromDouble(lower), NumOps.FromDouble(upper));
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
            int correct = 0;
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                if (NumOps.Compare(predArray[idx], actualArray[idx]) == 0)
                {
                    correct++;
                }
            }
            bootstrapValues[b] = (double)correct / n;
        }

        Array.Sort(bootstrapValues);

        double alpha = 1 - confidenceLevel;
        int lowerIdx = (int)Math.Floor(alpha / 2 * bootstrapSamples);
        int upperIdx = (int)Math.Ceiling((1 - alpha / 2) * bootstrapSamples) - 1;

        lowerIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, lowerIdx));
        upperIdx = Math.Max(0, Math.Min(bootstrapSamples - 1, upperIdx));

        return (NumOps.FromDouble(bootstrapValues[lowerIdx]), NumOps.FromDouble(bootstrapValues[upperIdx]));
    }

    private static double GetZScore(double confidenceLevel)
    {
        // Common z-scores for confidence levels
        return confidenceLevel switch
        {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => NormalQuantile((1 + confidenceLevel) / 2)
        };
    }

    private static double NormalQuantile(double p)
    {
        // Approximation of inverse normal CDF
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        // Rational approximation for central region
        if (p > 0.5)
        {
            return -NormalQuantile(1 - p);
        }

        double t = Math.Sqrt(-2 * Math.Log(p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t));
    }
}
