using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Results.Core;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics.Regression;

/// <summary>
/// Computes Adjusted R² Score: R² adjusted for the number of predictors in the model.
/// </summary>
/// <remarks>
/// <para>Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)</para>
/// <para><b>For Beginners:</b> Adjusted R² penalizes adding unnecessary predictors to a model.
/// Unlike regular R² which always increases (or stays the same) when adding features,
/// adjusted R² will decrease if the new feature doesn't improve the model enough.
/// <list type="bullet">
/// <item>Use when comparing models with different numbers of features</item>
/// <item>Lower than R² when predictors don't contribute meaningfully</item>
/// <item>Can be negative if the model performs very poorly</item>
/// </list></para>
/// </remarks>
public class AdjustedR2Metric<T> : IRegressionMetric<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numPredictors;

    public string Name => "AdjustedR2";
    public string Category => "Regression";
    public string Description => "R² adjusted for number of predictors.";
    public MetricDirection Direction => MetricDirection.HigherIsBetter;
    public T? MinValue => default; // Can be negative
    public T? MaxValue => NumOps.One;
    public bool RequiresProbabilities => false;
    public bool SupportsMultiClass => false;

    /// <summary>
    /// Initializes the Adjusted R² metric.
    /// </summary>
    /// <param name="numPredictors">Number of predictors (features) in the model. Default is 1.</param>
    public AdjustedR2Metric(int numPredictors = 1)
    {
        _numPredictors = Math.Max(1, numPredictors);
    }

    public T Compute(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals)
    {
        if (predictions.Length != actuals.Length)
            throw new ArgumentException("Predictions and actuals must have the same length.");
        if (predictions.Length == 0) return NumOps.Zero;

        int n = predictions.Length;
        if (n <= _numPredictors + 1) return NumOps.Zero; // Not enough samples

        // Calculate R²
        double sumActuals = 0;
        for (int i = 0; i < n; i++)
            sumActuals += NumOps.ToDouble(actuals[i]);
        double mean = sumActuals / n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double actual = NumOps.ToDouble(actuals[i]);
            double pred = NumOps.ToDouble(predictions[i]);
            double diffRes = actual - pred;
            double diffTot = actual - mean;
            ssRes += diffRes * diffRes;
            ssTot += diffTot * diffTot;
        }

        // If actuals are constant (zero variance), R² is only 1 if predictions are also perfect
        if (Math.Abs(ssTot) < 1e-10)
            return Math.Abs(ssRes) < 1e-10 ? NumOps.One : NumOps.Zero;
        double r2 = 1 - ssRes / ssTot;

        // Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
        double adjustedR2 = 1 - (1 - r2) * (n - 1.0) / (n - _numPredictors - 1.0);
        return NumOps.FromDouble(adjustedR2);
    }

    public MetricWithCI<T> ComputeWithCI(ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals,
        ConfidenceIntervalMethod ciMethod = ConfidenceIntervalMethod.PercentileBootstrap,
        double confidenceLevel = 0.95, int bootstrapSamples = 1000, int? randomSeed = null)
    {
        if (bootstrapSamples < 2)
            throw new ArgumentOutOfRangeException(nameof(bootstrapSamples), "Bootstrap samples must be at least 2.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        var value = Compute(predictions, actuals);
        var (lower, upper) = BootstrapCI(predictions, actuals, bootstrapSamples, confidenceLevel, randomSeed);
        return new MetricWithCI<T>(value, lower, upper, confidenceLevel, ciMethod, Name, Direction);
    }

    private (T, T) BootstrapCI(ReadOnlySpan<T> pred, ReadOnlySpan<T> actual, int samples, double conf, int? seed)
    {
        int n = pred.Length;
        if (n == 0) return (NumOps.Zero, NumOps.One);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        var values = new double[samples];
        var predArr = pred.ToArray(); var actArr = actual.ToArray();
        for (int b = 0; b < samples; b++)
        {
            var sp = new T[n]; var sa = new T[n];
            for (int i = 0; i < n; i++) { int idx = random.Next(n); sp[i] = predArr[idx]; sa[i] = actArr[idx]; }
            values[b] = NumOps.ToDouble(Compute(sp, sa));
        }
        Array.Sort(values);
        double alpha = 1 - conf;
        int lo = Math.Max(0, (int)(alpha / 2 * samples));
        int hi = Math.Min(samples - 1, (int)((1 - alpha / 2) * samples) - 1);
        return (NumOps.FromDouble(values[lo]), NumOps.FromDouble(values[hi]));
    }
}
